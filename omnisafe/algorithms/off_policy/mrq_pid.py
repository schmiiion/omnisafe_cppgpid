"""MR.Q representation learning and PID Lagrangian control in OmniSafe.

Based on the update design in Fujimoto et al., *Towards General-Purpose
Model-Free Reinforcement Learning*. The cost Q-function is the CMDP extension.
All MR.Q-specific models and updates are kept in this module.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy

import torch
from gymnasium.spaces import Box
from torch import nn
from torch.nn import functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.common.buffer import VectorMRQReplayBuffer
from omnisafe.common.logger import Logger
from omnisafe.common.pid_lagrange import PIDLagrangian


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(module.bias)


class _MLP(nn.Module):
    """Three-layer MLP with normalized hidden activations."""

    def __init__(self, input_dim: int, output_dim: int, width: int, activation: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, width), nn.Linear(width, width), nn.Linear(width, output_dim)],
        )
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(F.layer_norm(x, (x.shape[-1],)))
        return self.layers[-1](x)


class _Encoder(nn.Module):
    """State/action embeddings and one-step latent/reward/done predictors."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: object) -> None:
        super().__init__()
        self.state_net = _MLP(obs_dim, cfg.zs_dim, cfg.enc_hdim, cfg.enc_activ)
        self.action_net = nn.Linear(act_dim, cfg.za_dim)
        self.joint_net = _MLP(cfg.zs_dim + cfg.za_dim, cfg.zsa_dim, cfg.enc_hdim, cfg.enc_activ)
        self.dynamics_net = nn.Linear(cfg.zsa_dim, 1 + cfg.zs_dim + cfg.num_bins)
        self.cost_net = nn.Linear(cfg.zsa_dim, cfg.num_bins)
        self.activation = getattr(F, cfg.enc_activ)
        self.zs_dim = cfg.zs_dim

    def state(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.state_net(obs.flatten(start_dim=1))
        return self.activation(F.layer_norm(z, (z.shape[-1],)))

    def state_action(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        za = self.activation(self.action_net(action.flatten(start_dim=1)))
        return self.joint_net(torch.cat((zs, za), dim=-1))

    def predict(self, zs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        zsa = self.state_action(zs, action)
        predicted = self.dynamics_net(zsa)
        return (
            predicted[:, :1],
            predicted[:, 1 : 1 + self.zs_dim],
            predicted[:, 1 + self.zs_dim :],
            self.cost_net(zsa),
        )


class _Policy(nn.Module):
    def __init__(self, cfg: object, act_dim: int) -> None:
        super().__init__()
        self.net = _MLP(cfg.zs_dim, act_dim, cfg.policy_hdim, 'relu')

    def forward(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_activation = self.net(zs)
        return torch.tanh(pre_activation), pre_activation


class _TwinQ(nn.Module):
    def __init__(self, cfg: object) -> None:
        super().__init__()
        activations = {'elu': nn.ELU, 'relu': nn.ReLU, 'gelu': nn.GELU}
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    _MLP(cfg.zsa_dim, cfg.value_hdim, cfg.value_hdim, cfg.value_activ),
                    nn.LayerNorm(cfg.value_hdim),
                    activations[cfg.value_activ](),
                    nn.Linear(cfg.value_hdim, 1),
                )
                for _ in range(2)
            ],
        )

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(zsa) for head in self.heads], dim=-1)


class _MRQActorCritic(nn.Module):
    """Policy object accepted by the OmniSafe rollout/evaluation adapters."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: object) -> None:
        super().__init__()
        self.encoder = _Encoder(obs_dim, act_dim, cfg)
        self.actor = _Policy(cfg, act_dim)
        self.reward_critic = _TwinQ(cfg)
        self.cost_critic = _TwinQ(cfg)
        self.exploration_noise = cfg.exploration_noise if cfg.use_exploration_noise else 0.0
        self.apply(_initialize_linear)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.actor(self.encoder.state(obs))
        if not deterministic:
            action = action + torch.randn_like(action) * self.exploration_noise
        return action.clamp(-1.0, 1.0)

    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.step(obs, deterministic=deterministic)


class _TwoHot:
    """Interpolate target mass between bins uniform in symlog space."""

    def __init__(self, cfg: object, device: torch.device) -> None:
        positions = torch.linspace(cfg.lower, cfg.upper, cfg.num_bins, device=device)
        self.bins = positions.sign() * torch.expm1(positions.abs())

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1).contiguous()
        upper = torch.searchsorted(self.bins, target).clamp(1, self.bins.numel() - 1)
        lower = upper - 1
        fraction = ((target - self.bins[lower]) / (self.bins[upper] - self.bins[lower])).clamp(0, 1)
        distribution = torch.zeros_like(logits)
        distribution.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
        distribution.scatter_add_(1, upper[:, None], fraction[:, None])
        return -(distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


@registry.register
class MRQPID(DDPG):
    """MR.Q with a three-step cost critic and PID-controlled Lagrangian."""

    def _init_model(self) -> None:
        obs_space, act_space = self._env.observation_space, self._env.action_space
        if not isinstance(obs_space, Box) or not isinstance(act_space, Box):
            raise NotImplementedError('MRQPID requires continuous Box spaces')
        if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
            raise NotImplementedError('MRQPID supports vector observations and actions')
        cfg = self._cfgs.algo_cfgs
        self._actor_critic = _MRQActorCritic(
            obs_space.shape[0], act_space.shape[0], cfg,
        ).to(self._device)
        self._target_encoder = deepcopy(self._actor_critic.encoder).requires_grad_(False)
        self._target_policy = deepcopy(self._actor_critic.actor).requires_grad_(False)
        self._target_reward_critic = deepcopy(
            self._actor_critic.reward_critic,
        ).requires_grad_(False)
        self._target_cost_critic = deepcopy(self._actor_critic.cost_critic).requires_grad_(False)
        self._encoder_optimizer = torch.optim.AdamW(
            self._actor_critic.encoder.parameters(), lr=cfg.enc_lr, weight_decay=cfg.enc_wd,
        )
        self._policy_optimizer = torch.optim.AdamW(
            self._actor_critic.actor.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_wd,
        )
        self._reward_optimizer = torch.optim.AdamW(
            self._actor_critic.reward_critic.parameters(),
            lr=cfg.value_lr,
            weight_decay=cfg.value_wd,
        )
        self._cost_optimizer = torch.optim.AdamW(
            self._actor_critic.cost_critic.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_wd,
        )
        self._two_hot = _TwoHot(cfg, self._device)

    def _init(self) -> None:
        cfg = self._cfgs.algo_cfgs
        if cfg.start_learning_steps >= cfg.size:
            raise ValueError('MR.Q warmup must be shorter than replay-buffer capacity')
        if cfg.target_update_freq < 1 or cfg.update_iters < 1:
            raise ValueError('MR.Q update frequencies must be positive')
        self._buf = VectorMRQReplayBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=cfg.size,
            batch_size=cfg.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            horizon=max(cfg.enc_horizon, cfg.Q_horizon),
            alpha=cfg.alpha,
            min_priority=cfg.min_priority,
            device=self._device,
        )
        self._lagrange = PIDLagrangian(**self._cfgs.lagrange_cfgs)
        self._training_steps = 0
        self._environment_steps = 0
        self._reward_scale = 1.0
        self._target_reward_scale = 0.0
        self._policy_scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                self._policy_optimizer, start_factor=1.0, end_factor=1e-8,
                total_iters=self._epochs,
            )
            if self._cfgs.model_cfgs.linear_lr_decay else None
        )

    def _init_log(self) -> None:
        cfg = self._cfgs
        self._logger = Logger(
            output_dir=cfg.logger_cfgs.log_dir,
            exp_name=cfg.exp_name,
            seed=cfg.seed,
            use_tensorboard=cfg.logger_cfgs.use_tensorboard,
            use_wandb=cfg.logger_cfgs.use_wandb,
            config=cfg,
        )
        saved = {
            'pi': self._actor_critic,
            'target_encoder': self._target_encoder,
            'target_policy': self._target_policy,
            'target_reward_critic': self._target_reward_critic,
            'target_cost_critic': self._target_cost_critic,
            'lagrangian_multiplier': self._lagrange.lagrangian_multiplier,
        }
        if cfg.algo_cfgs.obs_normalize:
            saved['obs_normalizer'] = self._env.save()['obs_normalizer']
        self._logger.setup_torch_saver(saved)
        self._logger.torch_save()
        for key in ('Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen'):
            self._logger.register_key(key, window_length=cfg.logger_cfgs.window_lens)
        if cfg.train_cfgs.eval_episodes > 0:
            for key in ('Metrics/TestEpRet', 'Metrics/TestEpCost', 'Metrics/TestEpLen'):
                self._logger.register_key(key, window_length=cfg.logger_cfgs.window_lens)
        for key in (
            'Loss/Encoder', 'Loss/Loss_reward_critic', 'Loss/Loss_cost_critic',
            'Loss/Loss_pi', 'Value/reward_critic', 'Value/cost_critic',
            'Metrics/LagrangeMultiplier', 'Train/Epoch', 'Train/LR',
            'Train/RewardScale', 'Train/TargetRewardScale', 'TotalEnvSteps',
            'Time/Total', 'Time/Rollout', 'Time/Update', 'Time/Evaluate',
            'Time/Epoch', 'Time/FPS',
        ):
            self._logger.register_key(key)
        for key in self._env.env_spec_keys:
            self._logger.register_key(key)

    def learn(self) -> tuple[float, float, float]:
        """Collect experience and run all MR.Q updates using OmniSafe adapters."""
        self._logger.log('INFO: Start MR.Q-PID training')
        start_time = time.time()
        for epoch in range(self._epochs):
            self._epoch = epoch
            epoch_start = time.time()
            rollout_time = 0.0
            update_time = 0.0
            for _ in range(self._samples_per_epoch):
                rollout_start = time.time()
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(self._buf.size < self._cfgs.algo_cfgs.start_learning_steps),
                )
                self._environment_steps += (
                    self._update_cycle * self._cfgs.train_cfgs.vector_env_nums
                )
                rollout_time += time.time() - rollout_start
                update_start = time.time()
                replay_ready = (
                    self._buf.size > self._cfgs.algo_cfgs.start_learning_steps
                    and bool(self._buf.valid_starts.any())
                )
                if replay_ready:
                    for _ in range(self._cfgs.algo_cfgs.update_iters):
                        self._update()
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            # Update the PID controller from observed episode cost once per logging epoch.
            if (
                self._training_steps
                and epoch >= self._cfgs.algo_cfgs.warmup_epochs
                and self._logger._data['Metrics/EpCost']
            ):
                mean_cost = self._logger.get_stats('Metrics/EpCost')[0]
                if math.isfinite(mean_cost):
                    self._lagrange.pid_update(mean_cost)
            self._logger.store({
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            })
            eval_start = time.time()
            if self._cfgs.train_cfgs.eval_episodes > 0:
                self._env.eval_policy(
                    episode=self._cfgs.train_cfgs.eval_episodes,
                    agent=self._actor_critic,
                    logger=self._logger,
                )
            eval_time = time.time() - eval_start
            if self._policy_scheduler is not None and self._training_steps:
                self._policy_scheduler.step()
            elapsed = max(time.time() - epoch_start, 1e-8)
            self._logger.store({
                'Train/Epoch': epoch,
                'Train/LR': self._policy_optimizer.param_groups[0]['lr'],
                'Train/RewardScale': self._reward_scale,
                'Train/TargetRewardScale': self._target_reward_scale,
                'TotalEnvSteps': self._environment_steps,
                'Time/Total': time.time() - start_time,
                'Time/Rollout': rollout_time,
                'Time/Update': update_time,
                'Time/Evaluate': eval_time,
                'Time/Epoch': elapsed,
                'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / elapsed,
            })
            self._logger.dump_tabular()
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger._what_to_save['lagrangian_multiplier'] = (
                    self._lagrange.lagrangian_multiplier
                )
                self._logger.torch_save()

        result = tuple(
            self._logger.get_stats(key)[0]
            for key in ('Metrics/EpRet', 'Metrics/EpCost', 'Metrics/EpLen')
        )
        self._logger.close()
        self._env.close()
        return result

    def _log_when_not_update(self) -> None:
        self._logger.store({
            'Loss/Encoder': 0.0,
            'Loss/Loss_reward_critic': 0.0,
            'Loss/Loss_cost_critic': 0.0,
            'Loss/Loss_pi': 0.0,
            'Value/reward_critic': 0.0,
            'Value/cost_critic': 0.0,
        })

    def _snapshot_targets(self) -> None:
        for source, target in (
            (self._actor_critic.encoder, self._target_encoder),
            (self._actor_critic.actor, self._target_policy),
            (self._actor_critic.reward_critic, self._target_reward_critic),
            (self._actor_critic.cost_critic, self._target_cost_critic),
        ):
            target.load_state_dict(source.state_dict())
        self._target_reward_scale = self._reward_scale
        self._reward_scale = self._buf.reward_scale()

    def _update(self) -> None:
        """Periodic encoder phase, followed by one reward/cost Q and policy step."""
        cfg = self._cfgs.algo_cfgs
        self._training_steps += 1
        if (self._training_steps - 1) % cfg.target_update_freq == 0:
            self._snapshot_targets()
            for _ in range(cfg.target_update_freq):
                self._update_encoder(self._buf.sample_sequences(cfg.enc_horizon))
        self._update_rl(self._buf.sample_sequences(cfg.Q_horizon))

    def _update_encoder(self, batch: dict[str, torch.Tensor]) -> None:
        cfg = self._cfgs.algo_cfgs
        encoder = self._actor_critic.encoder
        next_obs = batch['next_obs']
        with torch.no_grad():
            target_zs = self._target_encoder.state(next_obs.reshape(-1, next_obs.shape[-1]))
            target_zs = target_zs.reshape(next_obs.shape[0], cfg.enc_horizon, -1)
        zs = encoder.state(batch['obs'][:, 0])
        loss = torch.zeros((), device=self._device)
        for index in range(cfg.enc_horizon):
            mask = batch['loss_mask'][:, index, None]
            done_pred, zs_pred, reward_pred, cost_pred = encoder.predict(zs, batch['act'][:, index])
            loss = loss + cfg.dyn_weight * ((zs_pred - target_zs[:, index]).square() * mask).mean()
            loss = loss + cfg.reward_weight * (
                self._two_hot.loss(reward_pred, batch['reward'][:, index]) * mask
            ).mean()
            if self._buf.env_terminates:
                loss = loss + cfg.done_weight * (
                    (done_pred - batch['terminated'][:, index, None].float()).square() * mask
                ).mean()
            if cfg.cost_prediction_weight > 0:
                loss = loss + cfg.cost_prediction_weight * (
                    self._two_hot.loss(cost_pred, batch['cost'][:, index]) * mask
                ).mean()
            zs = zs_pred
        self._encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._encoder_optimizer.step()
        self._logger.store({'Loss/Encoder': loss.item()})

    @staticmethod
    def _n_step_return(
        signal: torch.Tensor, terminated: torch.Tensor, gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Discounted observed return and the bootstrap coefficient."""
        total = torch.zeros((signal.shape[0], 1), device=signal.device)
        discount = torch.ones_like(total)
        for index in range(signal.shape[1]):
            total = total + discount * signal[:, index, None]
            discount = discount * gamma * (~terminated[:, index, None]).float()
        return total, discount

    def _update_rl(self, batch: dict[str, torch.Tensor]) -> None:
        cfg = self._cfgs.algo_cfgs
        obs, action = batch['obs'][:, 0], batch['act'][:, 0]
        reward, reward_discount = self._n_step_return(
            batch['reward'], batch['terminated'], cfg.gamma,
        )
        cost, cost_discount = self._n_step_return(
            batch['cost'], batch['terminated'], cfg.get('cost_gamma', cfg.gamma),
        )
        with torch.no_grad():
            next_zs = self._target_encoder.state(batch['next_obs'][:, -1])
            next_action, _ = self._target_policy(next_zs)
            noise = (torch.randn_like(next_action) * cfg.target_policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip,
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_zsa = self._target_encoder.state_action(next_zs, next_action)
            next_reward_q = self._target_reward_critic(target_zsa).min(dim=1, keepdim=True).values
            next_cost_q = self._target_cost_critic(target_zsa).max(dim=1, keepdim=True).values
            reward_target = (
                reward + reward_discount * next_reward_q * self._target_reward_scale
            ) / self._reward_scale
            cost_target = cost + cost_discount * next_cost_q
            zs = self._actor_critic.encoder.state(obs)
            zsa = self._actor_critic.encoder.state_action(zs, action)

        reward_q = self._actor_critic.reward_critic(zsa)
        cost_q = self._actor_critic.cost_critic(zsa)
        reward_loss = F.smooth_l1_loss(reward_q, reward_target.expand_as(reward_q))
        cost_loss = F.smooth_l1_loss(cost_q, cost_target.expand_as(cost_q))
        self._reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._actor_critic.reward_critic.parameters(), cfg.value_grad_clip,
        )
        self._reward_optimizer.step()
        self._cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._actor_critic.cost_critic.parameters(), cfg.value_grad_clip,
        )
        self._cost_optimizer.step()

        reward_error = (reward_q.detach() - reward_target).abs().amax(dim=1) * self._reward_scale
        cost_error = (cost_q.detach() - cost_target).abs().amax(dim=1)
        self._buf.update_priorities(
            batch['indices'], torch.maximum(reward_error, cfg.cost_priority_weight * cost_error),
        )

        frozen = (
            self._actor_critic.encoder,
            self._actor_critic.reward_critic,
            self._actor_critic.cost_critic,
        )
        for module in frozen:
            module.requires_grad_(False)
        try:
            policy_action, pre_activation = self._actor_critic.actor(zs)
            policy_zsa = self._actor_critic.encoder.state_action(zs, policy_action)
            policy_reward = self._actor_critic.reward_critic(policy_zsa).mean(dim=1)
            policy_cost = self._actor_critic.cost_critic(policy_zsa).max(dim=1).values
            multiplier = self._lagrange.lagrangian_multiplier
            policy_loss = (
                -self._reward_scale * policy_reward.mean() + multiplier * policy_cost.mean()
            ) / (1 + multiplier) + cfg.pre_activ_weight * pre_activation.square().mean()
            self._policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self._policy_optimizer.step()
        finally:
            for module in frozen:
                module.requires_grad_(True)
        self._logger.store({
            'Loss/Loss_reward_critic': reward_loss.item(),
            'Loss/Loss_cost_critic': cost_loss.item(),
            'Loss/Loss_pi': policy_loss.item(),
            'Value/reward_critic': reward_q.mean().item(),
            'Value/cost_critic': cost_q.mean().item(),
        })
