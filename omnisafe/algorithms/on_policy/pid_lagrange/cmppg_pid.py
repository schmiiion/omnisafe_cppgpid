"""Constrained model-phasic policy gradient with a PID Lagrangian."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.algorithms.on_policy.pid_lagrange.cppg_pid import CPPGPID
from omnisafe.common.buffer.vector_model_ppg_buffer import VectorModelPPGBuffer
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer


@registry.register
class CMPPGPID(CPPGPID):
    """Train a shared policy/model encoder with model and critic-value objectives.

    Policy phases retain PPO's separate reward/cost critics and GAE targets. The
    auxiliary phase trains latent dynamics, observed reward/cost predictions, and
    actor value heads matched to phase-stable critic predictions. A KL budget
    constrains policy drift during auxiliary updates; the critics stay fixed.
    """

    _AUX_COMPONENTS = ('Latent', 'Reward', 'Cost', 'ValueReward', 'ValueCost', 'KL')

    def _init(self) -> None:
        cfg = self._cfgs.algo_cfgs
        phase_steps = cfg.N_pi * cfg.steps_per_epoch
        if cfg.N_pi < 1 or self._cfgs.train_cfgs.total_steps % phase_steps:
            raise ValueError(
                'CMPPGPID needs whole phases: total_steps divisible by N_pi * steps_per_epoch',
            )
        if cfg.number_aux_mb_per_Npi < 1:
            raise ValueError('number_aux_mb_per_Npi must be positive')
        if cfg.E_aux < 1:
            raise ValueError('E_aux must be positive for model training')
        super()._init()
        self._target_backbone = deepcopy(self._actor_critic.actor.shared_backbone)
        self._target_backbone.requires_grad_(False)
        self._aux_metrics: dict[str, list[float]] = {}

    def _make_phasic_buffer(self, policy_buffer: VectorOnPolicyBuffer) -> VectorModelPPGBuffer:
        cfg = self._cfgs.algo_cfgs
        return VectorModelPPGBuffer(
            policy_buffer=policy_buffer,
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            rollout_size=self._steps_per_epoch,
            N_pi=cfg.N_pi,
            standardized_adv_r=cfg.standardized_rew_adv,
            standardized_adv_c=cfg.standardized_cost_adv,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            model_horizon=cfg.model_horizon,
            device=self._device,
        )

    def _init_log(self) -> None:
        PPO._init_log(self)
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _after_auxiliary_phase(self, total_env_steps: int) -> None:
        self._logger.write_scalars(
            {key: sum(values) / len(values) for key, values in self._aux_metrics.items()},
            global_step=total_env_steps,
            csv_filename='auxiliary_metrics.csv',
        )
        self._aux_metrics.clear()

    def _record_aux_losses(
        self,
        raw: dict[str, torch.Tensor],
        weighted: dict[str, torch.Tensor],
        total: torch.Tensor,
    ) -> None:
        for name in self._AUX_COMPONENTS:
            self._aux_metrics.setdefault(f'Loss/Aux{name}', []).append(raw[name].item())
            self._aux_metrics.setdefault(f'Loss/AuxWeighted{name}', []).append(
                weighted[name].item(),
            )
        self._aux_metrics.setdefault('Loss/AuxTotal', []).append(total.item())

    @torch.no_grad()
    def _compute_old_policy_snapshot(self, batch_size: int = 4096) -> None:
        """Freeze latent, critic, and policy-distribution targets for this phase."""
        self._target_backbone.load_state_dict(
            self._actor_critic.actor.shared_backbone.state_dict(),
        )
        self._target_backbone.eval()
        data = self._buf.get_aux_data()
        for start in range(0, data['obs'].shape[0], batch_size):
            end = min(start + batch_size, data['obs'].shape[0])
            obs = data['obs'][start:end]
            data['critic_value_r'][start:end] = self._actor_critic.reward_critic(obs)[0].reshape(-1)
            data['critic_value_c'][start:end] = self._actor_critic.cost_critic(obs)[0].reshape(-1)
            distribution = self._actor_critic.actor._distribution(obs)
            data['mean_old'][start:end] = distribution.mean
            data['std_old'][start:end] = distribution.stddev

    def _model_losses(
        self,
        data: dict[str, torch.Tensor],
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll the model on real actions and compare with frozen future latents."""
        actor = self._actor_critic.actor
        latent = actor.shared_backbone(data['obs'][starts])
        latent_losses = []
        reward_losses = []
        cost_losses = []
        for offset in range(self._cfgs.algo_cfgs.model_horizon):
            indices = starts + offset
            next_latent, reward, cost = actor.predict_model(latent, data['act'][indices])
            with torch.no_grad():
                target_latent = self._target_backbone(data['obs'][indices + 1])
            latent_losses.append(F.mse_loss(next_latent, target_latent))
            reward_losses.append(F.smooth_l1_loss(reward, data['reward'][indices]))
            cost_losses.append(F.smooth_l1_loss(cost, data['cost'][indices]))
            latent = next_latent
        return (
            torch.stack(latent_losses).mean(),
            torch.stack(reward_losses).mean(),
            torch.stack(cost_losses).mean(),
        )

    def _aux_update(self) -> None:
        """Update only the actor encoder and its prediction heads."""
        data = self._buf.get_aux_data()
        cfg = self._cfgs.algo_cfgs
        size = data['obs'].shape[0]
        batch_size = max(1, size // (cfg.N_pi * cfg.number_aux_mb_per_Npi))
        batches = DataLoader(TensorDataset(torch.arange(size)), batch_size=batch_size, shuffle=True)

        for (indices,) in batches:
            obs = data['obs'][indices]
            value_r, value_c = self._actor_critic.actor.predict_values(obs)
            value_r_loss = F.mse_loss(value_r, data['critic_value_r'][indices])
            value_c_loss = F.mse_loss(value_c, data['critic_value_c'][indices])
            starts = indices[data['model_valid'][indices]]
            if starts.numel():
                latent_loss, reward_loss, cost_loss = self._model_losses(data, starts)
            else:
                zero = value_r_loss.new_zeros(())
                latent_loss = reward_loss = cost_loss = zero

            old_distribution = torch.distributions.Normal(
                data['mean_old'][indices],
                data['std_old'][indices],
            )
            new_distribution = self._actor_critic.actor._distribution(obs)
            kl = torch.distributions.kl.kl_divergence(old_distribution, new_distribution).mean()

            beta_loss = -torch.exp(self._log_beta_clone) * (kl.detach() - self._aux_kl_target)
            self._beta_optimizer.zero_grad()
            beta_loss.backward()
            self._beta_optimizer.step()
            with torch.no_grad():
                self._log_beta_clone.clamp_(min=-9.2, max=4.6)
                beta = torch.exp(self._log_beta_clone.detach())

            raw = {
                'Latent': latent_loss,
                'Reward': reward_loss,
                'Cost': cost_loss,
                'ValueReward': value_r_loss,
                'ValueCost': value_c_loss,
                'KL': kl,
            }
            weights = {
                'Latent': cfg.latent_loss_weight,
                'Reward': cfg.reward_prediction_weight,
                'Cost': cfg.cost_prediction_weight,
                'ValueReward': cfg.reward_value_prediction_weight,
                'ValueCost': cfg.cost_value_prediction_weight,
                'KL': beta,
            }
            weighted = {name: loss * weights[name] for name, loss in raw.items()}
            actor_loss = sum(weighted.values())
            self._record_aux_losses(raw, weighted, actor_loss)
            self._aux_metrics.setdefault('Metrics/BetaClone', []).append(beta.item())
            self._actor_critic.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._actor_critic.actor_optimizer.step()
