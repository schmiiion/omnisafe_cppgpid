# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the PID-Lagrange version of the TRPO algorithm."""

import torch
import torch.nn as nn
import time
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.buffer.vector_ppg_buffer import VectorPPGBuffer
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients

@registry.register
class PTRPOPID(TRPO):
    """The PID-Lagrange version of the TRPO algorithm.

    A simple combination of the PID-Lagrange method and the Trust Region Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the TRPOPID specific model.

        The TRPOPID algorithm uses a PID-Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._number_phases = self._cfgs.train_cfgs.total_steps // (
                self._cfgs.algo_cfgs.N_pi * self._cfgs.algo_cfgs.steps_per_epoch)
        self._buf: VectorPPGBuffer = VectorPPGBuffer(
            policy_buffer=self._buf,
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            rollout_size=self._steps_per_epoch,
            N_pi=self._cfgs.algo_cfgs.N_pi,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device
        )
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)
        self._aux_kl_target = self._cfgs.algo_cfgs.aux_kl_target
        init_beta_clone = self._cfgs.algo_cfgs.init_beta_clone
        self._log_beta_clone = nn.Parameter(
            torch.tensor(np.log(init_beta_clone), dtype=torch.float32, device=self._device))
        self._beta_optimizer = torch.optim.Adam(
            [self._log_beta_clone],
            lr=self._cfgs.algo_cfgs.beta_lr,
            betas=(0.0, 0.999)  # Disabled momentum for tight locking
        )

    def _init_log(self) -> None:
        """Log the TRPOPID specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Metrics/LagrangeMultiplier | The PID-Lagrange multiplier. |
        +----------------------------+------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._aux_actor_optimizer = torch.optim.Adam(
            self._actor_critic.actor.parameters(),
            lr=self._cfgs.algo_cfgs.aux_lr,
        )

    def learn(self) -> tuple[float, float, float]:
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for phase in range(self._number_phases):

            for policy_phase in range(self._cfgs.algo_cfgs.N_pi):

                current_epoch = phase * self._cfgs.algo_cfgs.N_pi + policy_phase

                rollout_time = time.time()
                self._env.rollout(
                    steps_per_epoch=self._steps_per_epoch,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                )
                self._logger.store({'Time/Rollout': time.time() - rollout_time})

                update_time = time.time()
                self._update()
                self._logger.store({'Time/Update': time.time() - update_time})

                if self._cfgs.model_cfgs.exploration_noise_anneal:
                    self._actor_critic.annealing(current_epoch)

                if self._cfgs.model_cfgs.actor.lr is not None:
                    self._actor_critic.actor_scheduler.step()

                self._logger.store(
                    {
                        'TotalEnvSteps': (current_epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                        'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - rollout_time),
                        'Time/Total': (time.time() - start_time),
                        'Time/Epoch': (time.time() - rollout_time),
                        'Train/Epoch': current_epoch,
                        'Train/LR': (
                            0.0
                            if self._cfgs.model_cfgs.actor.lr is None
                            else self._actor_critic.actor_scheduler.get_last_lr()[0]
                        ),
                    },
                )

                self._logger.dump_tabular()

                # save model to disk
                if (current_epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                    current_epoch + 1
                ) == self._cfgs.train_cfgs.epochs:
                    self._logger.torch_save()

            ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
            ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
            ep_len = self._logger.get_stats('Metrics/EpLen')[0]

            self._compute_old_policy_snapshot()  # populate the aux buffer with pi_old(.|s_t) for all states in it
            print("start aux phase")
            for aux_phase in range(self._cfgs.algo_cfgs.E_aux):
                self._aux_update()

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the PID-Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old} (a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the PID-Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        self._lagrange.pid_update(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        TRPOPID uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [A^{R}_{\pi_{\theta}}(s, a)
            - \lambda A^C_{\pi_{\theta}}(s, a)]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """
        penalty = self._lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)

    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        r"""Update policy network.

        Natural Policy Gradient (NPG) update policy network using the conjugate gradient algorithm,
        following the steps:

        - Calculate the gradient of the policy network,
        - Use the conjugate gradient algorithm to calculate the step direction.
        - Update the policy network by taking a step in the step direction.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.

        Raises:
            AssertionError: If :math:`x` is not finite.
            AssertionError: If :math:`x H x` is not positive.
            AssertionError: If :math:`\alpha` is not finite.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = self._get_flat_policy_params()
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -self._get_flat_policy_gradients()
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        theta_new = theta_old + step_direction
        self._set_flat_policy_params(theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
            },
        )
    def _policy_params(self):
        """Return only the parameters relevant to the policy distribution (exclude aux heads)."""
        return [p for name, p in self._actor_critic.actor.named_parameters()
                if 'aux' not in name]  # adjust 'aux' to match your actual aux head

    def _fvp(self, params: torch.Tensor) -> torch.Tensor:
        """Build the Hessian-vector product.

        Build the `Hessian-vector product <https://en.wikipedia.org/wiki/Hessian_matrix>`_ , which
        is the second-order derivative of the KL-divergence.

        The Hessian-vector product is approximated by the Fisher information matrix, which is the
        second-order derivative of the KL-divergence.

        For details see `John Schulman's PhD thesis (pp. 40) <http://joschu.net/docs/thesis.pdf>`_ .

        Args:
            params (torch.Tensor): The parameters of the actor network.

        Returns:
            The Fisher vector product.
        """
        self._actor_critic.actor.zero_grad()
        q_dist = self._actor_critic.actor(self._fvp_obs)
        with torch.no_grad():
            p_dist = self._actor_critic.actor(self._fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        policy_params = self._policy_params()
        grads = torch.autograd.grad(
            kl,
            policy_params,
            create_graph=True,
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(
            kl_p,
            policy_params,
            retain_graph=False,
        )

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        distributed.avg_tensor(flat_grad_grad_kl)

        self._logger.store(
            {
                'Train/KL': kl.item(),
            },
        )
        return flat_grad_grad_kl + params * self._cfgs.algo_cfgs.cg_damping

    def _get_flat_policy_params(self) -> torch.Tensor:
        flat_params = []
        for name, param in self._actor_critic.actor.named_parameters():
            if param.requires_grad and 'aux' not in name:
                flat_params.append(param.data.view(-1))
        assert flat_params, 'No policy parameters found.'
        # for name, p in self._actor_critic.actor.named_parameters():
        #     print(name, p.requires_grad)
        return torch.cat(flat_params)

    def _get_flat_policy_gradients(self) -> torch.Tensor:
        flat_grads = []
        for name, param in self._actor_critic.actor.named_parameters():
            if param.requires_grad and 'aux' not in name:
                assert param.grad is not None, f'No gradient for policy param {name}'
                flat_grads.append(param.grad.view(-1))
        assert flat_grads, 'No policy gradients found.'
        return torch.cat(flat_grads)

    def _set_flat_policy_params(self, flat_params: torch.Tensor) -> None:
        assert isinstance(flat_params, torch.Tensor)
        offset = 0
        for name, param in self._actor_critic.actor.named_parameters():
            if param.requires_grad and 'aux' not in name:
                numel = param.numel()
                param.data.copy_(flat_params[offset:offset + numel].view_as(param))
                offset += numel
        assert offset == len(flat_params), f'Lengths do not match: {offset} vs. {len(flat_params)}'

    @torch.no_grad()
    def _compute_old_policy_snapshot(self, batch_size: int = 4096):
        """Captures the current policy distribution for all states in the auxiliary buffer."""
        total_obs = self._buf.get_aux_data()['obs']
        number_obs = total_obs.shape[0]
        for start in range(0, number_obs, batch_size):
            end = min(start + batch_size, number_obs)
            obs = total_obs[start:end]
            dist = self._actor_critic.actor._distribution(obs)
            self._buf._aux_buffer.data['mean_old'][start:end] = dist.mean
            self._buf._aux_buffer.data['std_old'][start:end] = dist.stddev

    def _aux_update(self) -> None:
        data = self._buf.get_aux_data()
        dataset_size = data["obs"].shape[0]
        batch_size = dataset_size // (self._cfgs.algo_cfgs.N_pi * self._cfgs.algo_cfgs.number_aux_mb_per_Npi)

        obs, target_value_r, target_value_c, mean_old, std_old = (
            data['obs'],
            data['target_value_r'],
            data['target_value_c'],
            data['mean_old'],
            data['std_old'],
        )

        dataloader = DataLoader(
            dataset=TensorDataset(obs, target_value_r, target_value_c, mean_old, std_old),
            batch_size=batch_size,
            shuffle=True,
        )

        for (obs, target_value_r, target_value_c, mean_old, std_old) in dataloader:
            #1. Distill features into actor network
            reward_pred, cost_pred = self._actor_critic.actor.auxiliary_forward(obs)

            L_reward = nn.functional.mse_loss(reward_pred, target_value_r)
            L_cost = nn.functional.mse_loss(cost_pred, target_value_c)

            old_distribution = torch.distributions.Normal(mean_old, std_old)
            new_distribution = self._actor_critic.actor._distribution(obs)
            kl = torch.distributions.kl.kl_divergence(old_distribution, new_distribution).mean()
            #kl = distributed.dist_avg(kl)

            beta_loss = -torch.exp(self._log_beta_clone) * (kl.detach() - self._aux_kl_target)
            self._beta_optimizer.zero_grad()
            beta_loss.backward()
            self._beta_optimizer.step()
            with torch.no_grad():
                self._log_beta_clone.clamp_(min=-9.2, max=4.6) # evaluates to 1e-4 and 100
                current_beta_val = torch.exp(self._log_beta_clone).item()
                self._logger.store({'Metrics/BetaClone': current_beta_val})

            L_joint = L_reward + self._cfgs.algo_cfgs.alpha_cost * L_cost + torch.exp(self._log_beta_clone.detach()) * kl
            self._aux_actor_optimizer.zero_grad()
            L_joint.backward()
            self._aux_actor_optimizer.step()

            #2. Update critics
            # Reward Critic
            self._actor_critic.reward_critic_optimizer.zero_grad()
            r_critic_pred = self._actor_critic.reward_critic(obs)[0]
            r_critic_loss = nn.functional.mse_loss(r_critic_pred, target_value_r)
            r_critic_loss.backward()
            self._actor_critic.reward_critic_optimizer.step()

            # Cost Critic
            self._actor_critic.cost_critic_optimizer.zero_grad()
            c_critic_pred = self._actor_critic.cost_critic(obs)[0]
            c_critic_loss = nn.functional.mse_loss(c_critic_pred, target_value_c)
            c_critic_loss.backward()
            self._actor_critic.cost_critic_optimizer.step()


