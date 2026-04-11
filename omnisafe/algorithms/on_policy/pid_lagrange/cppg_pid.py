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
"""Implementation of the CPPOPID (PID version of PPOLag) algorithm."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.common.buffer.vector_ppg_buffer import VectorPPGBuffer


@registry.register
class CPPGPID(PPO):
    """The CPPOPID (PID version of PPOLag) algorithm.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `CPPOPID <https://arxiv.org/abs/2007.03964>`_
    """

    def _init(self) -> None:
        """Initialize the CPPOPID specific model.

        The CPPOPID algorithm uses a PID-Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        # assert self._cfgs.train_cfgs.epochs % self._cfgs.algo_cfgs.N_pi == 0, \
        #     "Total epochs must be a multiple of N_pi!"
        self._number_phases = self._cfgs.train_cfgs.total_steps // (self._cfgs.algo_cfgs.N_pi * self._cfgs.algo_cfgs.steps_per_epoch)
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

    def _init_log(self) -> None:
        """Log the CPPOPID specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Metrics/LagrangeMultiplier | The PID-Lagrange multiplier. |
        +----------------------------+------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')


    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for phase in range(self._number_phases):
            phase_time = time.time()

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
            print("finished policy phase")

            ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
            ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
            ep_len = self._logger.get_stats('Metrics/EpLen')[0]

            print("start aux phase")
            self._compute_old_policy_snapshot() #populate the aux buffer with pi_old(.|s_t) for all states in it
            for aux_phase in range(self._cfgs.algo_cfgs.E_aux):
                self._aux_update()
            print("finished aux phase")

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the PID-Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm.
            When a lagrange multiplier is used,
            the :meth:`_loss_pi` method will return the loss of the policy as:

            .. math::

                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old} (a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the PID-Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes.
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        self._lagrange.pid_update(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})


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

            L_joint = L_reward + self._cfgs.algo_cfgs.alpha_cost * L_cost + self._cfgs.algo_cfgs.beta_clone * kl
            self._actor_critic.actor_optimizer.zero_grad()
            L_joint.backward()
            self._actor_critic.actor_optimizer.step()

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


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        CPPOPID uses the following surrogate loss:

        .. math::
            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """
        penalty = self._lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
