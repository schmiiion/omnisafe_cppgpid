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
"""Implementation of the CPO algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.buffer.vector_ppg_buffer import VectorPPGBuffer
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients


@registry.register
class PhasicCPO(TRPO):
    """The Phasic Constrained Policy Optimization (CPO) algorithm."""

    def _init(self) -> None:
        super()._init()
        #TODO: some assertion like this should be enabled, otherwise the training ends weirdly
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
        self._aux_actor_optimizer = torch.optim.Adam(
            self._actor_critic.actor.parameters(),
            lr=self._cfgs.algo_cfgs.aux_lr,
        )
        self._aux_kl_target = self._cfgs.algo_cfgs.aux_kl_target
        init_beta_clone = self._cfgs.algo_cfgs.init_beta_clone
        self._log_beta_clone = nn.Parameter(torch.tensor(np.log(init_beta_clone), dtype=torch.float32, device=self._device))
        self._beta_optimizer = torch.optim.Adam(
            [self._log_beta_clone],
            lr=self._cfgs.algo_cfgs.beta_lr,
            betas=(0.0, 0.999)  # Disabled momentum for tight locking
        )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/cost_gradient_norm')
        self._logger.register_key('Misc/A')
        self._logger.register_key('Misc/B')
        self._logger.register_key('Misc/q')
        self._logger.register_key('Misc/r')
        self._logger.register_key('Misc/s')
        self._logger.register_key('Misc/Lambda_star')
        self._logger.register_key('Misc/Nu_star')
        self._logger.register_key('Misc/OptimCase')
        self._logger.register_key('Metrics/BetaClone')

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

            print("start policy phase")
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

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        total_steps: int = 15,
        decay: float = 0.8,
        violation_c: int = 0,
        optim_case: int = 0,
    ) -> tuple[torch.Tensor, int]:
        r"""Use line-search to find the step size that satisfies the constraint.

        CPO uses line-search to find the step size that satisfies the constraint. The constraint is
        defined as:

        .. math::

            J^C (\theta + \alpha \delta) - J^C (\theta) \leq \max \{ 0, c \} \\
            D_{KL} (\pi_{\theta} (\cdot|s) || \pi_{\theta + \alpha \delta} (\cdot|s)) \leq \delta_{KL}

        where :math:`\delta_{KL}` is the constraint of KL divergence, :math:`\alpha` is the step size,
        :math:`c` is the violation of constraint.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
            violation_c (int, optional): The violation of constraint. Defaults to 0.
            optim_case (int, optional): The optimization case. Defaults to 0.

        Returns:
            A tuple of final step direction and the size of acceptance steps.
        """
        # get distance each time theta goes towards certain direction
        step_frac = 1.0
        # get and flatten parameters from pi-net
        theta_old = self._get_flat_policy_params()

        # reward improvement, g-flat as gradient of reward
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        # while not within_trust_region and not finish all steps:
        for step in range(total_steps):
            # get new theta
            new_theta = theta_old + step_frac * step_direction
            # set new theta as new actor parameters
            self._set_flat_policy_params(new_theta)
            # the last acceptance steps to next step
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    # loss of policy reward from target/expected reward
                    loss_reward = self._loss_pi(obs=obs, act=act, logp=logp, adv=adv_r)
                except ValueError:
                    step_frac *= decay
                    continue
                # loss of cost of policy cost from real/expected reward
                loss_cost = self._loss_pi_cost(obs=obs, act=act, logp=logp, adv_c=adv_c)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
            # compute improvement of reward
            loss_reward_improve = loss_reward_before - loss_reward
            # compute difference of cost
            loss_cost_diff = loss_cost - loss_cost_before

            # average across MPI processes...
            kl = distributed.dist_avg(kl)
            # pi_average of torch_kl above
            loss_reward_improve = distributed.dist_avg(loss_reward_improve)
            loss_cost_diff = distributed.dist_avg(loss_cost_diff)
            self._logger.log(
                f'Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}',
            )
            # check whether there are nan.
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                self._logger.log('WARNING: loss_pi not finite')
            if not torch.isfinite(kl):
                self._logger.log('WARNING: KL not finite')
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                self._logger.log('INFO: did not improve improve <0')
            # change of cost's range
            elif loss_cost_diff > max(-violation_c, 0):
                self._logger.log(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
            # check KL-distance to avoid too far gap
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'INFO: violated KL constraint {kl} at step {step + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self._logger.log(f'Accept step at i={step + 1}')
                break
            step_frac *= decay
        else:
            # if didn't find a step satisfy those conditions
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        self._logger.store(
            {
                'Train/KL': kl,
            },
        )

        self._set_flat_policy_params(theta_old)
        return step_frac * step_direction, acceptance_step

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the performance of cost on this moment.

        We compute the loss of cost of policy cost from real cost.

        .. math::

            L = \mathbb{E}_{\pi} \left[ \frac{\pi^{'} (a|s)}{\pi (a|s)} A^C (s, a) \right]

        where :math:`A^C (s, a)` is the cost advantage, :math:`\pi (a|s)` is the old policy,
        and :math:`\pi^{'} (a|s)` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The loss of the cost performance.
        """
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        return (ratio * adv_c).mean()

    # pylint: disable=invalid-name
    def _determine_case(
        self,
        b_grads: torch.Tensor,
        ep_costs: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Determine the case of the trust region update.

        Args:
            b_grad (torch.Tensor): Gradient of the cost function.
            ep_costs (torch.Tensor): Cost of the current episode.
            q (torch.Tensor): The quadratic term of the quadratic approximation of the cost function.
            r (torch.Tensor): The linear term of the quadratic approximation of the cost function.
            s (torch.Tensor): The constant term of the quadratic approximation of the cost function.

        Returns:
            optim_case: The case of the trust region update.
            A: The quadratic term of the quadratic approximation of the cost function.
            B: The linear term of the quadratic approximation of the cost function.
        """
        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), 'r is not finite'
            assert torch.isfinite(s).all(), 's is not finite'

            A = q - r**2 / (s + 1e-8)
            B = 2 * self._cfgs.algo_cfgs.target_kl - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif ep_costs < 0 <= B:
                # point in trust region is feasible but safety boundary intersects
                # ==> only part of trust region is feasible
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                # point in trust region is infeasible and cost boundary doesn't intersect
                # ==> entire trust region is infeasible
                optim_case = 1
                self._logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety half space is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self._logger.log('Alert! Attempting infeasible recovery!', 'red')

        return optim_case, A, B

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        ep_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if optim_case in (3, 4):
            # under 3 and 4 cases directly use TRPO method
            alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(data: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
                """Project data to [low, high] interval."""
                return torch.clamp(data, low, high)

            #  analytical Solution to LQCLP, employ lambda,nu to compute final solution of OLOLQC
            #  λ=argmax(f_a(λ),f_b(λ)) = λa_star or λb_star
            #  computing formula shown in appendix, lambda_a and lambda_b
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self._cfgs.algo_cfgs.target_kl))
            # λa_star = Proj(lambda_a ,0 ~ r/c)  λb_star=Proj(lambda_b,r/c~ +inf)
            # where projection(str,b,c)=max(b,min(str,c))
            # may be regarded as a projection from effective region towards safety region
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(lambda_a, torch.as_tensor(0.0), r_num / eps_cost)
                lambda_b_star = project(lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf))
            else:
                lambda_a_star = project(lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf))
                lambda_b_star = project(lambda_b, torch.as_tensor(0.0), r_num / eps_cost)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self._cfgs.algo_cfgs.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )

            # discard all negative values with torch.clamp(x, min=0)
            # Nu_star = (lambda_star * - r)/s
            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)
            # final x_star as final direction played as policy's loss to backward and update
            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:  # case == 0
            # purely decrease costs
            # without further check
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        return step_direction, lambda_star, nu_star

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

        Constrained Policy Optimization updates policy network using the
        `conjugate gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint.
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = self._get_flat_policy_params()
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -self._get_flat_policy_gradients()
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        self._actor_critic.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grads = self._get_flat_policy_gradients()
        ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction
        self._set_flat_policy_params(theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            },
        )

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

    def _policy_params(self):
        """Return only the parameters relevant to the policy distribution (exclude aux heads)."""
        return [p for name, p in self._actor_critic.actor.named_parameters()
                if 'aux' not in name]

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

