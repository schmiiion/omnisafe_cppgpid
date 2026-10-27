"""Small bridge to native OmniSafe solvers and fresh learner trajectories."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.envs.core import Wrapper


class LearnedCost(Wrapper):
    """Replace costs at the raw CMDP boundary, before OmniSafe normalizes them."""

    def __init__(self, env, constraint, device):
        super().__init__(env, device)
        self.constraint = constraint
        self.observation = None

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self.observation = obs.detach().clone()
        return obs, info

    def step(self, action):
        # Cost belongs to (s_t, a_t), not the next or auto-reset observation.
        costs = self.constraint.immediate_cost(self.observation, action)
        obs, reward, true_cost, terminated, truncated, info = self._env.step(action)
        self.observation = obs.detach().clone()
        info['true_cost'] = true_cost
        cost = costs.to(device=true_cost.device, dtype=true_cost.dtype)
        return obs, reward, cost.reshape(true_cost.shape), terminated, truncated, info


def make_learner(cfgs, constraint):
    """Construct once; all subsequent calls train the same native algorithm."""
    algorithm = registry.get(cfgs.algo)
    if algorithm.learn is not PolicyGradient.learn:
        raise ValueError(
            f'{cfgs.algo} has a custom training loop; this prototype supports algorithms '
            'using PolicyGradient.learn (e.g. PPOLag, TRPOLag, CPO, CPPOPID).',
        )
    learner = algorithm(env_id=cfgs.env_id, cfgs=cfgs)
    # Keep native wrappers, including action scaling and cost normalization.
    parent = learner._env
    while isinstance(parent._env, Wrapper):
        parent = parent._env
    parent._env = LearnedCost(parent._env, constraint, learner._device)
    return learner


def reset_rollout_buffers(learner):
    """Discard trajectory data, retaining all learned and running statistics.

    The next native rollout resets the environment and episode accumulators.
    Do not reset it here too: that would count an unused initial observation in
    the running normalizer. Neither the multiplier nor its optimizer is reset.
    """
    for buffer in learner._buf.buffers:
        buffer.ptr = buffer.path_start_idx = 0
        for values in buffer.data.values():
            values.zero_()
    # These costs were measured under the previous constraint. The multiplier
    # must use fresh episode costs, while retaining its value and Adam state.
    learner.logger._data['Metrics/EpCost'].clear()


@torch.no_grad()
def collect_trajectories(learner, env, constraint, episodes, recorder=None):
    """Sample the updated policy; return raw arrays and episode lengths.

    OmniSafe's evaluator returns metrics, not trajectories. Use its actor and CMDP
    interfaces directly here. Freeze policy observation statistics while sampling.
    True costs are used only for reporting and never as constraint-training labels.
    """
    actor = learner._actor_critic.actor
    normalizer = learner._env.save().get('obs_normalizer')
    device = learner._device
    # Safety Gymnasium may expose float64 bounds while actor and constraint use
    # float32; preserve the policy dtype when scaling actions for both of them.
    action_dtype = next(actor.parameters()).dtype
    low = torch.as_tensor(env.action_space.low, device=device, dtype=action_dtype)
    high = torch.as_tensor(env.action_space.high, device=device, dtype=action_dtype)
    observations, actions, lengths = [], [], []
    returns, true_costs, learned_costs = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        if recorder is not None:
            recorder.start_episode(env)
        episode_return = true_cost_sum = learned_cost_sum = 0.0
        for step in range(env.max_episode_steps):
            policy_obs = obs.to(device)
            # Normalizer.normalize() updates statistics even in eval() mode.
            if normalizer is not None and normalizer._count > 1:
                policy_obs = ((policy_obs - normalizer.mean) / normalizer.std).clamp(
                    -normalizer._clip, normalizer._clip,
                )
            action = actor.predict(policy_obs, deterministic=False)
            # Same transformation as OmniSafe's ActionScale([-1, 1]).
            native_action = (low + (high - low) * (action + 1) / 2).reshape(env.action_space.shape)
            raw_obs = obs.detach().cpu().numpy().reshape(-1).copy()
            raw_action = native_action.cpu().numpy().reshape(-1).copy()
            observations.append(raw_obs)
            actions.append(raw_action)
            if recorder is not None:
                recorder.record_step(env)
            learned_cost_sum += constraint.immediate_cost(obs.to(device), native_action).item()
            obs, reward, cost, terminated, truncated, _ = env.step(native_action)
            episode_return += reward.item()
            true_cost_sum += cost.item()
            if terminated.item() or truncated.item():
                break
        lengths.append(step + 1)
        returns.append(episode_return)
        true_costs.append(true_cost_sum)
        learned_costs.append(learned_cost_sum)
    metrics = {
        'ICRL/SampleReturn': float(np.mean(returns)),
        'ICRL/SampleTrueCost': float(np.mean(true_costs)),
        'ICRL/SampleLearnedCostBefore': float(np.mean(learned_costs)),
        'ICRL/SampleViolationRate': float(np.mean(np.asarray(true_costs) > 0)),
        'ICRL/SampleEpisodeLength': float(np.mean(lengths)),
    }
    return np.asarray(observations), np.asarray(actions), np.asarray(lengths), metrics


def mean_learned_episode_cost(constraint, observations, actions, episode_lengths):
    """Rescore stored transitions; average undiscounted episode sums, not step costs."""
    obs = torch.as_tensor(observations, dtype=torch.float32, device=constraint.device)
    acs = torch.as_tensor(actions, dtype=torch.float32, device=constraint.device)
    costs = constraint.immediate_cost(obs, acs)
    return float(costs.double().sum().item() / len(episode_lengths))


def log_iteration(learner, metrics, output):
    """Keep outer metrics separate from the native inner-loop CSV schema."""
    path = Path(output) / 'icrl_progress.csv'
    new_file = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        if new_file:
            writer.writeheader()
        writer.writerow(metrics)
    iteration = int(metrics['ICRL/Iteration'])
    learner.logger.write_scalars(
        {key: value for key, value in metrics.items() if isinstance(value, (int, float, np.number))},
        global_step=int(metrics['ICRL/PolicyEpoch']),
        tensorboard_step=iteration,
    )
    learner.logger.log(
        f'ICRL iteration {iteration} complete: '
        f'return={metrics["ICRL/SampleReturn"]:.3f}, '
        f'true cost={metrics["ICRL/SampleTrueCost"]:.3f}, '
        f'learned cost before={metrics["ICRL/SampleLearnedCostBefore"]:.3f}, '
        f'after={metrics["ICRL/SampleLearnedCostAfter"]:.3f}, '
        f'loss (last batch)={metrics["ICRL/backward/cn_loss"]:.6f}',
    )


def close_learner(learner):
    """Staged learn() leaves these resources open for the next outer iteration."""
    learner._env.close()
    if learner._env._eval_env is not None:
        learner._env._eval_env.close()
    learner.logger.close()
    if learner._cfgs.logger_cfgs.use_tensorboard:
        learner.logger._tensorboard_writer.close()
