WEIGHTS_PATH = "/Users/jonas/Uni/SoSe26/Masterarbeit/omnisafe_cppgpid/ICRL/expert/expert_weights/CPPOPID-{SafetyPointCircle1-v0}/seed-001-2026-10-22-07-46-05"
NUM_TRAJECTORIES = 50

# Run from the repository root: python -m ICRL.expert.weights2trajectories
# WEIGHTS_PATH can be an epoch-*.pt checkpoint or a run directory. A run
# directory selects its highest-numbered epoch checkpoint automatically.
# Metadata comes from config.json alongside the checkpoint's torch_save folder.
# NUM_TRAJECTORIES counts accepted, zero-cost episodes. Actions are deterministic;
# each attempt uses a fresh seed: seed, seed+1, ... . SafeLOOP exports use its
# saved actor without the planner, whose dynamics scaler is not in the checkpoint.

import json
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box

from ICRL import point_circle_xy_env  # noqa: F401  # Register expert's XY environment.
from omnisafe.common import Normalizer
from omnisafe.envs.core import make
from omnisafe.models.actor import ActorBuilder
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device, seed_all


@torch.no_grad()
def sample_trajectories(weights_path=WEIGHTS_PATH, num_trajectories=NUM_TRAJECTORIES,
                        output_path=None, max_attempts=None):
    """Save complete, strictly zero-true-cost expert episodes for ICRL.

    Try at most 10 times the requested count by default, or 25 times for a
    SafeLOOP actor-only export. If too few episodes qualify, raise without
    writing a partial dataset or overwriting an old export.
    """
    weights_path = Path(weights_path).expanduser().resolve()
    if weights_path.is_dir():
        weights_path = max((weights_path / 'torch_save').glob('epoch-*.pt'),
                           key=lambda path: int(path.stem.split('-')[-1]))
        print(f'Using latest checkpoint: {weights_path}', flush=True)
    run_dir = weights_path.parent.parent
    config_path = run_dir / 'config.json'
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    cfgs = Config.dict2config(metadata)
    device = get_device(cfgs.train_cfgs.device)
    torch.set_num_threads(cfgs.train_cfgs.torch_threads)
    seed_all(cfgs.seed)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    # Model-based checkpoints save actor_critic rather than pi. They do not save
    # the dynamics scaler needed to restore the planner, so export their actor.
    actor_only = 'actor_critic' in checkpoint
    if max_attempts is None:
        max_attempts = (25 if actor_only else 10) * num_trajectories
    dataset_env_id = cfgs.env_id.removesuffix('-modelbased') if actor_only else cfgs.env_id
    action_repeat = int(cfgs.algo_cfgs.action_repeat) if actor_only else 1

    # One environment collects complete episodes; training's vector count is irrelevant here.
    env = make(cfgs.env_id, num_envs=1, device=device, **(cfgs.get('env_cfgs') or {}))
    observation_mode = (
        'point_xy' if (cfgs.get('env_cfgs') or {}).get('class_name') == 'ICRLPointCircleXYEnv'
        else 'standard'
    )
    observation_dim = env.observation_space.shape[-1]
    episode_limit = env.max_episode_steps or env._env.spec.max_episode_steps
    observations, actions, rewards, costs = [], [], [], []
    lengths, episode_returns, episode_costs, seeds = [], [], [], []
    try:
        # Training actors operate in [-1, 1], before OmniSafe's ActionScale wrapper.
        policy_action_space = Box(-1.0, 1.0, shape=env.action_space.shape,
                                  dtype=env.action_space.dtype)
        actor = ActorBuilder(
            obs_space=env.observation_space,
            act_space=policy_action_space,
            hidden_sizes=cfgs.model_cfgs.actor.hidden_sizes,
            activation=cfgs.model_cfgs.actor.activation,
            weight_initialization_mode=cfgs.model_cfgs.weight_initialization_mode,
        ).build_actor(cfgs.model_cfgs.actor_type).to(device)
        if actor_only:
            actor.load_state_dict({key.removeprefix('actor.'): value
                                   for key, value in checkpoint['actor_critic'].items()
                                   if key.startswith('actor.')})
        else:
            actor.load_state_dict(checkpoint['pi'])
        actor.eval()

        normalizer = None
        if cfgs.algo_cfgs.obs_normalize:
            normalizer = Normalizer(env.observation_space.shape).to(device)
            normalizer.load_state_dict(checkpoint['obs_normalizer'])
        low = torch.as_tensor(env.action_space.low, device=device)
        high = torch.as_tensor(env.action_space.high, device=device)

        for attempt in range(max_attempts):
            episode_seed = cfgs.seed + attempt
            obs, info = env.reset(seed=episode_seed)
            raw_obs = info['obs_original'] if actor_only else obs
            episode_return = episode_cost = 0.0
            episode_obs, episode_actions, episode_rewards, episode_cost_values = [], [], [], []
            steps = 0
            done = False
            while steps < episode_limit and not done:
                policy_obs = obs
                # Calling Normalizer.normalize() would update the saved statistics,
                # even in eval mode. Apply them directly to keep the expert fixed.
                if normalizer is not None and normalizer._count > 1:
                    policy_obs = ((obs - normalizer.mean) / normalizer.std).clamp(
                        -normalizer._clip, normalizer._clip,
                    )
                policy_action = actor.predict(policy_obs, deterministic=True).clamp(-1, 1)
                native_action = low + (high - low) * (policy_action + 1) / 2

                # A model-based actor acts once per action-repeat interval. Store
                # every underlying raw (s, a) so the NPZ replays in the ordinary
                # SafetyGoal environment and ICRL receives its expected observations.
                for _ in range(min(action_repeat, episode_limit - steps)):
                    episode_obs.append(np.asarray(raw_obs, dtype=np.float32).reshape(-1).copy())
                    episode_actions.append(native_action.cpu().numpy().reshape(-1).copy())
                    obs, reward, cost, terminated, truncated, info = env.step(native_action)
                    raw_obs = info['obs_original'] if actor_only else obs
                    episode_rewards.append(reward.item())
                    episode_cost_values.append(cost.item())
                    episode_return += reward.item()
                    episode_cost += cost.item()
                    steps += 1
                    done = bool(terminated.item() or truncated.item())
                    if done:
                        break
            # Reject incomplete rollouts and the whole episode after any true
            # cost. Per-step checks also reject NaNs and cost cancellation.
            accepted = done and all(value == 0.0 for value in episode_cost_values)
            if accepted:
                observations.extend(episode_obs)
                actions.extend(episode_actions)
                rewards.extend(episode_rewards)
                costs.extend(episode_cost_values)
                lengths.append(steps)
                episode_returns.append(episode_return)
                episode_costs.append(episode_cost)
                seeds.append(episode_seed)
            print(f'Attempt {attempt + 1}: {"accepted" if accepted else "rejected"}, '
                  f'length={steps}, return={episode_return:.2f}, true_cost={episode_cost:g}; '
                  f'accepted={len(lengths)}/{num_trajectories}', flush=True)
            if len(lengths) == num_trajectories:
                break
    finally:
        env.close()

    if len(lengths) < num_trajectories:
        raise RuntimeError(
            f'Only {len(lengths)}/{num_trajectories} zero-cost trajectories collected '
            f'in {max_attempts} attempts. No dataset written; use a safer checkpoint '
            'or increase max_attempts.',
        )

    if output_path is None:
        # Keep exports together, identifying the task, training run and exact snapshot.
        snapshot = f'{cfgs.algo}-{cfgs.env_id}__{run_dir.name}__{weights_path.stem}'
        if observation_mode == 'point_xy':
            snapshot += '__point-xy'
        if actor_only:
            snapshot += '__actor-only'
        trajectory_dir = Path(__file__).resolve().parent / 'expert_trajectories' / snapshot
        output_path = trajectory_dir / f'{weights_path.stem}-{num_trajectories}-episodes.npz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_lengths=np.asarray(lengths, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray(costs, dtype=np.float32),
        episode_returns=np.asarray(episode_returns, dtype=np.float64),
        episode_costs=np.asarray(episode_costs, dtype=np.float64),
        episode_seeds=np.asarray(seeds, dtype=np.int64),
    )
    # Keep provenance next to the data. The ICRL loader reads only observations/actions.
    provenance = {
        'checkpoint': str(weights_path), 'config_path': str(config_path),
        'dataset_env_id': dataset_env_id,
        'source_config': metadata, 'num_trajectories': num_trajectories,
        'cost_filter': 'all per-step ground-truth costs == 0',
        'attempted_trajectories': attempt + 1,
        'rejected_trajectories': attempt + 1 - len(lengths),
        'max_attempts': max_attempts,
        'deterministic': True, 'observation_normalizer_frozen': True,
        'policy_mode': 'actor_only_no_planner' if actor_only else 'saved_actor',
        'source_action_repeat': action_repeat,
        'observations': 'raw pre-action states', 'actions': 'bounded native actions',
        'observation_mode': observation_mode,
        'observation_dim': observation_dim,
        'xy_observation_indices': [observation_dim - 2, observation_dim - 1]
        if observation_mode == 'point_xy' else None,
        'xy_coordinate_frame': 'world' if observation_mode == 'point_xy' else None,
        'mean_return': float(np.mean(episode_returns)),
        'mean_true_cost': float(np.mean(episode_costs)),
    }
    output_path.with_suffix('.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Saved {num_trajectories} trajectories ({sum(lengths)} transitions) to {output_path}')
    print(f'Set icrl.expert_dataset to: {output_path}')
    print(f'Set icrl.env_id: {dataset_env_id} in the ICRL configuration.')
    return output_path


if __name__ == '__main__':
    sample_trajectories()
