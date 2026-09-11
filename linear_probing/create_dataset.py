"""Collect a fixed Safety-Gymnasium dataset for linear representation probing."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
import safety_gymnasium
import torch

from linear_probing.artifacts import new_version_id, versioned_directory, versioned_path
from linear_probing.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, task_switches
from linear_probing.dataset_io import (
    DATASET_SCHEMA_VERSION,
    TARGET_SCHEMA_VERSION,
    file_sha256,
    reproducibility_metadata,
    save_dataset_artifact,
)
from linear_probing.snapshots import (
    build_actor,
    ensure_disjoint_snapshot_runs,
    load_snapshot,
    read_only_normalize,
    select_snapshots,
    snapshot_run_directory,
)
from linear_probing.targets import (
    SLIP_SPEED_EPSILON,
    TASK_GROUPS,
    TASK_TYPES,
    TARGET_DEFINITIONS,
    GridPathPlanner,
    Scene,
    local_targets,
    signed_angle,
    validate_planner_config,
)
from linear_probing.visualization import clear_explanation_figures, save_explanation_figure


SOURCES = ('random', 'evasive', 'converged')


def _task_from_env(env: Any) -> Any:
    base_env = env.unwrapped
    if not hasattr(base_env, 'task'):
        raise TypeError('Expected a Safety-Gymnasium environment exposing `env.unwrapped.task`.')
    return base_env.task


def _flat_observation(env: Any, observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        observation = gymnasium.spaces.flatten(env.observation_space, observation)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _scene_from_env(env: Any, agent_radius: float) -> Scene:
    task = _task_from_env(env)
    if not hasattr(task, 'hazards') or not hasattr(task, 'goal'):
        raise TypeError('Probe targets require a task with `hazards` and `goal` objects.')

    rotation = np.asarray(task.agent.mat, dtype=np.float64)
    heading = rotation[:2, 0]
    velocimeter = np.asarray(task.agent.get_sensor('velocimeter'), dtype=np.float64)
    if velocimeter.size < 2:
        raise ValueError('The agent velocimeter must provide at least two components.')
    return Scene(
        agent_xy=np.asarray(task.agent.pos[:2], dtype=np.float64),
        heading_xy=heading / max(float(np.linalg.norm(heading)), 1e-12),
        agent_velocity_xy=np.asarray(task.agent.vel[:2], dtype=np.float64),
        agent_velocity=velocimeter[:2],
        goal_xy=np.asarray(task.goal.pos[:2], dtype=np.float64),
        hazards_xy=np.asarray(
            [position[:2] for position in task.hazards.pos],
            dtype=np.float64,
        ).reshape(-1, 2),
        agent_radius=agent_radius,
        hazard_radius=float(task.hazards.size),
        goal_radius=float(task.goal.size),
    )


class AgentDynamics:
    """Read agent-only dynamic quantities from the current MuJoCo state."""

    def __init__(self, task: Any) -> None:
        self.agent = task.agent
        self.model: Any | None = None
        self.dof_indices = np.empty(0, dtype=np.int64)
        self.mass_matrix = np.empty((0, 0), dtype=np.float64)
        self._refresh_model()

    def _refresh_model(self) -> None:
        """Refresh model-dependent buffers after Safety-Gymnasium rebuilds its world."""
        self.model = self.agent.engine.model
        agent_body = self.model.body('agent').id
        dof_indices = []
        for dof_index, dof_body in enumerate(self.model.dof_bodyid):
            body = int(dof_body)
            while body > 0 and body != agent_body:
                body = int(self.model.body_parentid[body])
            if body == agent_body:
                dof_indices.append(dof_index)
        if not dof_indices:
            raise ValueError('Could not identify MuJoCo degrees of freedom for the agent body.')
        self.dof_indices = np.asarray(dof_indices, dtype=np.int64)
        self.mass_matrix = np.empty((self.model.nv, self.model.nv), dtype=np.float64)

    def kinetic_energy(self) -> float:
        """Return ``0.5 * qvel.T @ M(q) @ qvel`` for the agent body subtree."""
        if self.agent.engine.model is not self.model:
            self._refresh_model()
        data = self.agent.engine.data
        mujoco.mj_fullM(self.model, self.mass_matrix, data.qM)
        velocity = np.asarray(data.qvel, dtype=np.float64)[self.dof_indices]
        inertia = self.mass_matrix[np.ix_(self.dof_indices, self.dof_indices)]
        energy = 0.5 * float(velocity @ inertia @ velocity)
        if not np.isfinite(energy):
            raise FloatingPointError('MuJoCo returned non-finite agent kinetic energy.')
        if energy < -1e-10:
            raise FloatingPointError(f'MuJoCo returned negative agent kinetic energy: {energy}.')
        return max(0.0, energy)


def _planner_key(scene: Scene) -> tuple[bytes, bytes, float]:
    return (
        np.round(scene.hazards_xy, 6).tobytes(),
        np.round(scene.goal_xy, 6).tobytes(),
        round(scene.expanded_hazard_radius, 6),
    )


class RestoredPolicy:
    """Inference-only actor reconstructed from one converged snapshot."""

    def __init__(self, path: Path, device: str) -> None:
        self.path = path
        self.device = device
        self.snapshot = load_snapshot(path, device)
        self.actor = build_actor(self.snapshot, device)

    def validate_environment(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        environment_id: str,
    ) -> None:
        """Validate tensor interfaces and disclose intentional cross-environment transfer."""
        spec = self.snapshot['actor_spec']
        expected_observation = tuple(spec['observation_shape'])
        expected_action = tuple(spec['action_shape'])
        if expected_observation != observation_shape or expected_action != action_shape:
            raise ValueError(
                f'Collector {self.path} expects observation/action shapes '
                f'{expected_observation}/{expected_action}, but the dataset environment provides '
                f'{observation_shape}/{action_shape}.',
            )
        snapshot_environment = self.snapshot.get('config', {}).get('env_id')
        if snapshot_environment and snapshot_environment != environment_id:
            warnings.warn(
                f'Collector {self.path} was trained on {snapshot_environment}, but is being used '
                f'on {environment_id}. This is allowed as an explicit transfer experiment.',
                stacklevel=2,
            )
        dummy = torch.zeros(expected_observation, dtype=torch.float32, device=self.device)
        read_only_normalize(dummy, self.snapshot)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return the deterministic normalized action for one raw observation."""
        expected_shape = tuple(self.snapshot['actor_spec']['observation_shape'])
        if observation.shape != expected_shape:
            raise ValueError(
                f'Policy {self.path} expects observations shaped {expected_shape}, '
                f'but the environment produced {observation.shape}.',
            )
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        tensor, _ = read_only_normalize(tensor, self.snapshot)
        with torch.no_grad():
            action = self.actor.predict(tensor, deterministic=True)
        action = action.cpu().numpy().reshape(-1)
        expected_shape = tuple(self.snapshot['actor_spec']['action_shape'])
        if action.shape != expected_shape or not np.all(np.isfinite(action)):
            raise ValueError(
                f'Policy {self.path} produced an invalid action shaped {action.shape}.',
            )
        return action


def _native_action(action: np.ndarray, action_space: gymnasium.spaces.Box) -> np.ndarray:
    """Map a normalized actor/controller action from [-1, 1] to the environment bounds."""
    normalized = np.asarray(action, dtype=np.float64)
    if normalized.shape != action_space.shape or not np.all(np.isfinite(normalized)):
        raise ValueError(f'Controller produced an invalid action shaped {normalized.shape}.')
    normalized = np.clip(normalized, -1.0, 1.0)
    return np.asarray(
        action_space.low
        + 0.5 * (normalized + 1.0) * (action_space.high - action_space.low),
        dtype=action_space.dtype,
    )


def _evasive_action(scene: Scene, planner_config: dict[str, Any], turn_gain: float) -> np.ndarray:
    """Move toward the goal while steering along the nearest clear goal-length ray."""
    targets = local_targets(scene, planner_config)
    evasive_turn = targets['minimum_evasive_turn']
    if not np.isfinite(evasive_turn):
        evasive_turn = 0.0
    goal_direction = scene.goal_xy - scene.agent_xy
    goal_turn = signed_angle(scene.heading_xy, goal_direction)
    desired_turn = np.arctan2(
        np.sin(goal_turn + evasive_turn),
        np.cos(goal_turn + evasive_turn),
    )
    steering = np.clip(turn_gain * desired_turn, -1.0, 1.0)
    forward = 1.0 if abs(desired_turn) < np.pi / 2.0 else 0.25
    return np.array([forward, steering], dtype=np.float32)


def _source_counts(total: int) -> dict[str, int]:
    """Divide the candidate pool as evenly as possible between the three sources."""
    quotient, remainder = divmod(total, len(SOURCES))
    return {
        source: quotient + int(index < remainder)
        for index, source in enumerate(SOURCES)
    }


def collect_dataset(
    config: dict[str, Any],
    visualization_directory: Path | None = None,
    converged_policy_paths: list[Path] | None = None,
    artifact_version: str | None = None,
) -> dict[str, np.ndarray]:
    """Collect a large three-source pool and retain a uniform random subset."""
    dataset_config = config['dataset']
    collection_config = config['collection']
    planner_config = config['planner']
    visualization_config = config.get('visualization', {})
    switches = task_switches(config)
    num_samples = int(dataset_config['num_samples'])
    pool_multiplier = int(dataset_config['candidate_pool_multiplier'])
    candidate_pool_size = pool_multiplier * num_samples
    seed = int(dataset_config['seed'])
    max_steps = int(dataset_config['max_steps_per_episode'])
    agent_radius = float(dataset_config['agent_radius'])
    action_noise_std = float(collection_config['action_noise_std'])
    evasive_turn_gain = float(collection_config['evasive_turn_gain'])
    if num_samples <= 0 or max_steps <= 0 or pool_multiplier <= 0:
        raise ValueError('Sample count, episode length, and pool multiplier must be positive.')
    if candidate_pool_size < len(SOURCES):
        raise ValueError('The candidate pool must contain at least one sample per source.')
    if action_noise_std < 0.0:
        raise ValueError('`collection.action_noise_std` cannot be negative.')
    if not np.isfinite(action_noise_std):
        raise ValueError('`collection.action_noise_std` must be finite.')
    if not np.isfinite(evasive_turn_gain) or evasive_turn_gain < 0.0:
        raise ValueError('`collection.evasive_turn_gain` must be finite and non-negative.')
    if not np.isfinite(agent_radius) or agent_radius <= 0.0:
        raise ValueError('`dataset.agent_radius` must be finite and positive.')
    validate_planner_config(planner_config)

    policy_paths = converged_policy_paths or [
        Path(path) for path in collection_config.get('converged_policy_paths', [])
    ]
    policy_paths = [path.expanduser().resolve() for path in policy_paths]
    if not policy_paths:
        raise ValueError(
            'No converged policies found. Configure `collection.converged_policy_paths` or '
            '`snapshots.collection_directory`.',
        )
    policy_device = config['snapshots']['device']
    policies = [RestoredPolicy(path, policy_device) for path in policy_paths]

    selection_generator = np.random.default_rng(seed)
    selected_candidates = np.sort(
        selection_generator.choice(candidate_pool_size, size=num_samples, replace=False),
    )
    selected_lookup = {
        int(candidate_index): final_index
        for final_index, candidate_index in enumerate(selected_candidates)
    }

    num_examples = int(visualization_config.get('num_examples', 0))
    if num_examples < 0:
        raise ValueError('`visualization.num_examples` cannot be negative.')
    if num_examples > num_samples:
        raise ValueError('`visualization.num_examples` cannot exceed `dataset.num_samples`.')
    if num_examples and (
        int(visualization_config.get('width', 800)) <= 0
        or int(visualization_config.get('height', 800)) <= 0
    ):
        raise ValueError('Visualization width and height must be positive.')
    visualization_generator = np.random.default_rng(int(visualization_config.get('seed', 0)))
    visualization_indices = set(
        visualization_generator.choice(
            num_samples,
            size=num_examples,
            replace=False,
        ).tolist(),
    )
    visualization_candidates = {
        int(selected_candidates[index]): index for index in visualization_indices
    }

    env_kwargs = dict(dataset_config.get('env_kwargs', {}))
    if env_kwargs.get('autoreset') not in (None, False):
        raise ValueError('Dataset collection requires `env_kwargs.autoreset: false`.')
    env_kwargs['autoreset'] = False
    if visualization_indices:
        env_kwargs.update(
            {
                'render_mode': 'rgb_array',
                'camera_name': visualization_config.get('camera_name', 'fixedfar'),
                'width': int(visualization_config.get('width', 800)),
                'height': int(visualization_config.get('height', 800)),
            },
        )
        env_kwargs.pop('camera_id', None)
    env = safety_gymnasium.make(dataset_config['env_id'], **env_kwargs)
    try:
        if not isinstance(env.action_space, gymnasium.spaces.Box):
            raise TypeError('Candidate collection requires a continuous Box action space.')
        if env.action_space.shape != (2,):
            raise ValueError(
                'The simple evasive controller requires a two-dimensional action space.',
            )
        if (
            not np.all(np.isfinite(env.action_space.low))
            or not np.all(np.isfinite(env.action_space.high))
            or not np.all(env.action_space.low < env.action_space.high)
        ):
            raise ValueError('Candidate collection requires finite, ordered action bounds.')
        env.action_space.seed(seed)
        validation_observation, _ = env.reset(seed=seed)
        observation_shape = _flat_observation(env, validation_observation).shape
        for policy in policies:
            policy.validate_environment(
                observation_shape,
                tuple(env.action_space.shape),
                dataset_config['env_id'],
            )
        agent_dynamics = AgentDynamics(_task_from_env(env))
        action_space_metadata = {
            'shape': list(env.action_space.shape),
            'dtype': str(env.action_space.dtype),
            'low': env.action_space.low.tolist(),
            'high': env.action_space.high.tolist(),
        }
    except Exception:
        env.close()
        raise

    noise_generator = np.random.default_rng(seed + 1)

    records: dict[str, list[Any]] = {
        'observations': [],
        'candidate_indices': [],
        'episode_ids': [],
        'steps': [],
        'sources': [],
        'policy_snapshots': [],
        'agent_xy': [],
        'heading_xy': [],
        'agent_velocity_xy': [],
        'agent_velocity': [],
        'goal_xy': [],
        'hazards_xy': [],
        'agent_radius': [],
        'hazard_radius': [],
        'goal_radius': [],
    }
    target_records: dict[str, list[Any]] = {}
    visualization_examples: list[tuple[int, np.ndarray, Scene, dict[str, Any]]] = []
    planners: dict[tuple[bytes, bytes, float], GridPathPlanner] = {}
    source_candidate_counts = _source_counts(candidate_pool_size)
    next_episode_id = 0
    candidate_index = 0
    hazard_count: int | None = None

    try:
        for source in SOURCES:
            episode_id = next_episode_id
            next_episode_id += 1
            step = 0
            source_episode = 0
            observation, _ = env.reset(seed=seed + episode_id)

            for _ in range(source_candidate_counts[source]):
                scene = _scene_from_env(env, agent_radius)
                flat_observation = _flat_observation(env, observation)
                if hazard_count is None:
                    hazard_count = len(scene.hazards_xy)
                elif len(scene.hazards_xy) != hazard_count:
                    raise ValueError(
                        'The number of hazards changed; use one fixed environment configuration.',
                    )

                policy = policies[source_episode % len(policies)] if source == 'converged' else None
                final_index = selected_lookup.get(candidate_index)
                if final_index is not None:
                    records['observations'].append(flat_observation)
                    records['candidate_indices'].append(candidate_index)
                    records['episode_ids'].append(episode_id)
                    records['steps'].append(step)
                    records['sources'].append(source)
                    records['policy_snapshots'].append(str(policy.path) if policy else '')
                    records['agent_xy'].append(scene.agent_xy)
                    records['heading_xy'].append(scene.heading_xy)
                    records['agent_velocity_xy'].append(scene.agent_velocity_xy)
                    records['agent_velocity'].append(scene.agent_velocity)
                    records['goal_xy'].append(scene.goal_xy)
                    records['hazards_xy'].append(scene.hazards_xy)
                    records['agent_radius'].append(scene.agent_radius)
                    records['hazard_radius'].append(scene.hazard_radius)
                    records['goal_radius'].append(scene.goal_radius)

                    targets = local_targets(scene, planner_config)
                    targets['agent_kinetic_energy'] = agent_dynamics.kinetic_energy()
                    key = _planner_key(scene)
                    if key not in planners:
                        planners[key] = GridPathPlanner(scene, planner_config)
                    targets['safe_path_detour_ratio'] = planners[key].detour_ratio(
                        scene.agent_xy,
                        scene.goal_xy,
                    )
                    for name, value in targets.items():
                        target_records.setdefault(name, []).append(value)

                    if candidate_index in visualization_candidates:
                        frame = env.render()
                        if frame is None:
                            raise RuntimeError(
                                'Safety-Gymnasium returned no RGB frame for visualization.',
                            )
                        visualization_examples.append(
                            (final_index, np.asarray(frame).copy(), scene, targets.copy()),
                        )

                if source == 'random':
                    action = env.action_space.sample()
                else:
                    normalized_action = (
                        _evasive_action(
                            scene,
                            planner_config,
                            evasive_turn_gain,
                        )
                        if source == 'evasive'
                        else policy.predict(flat_observation)
                    )
                    normalized_action += noise_generator.normal(
                        scale=action_noise_std,
                        size=normalized_action.shape,
                    )
                    action = _native_action(normalized_action, env.action_space)

                transition = env.step(action)
                if len(transition) == 6:
                    observation, _, _, terminated, truncated, _ = transition
                else:
                    observation, _, terminated, truncated, _ = transition
                step += 1
                candidate_index += 1

                if terminated or truncated or step >= max_steps:
                    episode_id = next_episode_id
                    next_episode_id += 1
                    source_episode += 1
                    step = 0
                    observation, _ = env.reset(seed=seed + episode_id)

                if candidate_index % max(1, candidate_pool_size // 10) == 0:
                    print(f'Generated {candidate_index}/{candidate_pool_size} candidates')
    finally:
        env.close()

    if len(records['observations']) != num_samples:
        raise RuntimeError('Uniform candidate sampling did not produce the requested sample count.')

    raw_dispersion = np.asarray(target_records.pop('hazard_dispersion_raw'), dtype=np.float32)
    lower, upper = float(np.min(raw_dispersion)), float(np.max(raw_dispersion))
    if upper > lower:
        dispersion = (raw_dispersion - lower) / (upper - lower)
    else:
        dispersion = np.zeros_like(raw_dispersion)
    target_records['hazard_dispersion'] = dispersion.tolist()

    if visualization_examples and visualization_directory is None:
        visualization_directory = Path(visualization_config['output_directory'])
    if visualization_directory is not None:
        clear_explanation_figures(visualization_directory)
    if visualization_examples:
        assert visualization_directory is not None
        enabled_tasks = [task for task in TASK_TYPES if switches[task]]
        for sample_index, frame, scene, targets in visualization_examples:
            raw_value = targets['hazard_dispersion_raw']
            targets['hazard_dispersion'] = (
                (raw_value - lower) / (upper - lower) if upper > lower else 0.0
            )
            save_explanation_figure(
                frame=frame,
                scene=scene,
                targets=targets,
                enabled_tasks=enabled_tasks,
                planner_config=planner_config,
                output_path=visualization_directory / f'sample-{sample_index:06d}.png',
                sample_index=sample_index,
            )
        print(
            f'Saved {len(visualization_examples)} explanation figures '
            f'to {visualization_directory}',
        )

    arrays = {name: np.asarray(values) for name, values in records.items()}
    if arrays['observations'].ndim != 2 or not np.all(np.isfinite(arrays['observations'])):
        raise ValueError('Collected observations must be a finite two-dimensional array.')
    arrays['hazard_dispersion_raw'] = raw_dispersion
    arrays.update(
        {
            f'target_{name}': np.asarray(values, dtype=np.float32)
            for name, values in target_records.items()
        },
    )
    collection_policies = []
    for policy in policies:
        training_config = policy.snapshot.get('config', {})
        collection_policies.append(
            {
                'path': str(policy.path),
                'run_directory': str(snapshot_run_directory(policy.path)),
                'sha256': file_sha256(policy.path) if policy.path.is_file() else None,
                'algorithm': training_config.get('algo'),
                'environment': training_config.get('env_id'),
                'training_seed': training_config.get('seed'),
            },
        )
    metadata = {
        'artifact_version': artifact_version,
        'schema_version': DATASET_SCHEMA_VERSION,
        'target_schema_version': TARGET_SCHEMA_VERSION,
        'env_id': dataset_config['env_id'],
        'num_samples': num_samples,
        'candidate_pool_size': candidate_pool_size,
        'candidate_pool_multiplier': pool_multiplier,
        'seed': seed,
        'observation_is_raw': True,
        'observation_shape': list(arrays['observations'].shape[1:]),
        'observation_dtype': str(arrays['observations'].dtype),
        'action_space': action_space_metadata,
        'hazard_dispersion_raw_min': lower,
        'hazard_dispersion_raw_max': upper,
        'best_avoidance_side_labels': {'right': -1, 'left': 1},
        'target_units': {
            'angles': 'radians',
            'distances': 'world coordinates',
            'agent_velocity': 'MuJoCo local-frame units per second',
            'agent_kinetic_energy': 'MuJoCo energy units',
        },
        'target_definitions': TARGET_DEFINITIONS,
        'agent_kinetic_energy_dof_indices': agent_dynamics.dof_indices.tolist(),
        'signed_slip_speed_epsilon': SLIP_SPEED_EPSILON,
        'target_types': TASK_TYPES,
        'task_groups': TASK_GROUPS,
        'task_switches': switches,
        'dataset_config': dataset_config,
        'collection_config': collection_config,
        'collection_snapshot_selection': config['snapshots']['collection_selection'],
        'planner_config': planner_config,
        'visualization_config': visualization_config,
        'visualization_output_directory': (
            str(visualization_directory.resolve())
            if visualization_directory is not None
            else None
        ),
        'visualization_indices': sorted(visualization_indices),
        'source_candidate_counts': source_candidate_counts,
        'source_sample_counts': {
            source: records['sources'].count(source) for source in SOURCES
        },
        'collection_policy_paths': [str(path) for path in policy_paths],
        'collection_run_directories': [
            str(snapshot_run_directory(path)) for path in policy_paths
        ],
        'collection_policies': collection_policies,
        'reproducibility': reproducibility_metadata(Path(__file__).resolve().parent.parent),
        'array_schema': {
            name: {'shape': list(array.shape), 'dtype': str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    arrays['metadata_json'] = np.asarray(
        json.dumps(metadata),
    )
    return arrays


def main() -> None:
    """Create and save the configured probe dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    artifact_version = new_version_id()
    output_base_path = resolve_path(config['dataset']['output_path'], args.config)
    metadata_base_path = resolve_path(config['dataset']['metadata_path'], args.config)
    output_path = versioned_path(output_base_path, artifact_version)
    metadata_path = versioned_path(metadata_base_path, artifact_version)
    visualization_config = config.get('visualization', {})
    visualization_directory = None
    if 'output_directory' in visualization_config:
        visualization_directory = versioned_directory(
            resolve_path(visualization_config['output_directory'], args.config),
            artifact_version,
        )
    configured_policy_paths = config['collection'].get('converged_policy_paths', [])
    if configured_policy_paths:
        converged_policy_paths = [
            resolve_path(path, args.config) for path in configured_policy_paths
        ]
    else:
        collection_directory = resolve_path(
            config['snapshots']['collection_directory'],
            args.config,
        )
        snapshot_paths = sorted(collection_directory.glob(config['snapshots']['pattern']))
        converged_policy_paths = select_snapshots(
            snapshot_paths,
            config['snapshots']['collection_selection'],
        )
        if not converged_policy_paths:
            raise FileNotFoundError(
                'No collection snapshots matched the configured directory, pattern, and '
                f'selection under {collection_directory}.',
            )
    probing_directory = resolve_path(config['snapshots']['probing_directory'], args.config)
    probing_paths = select_snapshots(
        sorted(probing_directory.glob(config['snapshots']['pattern'])),
        config['snapshots']['probing_selection'],
    )
    ensure_disjoint_snapshot_runs(converged_policy_paths, probing_paths)
    version_directories = {output_path.parent, metadata_path.parent}
    if visualization_directory is not None:
        version_directories.add(visualization_directory)
    existing_directories = sorted(path for path in version_directories if path.exists())
    if existing_directories:
        raise FileExistsError(
            f'Artifact version {artifact_version} already exists: {existing_directories}',
        )
    for directory in sorted(version_directories):
        directory.mkdir(parents=True, exist_ok=False)
    arrays = collect_dataset(
        config,
        visualization_directory,
        converged_policy_paths,
        artifact_version,
    )
    save_dataset_artifact(output_path, metadata_path, arrays)
    print(f'Dataset version: {artifact_version}')
    print(f'Saved {len(arrays["observations"])} samples to {output_path}')
    print(f'Saved dataset metadata to {metadata_path}')


if __name__ == '__main__':
    main()
