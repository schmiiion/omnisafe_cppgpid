"""Focused regression tests for the linear-probing pipeline."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from linear_probing import targets
from linear_probing.artifacts import (
    VERSION_PATTERN,
    new_version_id,
    resolve_dataset_version,
    resolve_versioned_file,
    versioned_directory,
    versioned_path,
)
from linear_probing.aux_only_plotting import load_auxiliary_scores, split_phase_thirds
from linear_probing.config import DEFAULT_CONFIG_PATH, load_config, task_switches
from linear_probing.dataset_io import (
    DATASET_SCHEMA_VERSION,
    TARGET_SCHEMA_VERSION,
    validate_dataset_artifact,
)
from linear_probing.policy_only_plotting import flatten_policy_timeline
from linear_probing.run_probes import fit_probe, grouped_train_test_split
from linear_probing.snapshots import read_only_normalize, select_snapshots
from linear_probing.targets import (
    TASK_GROUPS,
    TASK_TYPES,
    GridPathPlanner,
    Scene,
    point_segment_distances,
)


def _scene(
    *,
    agent: tuple[float, float] = (0.0, 0.0),
    goal: tuple[float, float] = (1.0, 0.0),
    hazards: tuple[tuple[float, float], ...] = (),
    agent_radius: float = 0.1,
    hazard_radius: float = 0.1,
) -> Scene:
    return Scene(
        agent_xy=np.asarray(agent, dtype=np.float64),
        heading_xy=np.array([1.0, 0.0]),
        agent_velocity_xy=np.zeros(2),
        agent_velocity=np.zeros(2),
        goal_xy=np.asarray(goal, dtype=np.float64),
        hazards_xy=np.asarray(hazards, dtype=np.float64).reshape(-1, 2),
        agent_radius=agent_radius,
        hazard_radius=hazard_radius,
        goal_radius=0.3,
    )


def _planner_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        'bounds': [0.0, 1.0, 0.0, 1.0],
        'grid_resolution': 1.0,
        'avoidance_angle_step_degrees': 1.0,
        'maximum_avoidance_angle_degrees': 90.0,
    }
    config.update(overrides)
    return config


class TestLinearProbingCorrectness(TestCase):
    """Protect invariants that previously failed silently."""

    def test_clear_direct_path_has_unit_detour_ratio(self) -> None:
        scene = _scene(hazards=((0.5, 0.8),))
        planner = GridPathPlanner(scene, _planner_config())

        self.assertEqual(planner.detour_ratio(scene.agent_xy, scene.goal_xy), 1.0)
        np.testing.assert_allclose(
            planner.shortest_path(scene.agent_xy, scene.goal_xy),
            np.stack((scene.agent_xy, scene.goal_xy)),
        )

    def test_coarse_grid_does_not_cross_hazard_between_free_nodes(self) -> None:
        scene = _scene(hazards=((0.5, 0.0),))
        planner = GridPathPlanner(scene, _planner_config())

        ratio = planner.detour_ratio(scene.agent_xy, scene.goal_xy)
        path = planner.shortest_path(scene.agent_xy, scene.goal_xy)

        self.assertGreater(ratio, 1.0)
        self.assertGreater(len(path), 2)
        for start, end in zip(path[:-1], path[1:]):
            distances = point_segment_distances(scene.hazards_xy, start, end)
            self.assertTrue(np.all(distances > scene.expanded_hazard_radius))

    def test_occupied_planner_start_is_undefined(self) -> None:
        scene = _scene(
            goal=(1.0, 0.0),
            hazards=((0.0, 0.0),),
            agent_radius=0.1,
            hazard_radius=0.2,
        )
        planner = GridPathPlanner(scene, _planner_config())

        self.assertTrue(np.isnan(planner.detour_ratio(scene.agent_xy, scene.goal_xy)))
        self.assertEqual(planner.shortest_path(scene.agent_xy, scene.goal_xy).size, 0)

    def test_evasive_search_never_exceeds_maximum_angle(self) -> None:
        checked_angles: list[float] = []

        def always_blocked(_scene: Scene, direction: np.ndarray | None = None) -> np.ndarray:
            assert direction is not None
            checked_angles.append(abs(float(np.arctan2(direction[1], direction[0]))))
            return np.array([-1.0])

        with patch.object(targets, 'direct_path_clearances', always_blocked):
            angle, side = targets.minimum_evasive_turn(
                _scene(),
                _planner_config(
                    avoidance_angle_step_degrees=7.0,
                    maximum_avoidance_angle_degrees=90.0,
                ),
                direct_path_blocked=True,
            )

        self.assertTrue(np.isnan(angle))
        self.assertTrue(np.isnan(side))
        self.assertTrue(checked_angles)
        self.assertLessEqual(max(checked_angles), np.deg2rad(90.0) + 1e-12)

    def test_dataset_rejects_outdated_schema(self) -> None:
        cases = (
            (DATASET_SCHEMA_VERSION - 1, TARGET_SCHEMA_VERSION, 'Unsupported dataset schema'),
            (DATASET_SCHEMA_VERSION, TARGET_SCHEMA_VERSION - 1, 'Outdated target definitions'),
        )
        with TemporaryDirectory() as directory:
            temporary_directory = Path(directory)
            for schema_version, target_schema_version, message in cases:
                with self.subTest(message=message):
                    metadata = {
                        'schema_version': schema_version,
                        'target_schema_version': target_schema_version,
                        'observation_is_raw': True,
                        'num_samples': 1,
                        'array_schema': {},
                    }
                    dataset = {
                        'observations': np.zeros((1, 2), dtype=np.float32),
                        'metadata_json': np.asarray(json.dumps(metadata)),
                    }

                    with self.assertRaisesRegex(ValueError, message):
                        validate_dataset_artifact(
                            temporary_directory / 'dataset.npz',
                            temporary_directory / 'dataset.metadata.json',
                            dataset,
                        )

    def test_default_task_config_matches_exact_grouping(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        grouped = config['tasks']

        self.assertEqual(tuple(grouped), tuple(TASK_GROUPS))
        for group, difficulties in TASK_GROUPS.items():
            self.assertEqual(tuple(grouped[group]), tuple(difficulties))
            for difficulty, expected_tasks in difficulties.items():
                self.assertEqual(tuple(grouped[group][difficulty]), expected_tasks)
        self.assertEqual(tuple(task_switches(config)), tuple(TASK_TYPES))
        self.assertEqual(config['dataset']['version'], 'latest')

        incomplete = deepcopy(config)
        incomplete['tasks']['agent_related']['easy'].pop('agent_velocity')
        with self.assertRaisesRegex(ValueError, 'tasks must be exactly'):
            task_switches(incomplete)

    def test_timestamped_artifact_paths(self) -> None:
        version = new_version_id()

        self.assertRegex(version, VERSION_PATTERN)
        self.assertEqual(
            versioned_path(Path('/tmp/data/probe.npz'), version),
            Path('/tmp/data') / version / 'probe.npz',
        )
        self.assertEqual(
            versioned_directory(Path('/tmp/figures'), version),
            Path('/tmp/figures') / version,
        )

    def test_latest_dataset_version_ignores_incomplete_runs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_base = root / 'probe.npz'
            metadata_base = root / 'probe.metadata.json'
            complete = '20260101T120000.000001Z'
            incomplete = '20260101T120000.000002Z'
            (root / complete).mkdir()
            (root / complete / dataset_base.name).touch()
            (root / complete / metadata_base.name).touch()
            (root / incomplete).mkdir()
            (root / incomplete / dataset_base.name).touch()

            dataset_path, metadata_path, version = resolve_dataset_version(
                dataset_base,
                metadata_base,
                'latest',
            )

            self.assertEqual(version, complete)
            self.assertEqual(dataset_path, root / complete / dataset_base.name)
            self.assertEqual(metadata_path, root / complete / metadata_base.name)

    def test_exact_and_unversioned_dataset_resolution(self) -> None:
        dataset_base = Path('/tmp/data/probe.npz')
        metadata_base = Path('/tmp/data/probe.metadata.json')
        version = '20260101T120000.123456Z'

        self.assertEqual(
            resolve_dataset_version(dataset_base, metadata_base, version),
            (
                dataset_base.parent / version / dataset_base.name,
                metadata_base.parent / version / metadata_base.name,
                version,
            ),
        )
        self.assertEqual(
            resolve_dataset_version(dataset_base, metadata_base, 'unversioned'),
            (dataset_base, metadata_base, None),
        )

    def test_latest_versioned_file_resolution(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'results.csv'
            older = root / '20260101T120000.000001Z'
            newer = root / '20260101T120000.000002Z'
            older.mkdir()
            newer.mkdir()
            (older / base.name).touch()

            self.assertEqual(
                resolve_versioned_file(base, 'latest'),
                (older / base.name, older.name),
            )

    def test_auxiliary_plot_data_excludes_pre_auxiliary_snapshot(self) -> None:
        fieldnames = [
            'algorithm',
            'environment',
            'training_seed',
            'run_directory',
            'result_version',
            'phase',
            'stage',
            'stage_iteration',
            'task',
            'task_type',
            'r2',
            'balanced_accuracy',
        ]
        rows = []
        for run, seed, run_offset in (('/tmp/run-1', 1, 0.0), ('/tmp/run-2', 2, 0.1)):
            for phase in (0, 1):
                for epoch in (0, 1, 2):
                    for task, task_type in (
                        ('goal_distance', 'regression'),
                        ('direct_path_blocked', 'classification'),
                    ):
                        rows.append(
                            {
                                'algorithm': 'CPPGPID',
                                'environment': 'SafetyPointGoal1-v0',
                                'training_seed': str(seed),
                                'run_directory': run,
                                'result_version': '20260101T120000.000001Z',
                                'phase': str(phase),
                                'stage': 'auxiliary',
                                'stage_iteration': str(epoch),
                                'task': task,
                                'task_type': task_type,
                                'r2': str(phase * 0.2 + epoch / 10 + run_offset),
                                'balanced_accuracy': str(0.5 + epoch / 10 + run_offset / 2),
                            },
                        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            with path.open('w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            runs, phases, epochs, tasks, matrices, identity = load_auxiliary_scores(path)

        self.assertEqual(runs, ['/tmp/run-1', '/tmp/run-2'])
        self.assertEqual(phases, [0, 1])
        self.assertEqual(epochs, [1, 2])
        self.assertEqual(tasks, ['goal_distance', 'direct_path_blocked'])
        np.testing.assert_allclose(
            matrices['goal_distance'],
            [[[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.3], [0.4, 0.5]]],
        )
        np.testing.assert_allclose(
            matrices['direct_path_blocked'],
            [[[0.6, 0.7], [0.6, 0.7]], [[0.65, 0.75], [0.65, 0.75]]],
        )
        self.assertEqual(identity['algorithm'], 'CPPGPID')
        self.assertEqual(identity['training_seeds'], [1, 2])

    def test_auxiliary_phases_are_split_into_chronological_thirds(self) -> None:
        groups = split_phase_thirds(list(range(8)))

        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5], [6, 7]])
        self.assertLessEqual(max(map(len, groups)) - min(map(len, groups)), 1)

    def test_policy_timeline_is_flattened_in_phase_major_order(self) -> None:
        values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)

        coordinates, timeline = flatten_policy_timeline(
            runs=['run-1', 'run-2'],
            phases=[0, 1],
            iterations=[0, 1, 2],
            matrices={'goal_distance': values},
        )

        self.assertEqual(coordinates, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        np.testing.assert_array_equal(timeline['goal_distance'], values.reshape(2, 6))

    def test_required_normalizer_state_cannot_be_missing(self) -> None:
        snapshot = {
            'actor_spec': {
                'observation_shape': (2,),
                'normalizer_state_dict_key': 'obs_normalizer',
            },
            'metadata': {'observation_normalized': True},
        }

        with self.assertRaisesRegex(
            ValueError,
            'missing its required observation-normalizer state',
        ):
            read_only_normalize(torch.zeros(2), snapshot)

    def test_non_integral_normalizer_count_is_rejected(self) -> None:
        snapshot = {
            'actor_spec': {
                'observation_shape': (2,),
                'normalizer_state_dict_key': 'obs_normalizer',
            },
            'metadata': {'observation_normalized': True},
            'obs_normalizer': {
                '_count': torch.tensor(1.5),
                '_mean': torch.zeros(2),
                '_std': torch.ones(2),
                '_clip': torch.ones(2),
            },
        }

        with self.assertRaisesRegex(ValueError, 'finite non-negative integer'):
            read_only_normalize(torch.zeros(2), snapshot)

    def test_grouped_split_preserves_every_collection_source(self) -> None:
        episodes = np.repeat(np.arange(6), 2)
        sources = np.repeat(['random', 'random', 'evasive', 'evasive', 'converged', 'converged'], 2)
        train, test = grouped_train_test_split(episodes, 0.5, 3, sources)

        self.assertFalse(np.any(train & test))
        for source in np.unique(sources):
            self.assertTrue(np.any(train & (sources == source)))
            self.assertTrue(np.any(test & (sources == source)))
        self.assertFalse(set(episodes[train]) & set(episodes[test]))

    def test_explicitly_missing_snapshot_phase_is_rejected(self) -> None:
        paths = [
            Path('/tmp/run/representation_checkpoints/phase-001/aux-epoch-000.pt'),
        ]
        selection = {
            'phases': [2],
            'stages': ['auxiliary'],
            'policy_iterations': 'all',
            'auxiliary_epochs': 'all',
        }

        with self.assertRaisesRegex(ValueError, 'phases are missing'):
            select_snapshots(paths, selection)

    def test_probe_reports_degenerate_finite_split_without_fitting(self) -> None:
        metrics = fit_probe(
            embeddings=np.zeros((4, 2), dtype=np.float32),
            target=np.full(4, np.nan, dtype=np.float32),
            task_type='regression',
            train_mask=np.array([True, True, False, False]),
            test_mask=np.array([False, False, True, True]),
            probing_config={},
        )

        self.assertEqual(metrics['status'], 'fewer than two finite train/test samples')
        self.assertEqual(metrics['n_train'], 0)
        self.assertEqual(metrics['n_test'], 0)

    def test_probe_rejects_constant_regression_target(self) -> None:
        metrics = fit_probe(
            embeddings=np.arange(12, dtype=np.float32).reshape(6, 2),
            target=np.ones(6, dtype=np.float32),
            task_type='regression',
            train_mask=np.array([True, True, True, False, False, False]),
            test_mask=np.array([False, False, False, True, True, True]),
            probing_config={},
        )

        self.assertEqual(metrics['status'], 'constant training target')
        self.assertTrue(np.isnan(metrics['r2']))

    def test_probe_rejects_single_class_training_split(self) -> None:
        metrics = fit_probe(
            embeddings=np.arange(12, dtype=np.float32).reshape(6, 2),
            target=np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
            task_type='classification',
            train_mask=np.array([True, True, True, False, False, False]),
            test_mask=np.array([False, False, False, True, True, True]),
            probing_config={},
        )

        self.assertEqual(
            metrics['status'],
            'training split does not contain exactly two classes',
        )
        self.assertTrue(np.isnan(metrics['accuracy']))

    def test_circular_probe_handles_wraparound_angles(self) -> None:
        angles = np.array(
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -np.pi + 0.05, np.pi - 0.05],
            dtype=np.float64,
        )
        embeddings = np.column_stack((np.sin(angles), np.cos(angles)))
        metrics = fit_probe(
            embeddings=embeddings,
            target=angles,
            task_type='circular_regression',
            train_mask=np.array([True] * 6 + [False] * 2),
            test_mask=np.array([False] * 6 + [True] * 2),
            probing_config={'ridge_alpha': 1e-12},
        )

        self.assertEqual(metrics['status'], 'ok')
        self.assertLess(metrics['mae'], 1e-8)
