"""Checkpoint/metadata loading and raw demonstration export without a simulator."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.common import Normalizer
from omnisafe.models.actor import ActorBuilder
from ICRL.expert.weights2trajectories import sample_trajectories
from ICRL.utils.data import load_expert_data


class ExpertSamplingEnv:
    observation_space = Box(-100, 100, shape=(3,))
    action_space = Box(np.array([-2., 2.], dtype=np.float32),
                       np.array([2., 6.], dtype=np.float32))
    max_episode_steps = 4

    def __init__(self, costs_by_seed=None):
        self.seeds = []
        self.executed_actions = []
        self.closed = False
        self.costs_by_seed = costs_by_seed or {}

    def reset(self, seed):
        self.seeds.append(seed)
        self.step_costs = self.costs_by_seed.get(seed, [0.] * (2 + seed % 2))
        self.length = len(self.step_costs)
        self.count = 0
        self.obs = torch.tensor([4., 8., 10.])
        return self.obs, {}

    def step(self, action):
        self.executed_actions.append(action.clone().numpy())
        self.count += 1
        self.obs[0] += 1  # Reuse/mutate the observation storage, as simulators may.
        return (self.obs, torch.tensor(2.), torch.tensor(self.step_costs[self.count - 1]),
                torch.tensor(self.count == self.length), torch.tensor(False), {})

    def close(self):
        self.closed = True


class ModelBasedSamplingEnv:
    observation_space = Box(-100, 100, shape=(3,))
    action_space = Box(-1., 1., shape=(2,))
    max_episode_steps = 7

    def __init__(self):
        self.actions = []
        self.closed = False

    def reset(self, seed):
        self.seed = seed
        self.step_count = 0
        return torch.tensor([1., 0., 0.]), {'obs_original': self._raw_obs()}

    def _raw_obs(self):
        return np.array([self.seed, self.step_count, 42., 43.], dtype=np.float32)

    def step(self, action):
        self.actions.append(action.clone().numpy())
        self.step_count += 1
        coordinate_obs = torch.tensor([1. + self.step_count, 0., 0.])
        cost = float(self.seed == 17 and self.step_count == 3)
        return (coordinate_obs, torch.tensor(2.), torch.tensor(cost),
                torch.tensor(False), torch.tensor(self.step_count == 7),
                {'obs_original': self._raw_obs()})

    def close(self):
        self.closed = True


class ExpertSamplingTests(unittest.TestCase):
    def make_checkpoint(self, run_dir, normalize=False):
        weights = run_dir / 'torch_save/epoch-12.pt'
        weights.parent.mkdir()
        cfg = {
            'env_id': 'MetadataSelectedEnv-v0', 'seed': 17, 'algo': 'CPPOPID',
            'env_cfgs': {'sensor_scale': 2},
            'train_cfgs': {'device': 'cpu', 'torch_threads': 1, 'vector_env_nums': 8},
            'algo_cfgs': {'obs_normalize': normalize},
            'model_cfgs': {'actor_type': 'gaussian_learning',
                           'weight_initialization_mode': 'kaiming_uniform',
                           'actor': {'hidden_sizes': [], 'activation': 'tanh'}},
        }
        (run_dir / 'config.json').write_text(json.dumps(cfg))
        actor = ActorBuilder(ExpertSamplingEnv.observation_space, Box(-1., 1., shape=(2,)),
                             hidden_sizes=[], activation='tanh').build_actor('gaussian_learning')
        with torch.no_grad():
            actor.mean[0].weight.zero_()
            actor.mean[0].weight[0, 0] = .25
            actor.mean[0].bias.copy_(torch.tensor([0., 2.]))
        checkpoint = {'pi': actor.state_dict()}
        if normalize:
            normalizer = Normalizer((3,), clip=5)
            normalizer._mean.copy_(torch.tensor([2., 4., 6.]))
            normalizer._std.fill_(2)
            normalizer._var.fill_(4)
            normalizer._count.fill_(10)
            checkpoint['obs_normalizer'] = normalizer.state_dict()
        torch.save(checkpoint, weights)
        return weights, cfg, checkpoint

    def test_metadata_frozen_normalization_native_actions_and_icrl_loading(self):
        for normalize in (True, False):
            with self.subTest(normalize=normalize), tempfile.TemporaryDirectory() as temp:
                run_dir = Path(temp)
                env = ExpertSamplingEnv()
                weights, cfg, checkpoint = self.make_checkpoint(run_dir, normalize)
                created_normalizers = []

                def build_normalizer(shape):
                    result = Normalizer(shape)
                    created_normalizers.append(result)
                    return result

                with redirect_stdout(io.StringIO()), patch(
                    'ICRL.expert.weights2trajectories.make', return_value=env,
                ) as make, patch(
                    'ICRL.expert.weights2trajectories.Normalizer', side_effect=build_normalizer,
                ), patch(
                    'ICRL.expert.weights2trajectories.__file__',
                    str(Path(temp) / 'expert/weights2trajectories.py'),
                ):
                    output = sample_trajectories(weights, 3)
                self.assertEqual(output, Path(temp).resolve() / 'expert/expert_trajectories' /
                                 f'CPPOPID-MetadataSelectedEnv-v0__{run_dir.name}__epoch-12' /
                                 'epoch-12-3-episodes.npz')
                make.assert_called_once_with('MetadataSelectedEnv-v0', num_envs=1,
                                             device=torch.device('cpu'), sensor_scale=2)
                self.assertTrue(env.closed)
                self.assertEqual(env.seeds, [17, 18, 19])
                obs, actions = load_expert_data(output)
                np.testing.assert_array_equal(obs[:, 0], [4, 5, 6, 4, 5, 4, 5, 6])
                np.testing.assert_array_equal(obs[:, 1:], np.tile([8., 10.], (8, 1)))
                np.testing.assert_allclose(actions, env.executed_actions)
                expected_first_actions = [.5, .75, 1., .5, .75, .5, .75, 1.] if normalize else [2.] * 8
                np.testing.assert_allclose(actions[:, 0], expected_first_actions)
                np.testing.assert_array_equal(actions[:, 1], np.full(8, 6.))
                with np.load(output, allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['episode_lengths'], [3, 2, 3])
                    np.testing.assert_allclose(data['episode_returns'], [6, 4, 6])
                    np.testing.assert_array_equal(data['episode_costs'], [0., 0., 0.])
                    np.testing.assert_array_equal(data['costs'], np.zeros(8))
                    self.assertEqual(len(obs), data['episode_lengths'].sum())
                provenance = json.loads(output.with_suffix('.json').read_text())
                self.assertEqual(provenance['source_config'], cfg)
                self.assertTrue(provenance['deterministic'])
                self.assertEqual(provenance['attempted_trajectories'], 3)
                self.assertEqual(provenance['rejected_trajectories'], 0)
                self.assertEqual(provenance['max_attempts'], 30)
                if normalize:
                    for key, value in checkpoint['obs_normalizer'].items():
                        torch.testing.assert_close(created_normalizers[0].state_dict()[key], value,
                                                   rtol=0, atol=0)

    def test_run_directory_selects_the_highest_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            latest, _, _ = self.make_checkpoint(run_dir)
            (latest.parent / 'epoch-3.pt').write_bytes(latest.read_bytes())
            output = run_dir / 'selected.npz'
            with redirect_stdout(io.StringIO()), patch(
                'ICRL.expert.weights2trajectories.make', return_value=ExpertSamplingEnv(),
            ):
                sample_trajectories(run_dir, 1, output_path=output)
            provenance = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(provenance['checkpoint'], str(latest.resolve()))

    def test_rejects_whole_episodes_and_replaces_them_with_zero_cost_episodes(self):
        with tempfile.TemporaryDirectory() as temp:
            weights, _, _ = self.make_checkpoint(Path(temp))
            env = ExpertSamplingEnv({
                17: [0., 0., .25], 18: [0., 0.], 19: [0., float('nan')],
                20: [1., -1.], 21: [0., 1e-12], 22: [0.], 23: [0., 0., 0., 0.],
            })
            output = Path(temp) / 'safe.npz'
            with redirect_stdout(io.StringIO()), patch(
                'ICRL.expert.weights2trajectories.make', return_value=env,
            ):
                sample_trajectories(weights, 3, output_path=output)
            self.assertTrue(env.closed)
            self.assertEqual(env.seeds, list(range(17, 24)))
            self.assertEqual(len(env.executed_actions), 16)  # Includes discarded attempts.
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['episode_seeds'], [18, 22, 23])
                np.testing.assert_array_equal(data['episode_lengths'], [2, 1, 4])
                np.testing.assert_array_equal(data['episode_returns'], [4., 2., 8.])
                np.testing.assert_array_equal(data['episode_costs'], np.zeros(3))
                np.testing.assert_array_equal(data['costs'], np.zeros(7))
            obs, actions = load_expert_data(output)
            np.testing.assert_array_equal(obs[:, 0], [4, 5, 4, 4, 5, 6, 7])
            self.assertEqual(actions.shape, (7, 2))
            provenance = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(provenance['num_trajectories'], 3)
            self.assertEqual(provenance['attempted_trajectories'], 7)
            self.assertEqual(provenance['rejected_trajectories'], 4)
            self.assertEqual(provenance['mean_true_cost'], 0.)

    def test_attempt_limit_never_saves_partial_data_or_overwrites_an_existing_export(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                weights, _, _ = self.make_checkpoint(Path(temp))
                env = ExpertSamplingEnv({17: [0., 0.], 18: [0., .5]})
                output = Path(temp) / 'safe.npz'
                if existing:
                    output.write_bytes(b'previous dataset')
                    output.with_suffix('.json').write_text('previous provenance')
                with redirect_stdout(io.StringIO()), patch(
                    'ICRL.expert.weights2trajectories.make', return_value=env,
                ), self.assertRaisesRegex(RuntimeError, 'Only 1/2 zero-cost trajectories'):
                    sample_trajectories(weights, 2, output_path=output, max_attempts=2)
                self.assertTrue(env.closed)
                self.assertEqual(env.seeds, [17, 18])
                if existing:
                    self.assertEqual(output.read_bytes(), b'previous dataset')
                    self.assertEqual(output.with_suffix('.json').read_text(), 'previous provenance')
                else:
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix('.json').exists())

    def test_safeloop_actor_only_repeats_actions_but_saves_every_raw_step(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            weights = run_dir / 'torch_save/epoch-25.pt'
            weights.parent.mkdir()
            config = {
                'env_id': 'SafetyPointGoal1-v0-modelbased', 'seed': 17, 'algo': 'SafeLOOP',
                'train_cfgs': {'device': 'cpu', 'torch_threads': 1},
                'algo_cfgs': {'obs_normalize': False, 'action_repeat': 5},
                'model_cfgs': {'actor_type': 'gaussian_sac',
                               'weight_initialization_mode': 'kaiming_uniform',
                               'actor': {'hidden_sizes': [], 'activation': 'relu'}},
            }
            (run_dir / 'config.json').write_text(json.dumps(config))
            actor = ActorBuilder(ModelBasedSamplingEnv.observation_space,
                                 ModelBasedSamplingEnv.action_space, [],
                                 activation='relu').build_actor('gaussian_sac')
            with torch.no_grad():
                actor.net[0].weight.zero_()
                actor.net[0].bias.zero_()
                actor.net[0].weight[0, 0] = .2
            torch.save({'actor_critic': {f'actor.{key}': value
                                         for key, value in actor.state_dict().items()},
                        'dynamics': {}}, weights)
            env = ModelBasedSamplingEnv()
            output = run_dir / 'actor-only.npz'
            with redirect_stdout(io.StringIO()), patch(
                'ICRL.expert.weights2trajectories.make', return_value=env,
            ) as make:
                sample_trajectories(weights, 1, output_path=output, max_attempts=2)

            make.assert_called_once_with('SafetyPointGoal1-v0-modelbased', num_envs=1,
                                         device=torch.device('cpu'))
            self.assertTrue(env.closed)
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['episode_lengths'], [7])
                np.testing.assert_array_equal(data['episode_seeds'], [18])
                np.testing.assert_array_equal(data['costs'], np.zeros(7))
                np.testing.assert_array_equal(data['episode_returns'], [14.])
                np.testing.assert_array_equal(data['observations'][:, 0], [18.] * 7)
                np.testing.assert_array_equal(data['observations'][:, 1], np.arange(7))
                np.testing.assert_allclose(data['actions'][:5, 0], np.tanh(.2), atol=1e-7)
                np.testing.assert_allclose(data['actions'][5:, 0], np.tanh(1.2), atol=1e-7)
                np.testing.assert_array_equal(data['actions'][:, 1], np.zeros(7))
            provenance = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(provenance['source_config'], config)
            self.assertEqual(provenance['dataset_env_id'], 'SafetyPointGoal1-v0')
            self.assertEqual(provenance['policy_mode'], 'actor_only_no_planner')
            self.assertEqual(provenance['source_action_repeat'], 5)
            self.assertEqual(provenance['rejected_trajectories'], 1)


if __name__ == '__main__':
    unittest.main()
