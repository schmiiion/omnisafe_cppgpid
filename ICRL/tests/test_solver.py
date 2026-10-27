"""Exercise real PPO updates, cost replacement and two complete outer iterations."""

import csv
from copy import deepcopy
import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from omnisafe.envs.core import env_register, make
from omnisafe.envs.custom_env import CustomEnv
from ICRL.ICRL_solver import main
from ICRL.constraint_models.constraint_net import ConstraintNet, VariationalConstraintNet
from ICRL.inner_loop import (
    LearnedCost, close_learner, collect_trajectories, make_learner, reset_rollout_buffers,
)
from ICRL.utils.config import (
    DEFAULT_ICRL_CONFIG, DEFAULT_OMNISAFE_CONFIG, configure_observation_mode,
    load_omnisafe_config,
)
from ICRL.utils.data import load_expert_data


@env_register
class ICRLTestEnv(CustomEnv):
    """Four-step episodes with raw costs deliberately outside learned cost bounds."""

    _support_envs = ['ICRLTest-v0']
    need_time_limit_wrapper = False

    def __init__(self, env_id, **kwargs):
        super().__init__(env_id, **kwargs)
        self._action_space = Box(2.0, 4.0, shape=(2,))

    @property
    def max_episode_steps(self):
        return 4

    def reset(self, seed=None, options=None):
        self._count = 0
        return torch.tensor([0., 10., 0.]), {}

    def step(self, action):
        self._count += 1
        obs = torch.tensor([float(self._count), 10. + self._count, -float(self._count)])
        reward = 1 - .01 * ((action - 3) ** 2).sum()
        return obs, reward, torch.tensor(17.), torch.tensor(self._count == 4), torch.tensor(False), {}

    def set_seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)


class SolverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.inner_path = self.directory / 'PPOLag.yaml'
        config = yaml.safe_load(DEFAULT_OMNISAFE_CONFIG.read_text())
        cfg = config['defaults']
        cfg['train_cfgs'].update(torch_threads=1, total_steps=16)
        cfg['algo_cfgs'].update(steps_per_epoch=8, update_iters=1, batch_size=4)
        cfg['model_cfgs']['actor']['hidden_sizes'] = [8]
        cfg['model_cfgs']['critic']['hidden_sizes'] = [8]
        cfg['logger_cfgs'].update(log_dir=str(self.directory / 'runs'), use_wandb=False)
        self.inner_path.write_text(yaml.safe_dump(config))
        self.cfg = load_omnisafe_config(self.inner_path, 'ICRLTest-v0')
        torch.set_num_threads(1)
        self.expert_obs = np.tile([0., 10., 0.], (8, 1)).astype(np.float32)
        self.expert_acs = np.full((8, 2), 3., dtype=np.float32)

    def constraint(self):
        return ConstraintNet(
            obs_dim=3, acs_dim=2, hidden_sizes=(8,), batch_size=None,
            lr_schedule=lambda _: .001, expert_obs=self.expert_obs,
            expert_acs=self.expert_acs, is_discrete=False, clip_obs=None,
            action_low=np.array([2., 2.]), action_high=np.array([4., 4.]),
            device='cpu', log_file=io.StringIO(),
        )

    def test_npz_and_benchmark_pickles_produce_same_flat_arrays(self):
        npz = self.directory / 'expert.npz'
        np.savez(npz, observations=self.expert_obs.reshape(2, 4, 3),
                 actions=self.expert_acs.reshape(2, 4, 2))
        pickles = self.directory / 'episodes'
        pickles.mkdir()
        for index in range(2):
            with (pickles / f'{index}.pkl').open('wb') as stream:
                pickle.dump({'original_observations': self.expert_obs[:4],
                             'actions': self.expert_acs[:4]}, stream)
        for path in (npz, pickles):
            obs, actions = load_expert_data(path)
            np.testing.assert_array_equal(obs, self.expert_obs)
            np.testing.assert_array_equal(actions, self.expert_acs)

    def test_main_rejects_same_shape_experts_from_a_different_task(self):
        dataset = self.directory / 'expert.npz'
        np.savez(dataset, observations=self.expert_obs, actions=self.expert_acs)
        dataset.with_suffix('.json').write_text(json.dumps({
            'source_config': {'env_id': 'DifferentTask-v0'},
        }))
        outer = yaml.safe_load(DEFAULT_ICRL_CONFIG.read_text())
        outer['icrl']['env_id'] = 'ICRLTest-v0'
        outer['icrl']['expert_dataset'] = str(dataset)
        path = self.directory / 'outer.yaml'
        path.write_text(yaml.safe_dump(outer))
        with self.assertRaisesRegex(ValueError, 'Expert trajectories are from DifferentTask-v0'):
            main(['--icrl-config', str(path), '--omnisafe-config', str(self.inner_path)])

    def test_point_xy_requires_matching_expert_provenance(self):
        dataset = self.directory / 'legacy.npz'
        np.savez(dataset, observations=np.zeros((4, 30), np.float32),
                 actions=np.zeros((4, 2), np.float32))
        outer = yaml.safe_load(DEFAULT_ICRL_CONFIG.read_text())
        outer['icrl'].update(env_id='SafetyPointCircle1-v0', observation_mode='point_xy',
                             expert_dataset=str(dataset))
        path = self.directory / 'outer_xy.yaml'
        path.write_text(yaml.safe_dump(outer))
        with self.assertRaisesRegex(ValueError, 'XY-observing policy'):
            main(['--icrl-config', str(path), '--omnisafe-config', str(self.inner_path)])

    def test_point_xy_reaches_actor_and_learned_cost(self):
        cfgs = load_omnisafe_config(self.inner_path, 'SafetyPointCircle1-v0')
        configure_observation_mode(cfgs, 'point_xy')
        env = make(cfgs.env_id, num_envs=1, device='cpu', **cfgs.env_cfgs)
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=11)
        expert_obs = np.repeat(observation.numpy()[None], 4, axis=0)
        expert_acs = np.zeros((4, 2), np.float32)
        constraint = ConstraintNet(
            obs_dim=30, acs_dim=2, hidden_sizes=(8,), batch_size=None,
            lr_schedule=lambda _: .001, expert_obs=expert_obs, expert_acs=expert_acs,
            is_discrete=False, clip_obs=None, action_low=env.action_space.low,
            action_high=env.action_space.high, device='cpu', log_file=io.StringIO(),
        )
        learner = make_learner(cfgs, constraint)
        self.addCleanup(close_learner, learner)
        policy_obs, _ = learner._env.reset()
        self.assertEqual(policy_obs.shape[-1], 30)
        action = learner._actor_critic.actor.predict(policy_obs)
        with redirect_stdout(io.StringIO()):
            next_obs, _, learned_cost, _, _, _ = learner._env.step(action)
        self.assertEqual(next_obs.shape[-1], 30)
        self.assertTrue(torch.isfinite(learned_cost).all())

    def test_cost_uses_pre_action_state_and_native_action(self):
        model = self.constraint()
        raw_env = ICRLTestEnv('ICRLTest-v0')
        env = LearnedCost(raw_env, model, torch.device('cpu'))
        obs, _ = env.reset()
        action = torch.tensor([3.5, 2.5])
        expected = model.cost_function(obs.numpy()[None], action.numpy()[None])[0]
        next_obs, _, cost, _, _, info = env.step(action)
        self.assertAlmostEqual(cost.item(), expected)
        self.assertEqual(info['true_cost'].item(), 17.)
        self.assertEqual(next_obs[0].item(), 1.)
        # The wrapper references the live model: no recreation after an outer update.
        with torch.no_grad():
            model.network[-2].bias.add_(5)
        self.assertLess(env.step(action)[2].item(), cost.item())

    def test_staged_training_sampling_and_normalization(self):
        learner = make_learner(self.cfg, self.constraint())
        self.addCleanup(close_learner, learner)
        with redirect_stdout(io.StringIO()):
            learner.learn(epochs=1, close=False)
        normalizer = learner._env.save()['obs_normalizer']
        statistics = {key: value.clone() for key, value in normalizer.state_dict().items()}
        actor = learner._actor_critic.actor
        identity = id(actor)
        optimizer_steps = [state['step'].item() for state in learner._actor_critic.actor_optimizer.state.values()]
        sampler = ICRLTestEnv('ICRLTest-v0')
        with patch.object(actor, 'predict', return_value=torch.tensor([.5, -.5])):
            obs, actions, lengths, metrics = collect_trajectories(learner, sampler, self.constraint(), 2)
        np.testing.assert_array_equal(lengths, [4, 4])
        np.testing.assert_array_equal(obs[0], [0., 10., 0.])
        np.testing.assert_allclose(actions, np.tile([3.5, 2.5], (8, 1)))
        self.assertEqual(metrics['ICRL/SampleTrueCost'], 68.)
        for key, value in statistics.items():
            torch.testing.assert_close(normalizer.state_dict()[key], value, rtol=0, atol=0)
        with redirect_stdout(io.StringIO()):
            learner.learn(epochs=1, close=False)
        self.assertEqual(id(learner._actor_critic.actor), identity)
        self.assertEqual(learner._learning_epoch, 2)
        self.assertEqual(learner.logger.current_epoch, 2)
        self.assertTrue(all(state['step'].item() > previous for state, previous in zip(
            learner._actor_critic.actor_optimizer.state.values(), optimizer_steps)))
        self.assertLess(learner.logger.get_stats('Metrics/EpCost')[0], 4.)

    def test_main_runs_two_real_outer_iterations_and_writes_logs(self):
        np.savez(self.directory / 'expert.npz', observations=self.expert_obs, actions=self.expert_acs)
        outer = yaml.safe_load(DEFAULT_ICRL_CONFIG.read_text())
        outer['icrl'].update(env_id='ICRLTest-v0', iterations=2, inner_steps=16, sample_rollouts=2,
                             expert_dataset=str(self.directory / 'expert.npz'),
                             runs_dir=str(self.directory / 'icrl_runs'))
        outer['constraint'].update(
            model='standard', hidden_sizes=[8], backward_iters=2, learning_rate=.004,
            cn_eps=.0003, anneal_clr_by_factor=.5,
        )
        outer['visualize_cost'] = {
            'enabled': True, 'trajectories_per_source': 1, 'step_stride': 2,
        }
        outer['visualize_rollouts'] = {
            'enabled': True, 'seeds': [1234], 'view': 'map',
            'frame_stride': 2, 'fps': 10, 'size': 64,
        }
        path = self.directory / 'outer.yaml'
        path.write_text(yaml.safe_dump(outer))
        visualization_calls = []
        rollout_video_calls = []
        fit_calls = []
        fitted_costs = []
        trace = MagicMock()

        original_train_nn = ConstraintNet.train_nn

        def tracked_fit(model, *args, **kwargs):
            observations, actions = kwargs['nominal_obs'], kwargs['nominal_acs']
            episodes = len(kwargs['episode_lengths'])
            cost_before = float(model.cost_function(observations, actions).sum() / episodes)
            weights_before = {key: value.detach().clone()
                              for key, value in model.network.state_dict().items()}
            result = original_train_nn(model, *args, **kwargs)
            fit_calls.append((kwargs['iterations'], model.eps,
                              model.optimizer.param_groups[0]['lr']))
            cost_after = float(model.cost_function(observations, actions).sum() / episodes)
            changed = any(not torch.equal(weights_before[key], value)
                          for key, value in model.network.state_dict().items())
            fitted_costs.append((cost_before, cost_after, changed))
            return result

        def visualize(**kwargs):
            visualization_calls.append(kwargs)
            directory = Path(kwargs['output_dir']) / 'visualize_cost'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f'iteration-{kwargs["iteration"]:03d}.png').write_bytes(b'plot')
            return {'expert_replay_steps': 0}

        def record_video(*args, **kwargs):
            rollout_video_calls.append((args, kwargs))
            return [{'steps': 3}]

        with redirect_stdout(io.StringIO()), patch(
            'ICRL.ICRL_solver.reset_rollout_buffers', wraps=reset_rollout_buffers,
        ) as reset_buffers, patch(
            'ICRL.ICRL_solver.record_live_iteration', side_effect=record_video,
        ), patch(
            'ICRL.ICRL_solver.WorldTrace', return_value=trace,
        ), patch(
            'ICRL.ICRL_solver.visualize_iteration', side_effect=visualize,
        ), patch.object(
            ConstraintNet, 'train_nn', tracked_fit,
        ), patch.object(
            ICRLTestEnv, 'step', autospec=True, side_effect=ICRLTestEnv.step,
        ) as env_step:
            main(['--icrl-config', str(path), '--omnisafe-config', str(self.inner_path)])
        self.assertEqual(reset_buffers.call_count, 2)
        self.assertEqual([call[0] for call in fit_calls], [2, 2])
        for (_, eps, lr), expected_lr in zip(fit_calls, [.004, .002]):
            self.assertAlmostEqual(eps, .0003)
            self.assertAlmostEqual(lr, expected_lr)
        # 32 training + 16 sampling steps; the video recorder is mocked here.
        self.assertEqual(env_step.call_count, 48)
        progress = next((self.directory / 'icrl_runs').rglob('icrl/icrl_progress.csv'))
        with progress.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual([float(row['ICRL/TrainingSteps']) for row in rows], [16, 32])
        self.assertEqual([float(row['ICRL/PolicyEpoch']) for row in rows], [2, 4])
        np.testing.assert_allclose(
            [float(row['ICRL/ConstraintLearningRate']) for row in rows], [.004, .002],
        )
        self.assertEqual(float(rows[-1]['ICRL/SamplingSteps']), 16)
        self.assertEqual(float(rows[-1]['ICRL/TotalInteractions']), 48)
        self.assertEqual([float(row['ICRL/VideoEvalSteps']) for row in rows], [3, 6])
        self.assertEqual([float(row['ICRL/VisualizationReplaySteps']) for row in rows], [0, 0])
        self.assertEqual([float(row['ICRL/AllEnvironmentSteps']) for row in rows], [27, 54])
        self.assertEqual(len(rollout_video_calls), 2)
        self.assertEqual([call[0][5] for call in rollout_video_calls], [1, 2])
        self.assertTrue(all(call[0][4] == [1234] for call in rollout_video_calls))
        self.assertEqual(float(rows[-1]['ICRL/SampleTrueCost']), 68)
        self.assertTrue(np.isfinite(float(rows[-1]['ICRL/backward/cn_loss'])))
        output = progress.parent
        self.assertEqual(output.name, 'icrl')
        native_output = output.parent
        self.assertEqual(native_output.parent.name, self.cfg.exp_name)
        self.assertEqual([call['iteration'] for call in visualization_calls], [1, 2])
        self.assertTrue(all(call['output_dir'].resolve() == output.resolve()
                            for call in visualization_calls))
        self.assertTrue(all(call['trajectories_per_source'] == 1 for call in visualization_calls))
        self.assertTrue(all(call['step_stride'] == 2 for call in visualization_calls))
        self.assertTrue(all(call['learner_trace'] is trace for call in visualization_calls))
        self.assertEqual(len(list((output / 'visualize_cost').glob('iteration-*.png'))), 2)
        for row, (cost_before, cost_after, _) in zip(rows, fitted_costs):
            self.assertAlmostEqual(float(row['ICRL/SampleLearnedCostBefore']), cost_before, places=6)
            self.assertAlmostEqual(float(row['ICRL/SampleLearnedCostAfter']), cost_after, places=6)
        self.assertTrue(any(changed for _, _, changed in fitted_costs))
        self.assertEqual(list(output.glob('constraint-*.pt')), [])
        metadata = json.loads((output / 'run_metadata.json').read_text())
        self.assertEqual(metadata['icrl_config'], str(path.resolve()))
        self.assertEqual(metadata['omnisafe_log_dir'], str(native_output.resolve()))
        native_config = json.loads((native_output / 'config.json').read_text())
        self.assertEqual(Path(native_config['logger_cfgs']['log_dir']).resolve(),
                         (self.directory / 'icrl_runs').resolve())
        self.assertEqual(metadata['visualize_cost']['step_stride'], 2)
        self.assertEqual(metadata['visualize_rollouts']['seeds'], [1234])
        self.assertTrue((native_output / 'torch_save/epoch-2.pt').exists())
        self.assertTrue((native_output / 'torch_save/epoch-4.pt').exists())
        events = next(native_output.rglob('events.out.tfevents.*'))
        accumulator = EventAccumulator(str(events.parent)).Reload()
        for tag in ('ICRL/SampleReturn', 'ICRL/SampleTrueCost',
                    'ICRL/SampleLearnedCostBefore', 'ICRL/SampleLearnedCostAfter',
                    'ICRL/backward/cn_loss', 'ICRL/VisualizationReplaySteps',
                    'ICRL/VideoEvalSteps',
                    'ICRL/AllEnvironmentSteps'):
            events = accumulator.Scalars(tag)
            self.assertEqual([event.step for event in events], [1, 2])
            np.testing.assert_allclose([event.value for event in events],
                                       [float(row[tag]) for row in rows], rtol=1e-6)
        self.assertEqual([event.step for event in accumulator.Scalars('Metrics/EpRet')], [1, 2, 3, 4])

    def test_variational_outer_iteration_fits_and_logs_a_consistent_mean_cost(self):
        dataset = self.directory / 'expert.npz'
        np.savez(dataset, observations=self.expert_obs, actions=self.expert_acs)
        outer = yaml.safe_load(DEFAULT_ICRL_CONFIG.read_text())
        outer['icrl'].update(env_id='ICRLTest-v0', iterations=1, inner_steps=8, sample_rollouts=1,
                             expert_dataset=str(dataset),
                             runs_dir=str(self.directory / 'icrl_runs'))
        outer['constraint'].update(
            model='variational', mode='mean', hidden_sizes=[8], backward_iters=2,
        )
        outer['visualize_rollouts']['enabled'] = False
        path = self.directory / 'variational_outer.yaml'
        path.write_text(yaml.safe_dump(outer))
        fitted_costs = {}
        original_train_nn = VariationalConstraintNet.train_nn

        def tracked_fit(model, *args, **kwargs):
            observations, actions = kwargs['nominal_obs'], kwargs['nominal_acs']
            episodes = len(kwargs['episode_lengths'])
            fitted_costs['mode'] = model.mode
            fitted_costs['before'] = float(
                model.cost_function(observations, actions, 'mean').sum() / episodes,
            )
            result = original_train_nn(model, *args, **kwargs)
            fitted_costs['after'] = float(
                model.cost_function(observations, actions, 'mean').sum() / episodes,
            )
            return result

        with redirect_stdout(io.StringIO()), patch(
            'ICRL.ICRL_solver.visualize_iteration',
        ) as visualize, patch.object(
            VariationalConstraintNet, 'train_nn', tracked_fit,
        ):
            main(['--icrl-config', str(path), '--omnisafe-config', str(self.inner_path)])
        visualize.assert_not_called()

        progress = next((self.directory / 'icrl_runs').rglob('icrl/icrl_progress.csv'))
        with progress.open() as stream:
            row, = list(csv.DictReader(stream))
        output = progress.parent
        self.assertEqual(fitted_costs['mode'], 'mean')
        for label in ('Before', 'After'):
            expected = fitted_costs[label.lower()]
            self.assertAlmostEqual(float(row[f'ICRL/SampleLearnedCost{label}']),
                                   float(expected), places=5)
        self.assertEqual(list(output.glob('constraint-*.pt')), [])
        self.assertTrue(np.isfinite(float(row['ICRL/backward/cn_loss'])))
        native_output = output.parent
        events = EventAccumulator(str(next(native_output.rglob('events.out.tfevents.*')).parent)).Reload()
        self.assertEqual(len(events.Scalars('ICRL/SampleLearnedCostAfter')), 1)

    def test_outer_boundary_preserves_learning_state_and_resets_rollouts(self):
        self.cfg.algo_cfgs.reward_normalize = True
        self.cfg.algo_cfgs.cost_normalize = True
        constraint = self.constraint()
        learner = make_learner(self.cfg, constraint)
        self.addCleanup(close_learner, learner)
        with redirect_stdout(io.StringIO()):
            learner.learn(epochs=1, close=False)

        # Leave the simulator partway through an episode, and deliberately dirty
        # the buffers to verify that the next block cannot reuse old samples.
        learner._env.step(torch.zeros(1, 2))
        for buffer in learner._buf.buffers:
            buffer.ptr = buffer.path_start_idx = 1
            for values in buffer.data.values():
                values.fill_(123)
        normalizers = learner._env.save()
        self.assertEqual(set(normalizers), {'obs_normalizer', 'reward_normalizer', 'cost_normalizer'})
        actor_critic = learner._actor_critic

        def learning_state():
            return deepcopy({
                'weights': actor_critic.state_dict(),
                'actor_optimizer': actor_critic.actor_optimizer.state_dict(),
                'reward_optimizer': actor_critic.reward_critic_optimizer.state_dict(),
                'cost_optimizer': actor_critic.cost_critic_optimizer.state_dict(),
                'scheduler': actor_critic.actor_scheduler.state_dict(),
                'multiplier': learner._lagrange.lagrangian_multiplier.detach(),
                'lagrange_optimizer': learner._lagrange.lambda_optimizer.state_dict(),
                'normalizers': {key: value.state_dict() for key, value in normalizers.items()},
            })

        def assert_same(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_same(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_same(a, b)
            else:
                self.assertEqual(left, right)

        before = learning_state()
        reset_rollout_buffers(learner)
        assert_same(before, learning_state())
        self.assertIs(learner._actor_critic, actor_critic)
        self.assertEqual(len(learner.logger._data['Metrics/EpCost']), 0)
        for buffer in learner._buf.buffers:
            self.assertEqual((buffer.ptr, buffer.path_start_idx), (0, 0))
            self.assertTrue(all(torch.count_nonzero(value) == 0 for value in buffer.data.values()))

        # An outer fit updates the constraint while all learner state stays idle.
        old_cost_weights = deepcopy(constraint.state_dict())
        constraint.train_nn(1, self.expert_obs + 1, self.expert_acs, np.array([4, 4]))
        self.assertTrue(any(not torch.equal(value, old_cost_weights[key])
                            for key, value in constraint.state_dict().items()))
        assert_same(before, learning_state())
        cost_optimizer_state = deepcopy(constraint.optimizer.state_dict())
        reset_rollout_buffers(learner)
        assert_same(cost_optimizer_state, constraint.optimizer.state_dict())
        assert_same(before, learning_state())

        cost_env = learner._env._env
        while not isinstance(cost_env, LearnedCost):
            cost_env = cost_env._env
        states = []
        original_step = cost_env.step

        def record_step(action):
            states.append(cost_env.observation.clone())
            return original_step(action)

        with redirect_stdout(io.StringIO()), patch.object(cost_env, 'step', side_effect=record_step):
            learner.learn(epochs=1, close=False)
        torch.testing.assert_close(states[0], torch.tensor([0., 10., 0.]))
        self.assertEqual(learner._learning_epoch, 2)
        self.assertEqual(actor_critic.actor_scheduler.last_epoch,
                         before['scheduler']['last_epoch'] + 1)
        for key, normalizer in normalizers.items():
            self.assertGreater(normalizer._count.item(), before['normalizers'][key]['_count'].item())
        for key, state in learner._lagrange.lambda_optimizer.state_dict()['state'].items():
            self.assertGreater(state['step'].item(), before['lagrange_optimizer']['state'][key]['step'].item())

    def test_staging_matches_uninterrupted_native_training(self):
        results = []
        for staged in (False, True):
            self.cfg.logger_cfgs.log_dir = str(self.directory / str(staged))
            torch.manual_seed(42)
            learner = make_learner(self.cfg, self.constraint())
            with redirect_stdout(io.StringIO()):
                if staged:
                    learner.learn(epochs=1, close=False)
                    learner.learn(epochs=1, close=False)
                else:
                    learner.learn()
            results.append({key: value.clone() for key, value in learner._actor_critic.state_dict().items()})
            close_learner(learner)
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
