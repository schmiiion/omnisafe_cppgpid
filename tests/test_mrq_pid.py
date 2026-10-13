"""Small end-to-end checks for MR.Q with PID-controlled constraints."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from gymnasium.spaces import Box

from omnisafe.algorithms.off_policy.mrq_pid import MRQPID
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.utils.config import get_default_kwargs_yaml


class _TinyAdapter:
    observation_space = Box(low=-10, high=10, shape=(3,))
    action_space = Box(low=-1, high=1, shape=(2,))
    env_spec_keys: list[str] = []

    def __init__(self) -> None:
        self.obs = torch.zeros((1, 3))
        self.steps = 0
        self.episode_cost = 0.0

    def rollout(self, rollout_step, agent, buffer, logger, use_rand_action) -> None:
        for _ in range(rollout_step):
            action = torch.rand((1, 2)) * 2 - 1 if use_rand_action else agent.step(self.obs)
            next_obs = self.obs + 0.1
            cost = torch.tensor([float(self.steps % 2)])
            terminal = torch.tensor([self.steps % 3 == 2])
            buffer.store(
                obs=self.obs, act=action, reward=torch.ones(1), cost=cost,
                next_obs=next_obs, terminated=terminal, truncated=torch.tensor([False]),
            )
            self.episode_cost += cost.item()
            self.obs = next_obs
            self.steps += 1
            if bool(terminal[0]):
                logger.store({
                    'Metrics/EpRet': 3.0,
                    'Metrics/EpCost': self.episode_cost,
                    'Metrics/EpLen': 3.0,
                })
                self.obs = torch.zeros_like(self.obs)
                self.episode_cost = 0.0

    def close(self) -> None:
        pass


class TestMRQPID(unittest.TestCase):
    def test_pid_variant_owns_training_implementation(self) -> None:
        self.assertIn('learn', MRQPID.__dict__)
        self.assertIn('_update', MRQPID.__dict__)
        self.assertIn('_update_encoder', MRQPID.__dict__)
        self.assertIn('_update_rl', MRQPID.__dict__)

    def test_n_step_return_stops_at_terminal(self) -> None:
        values = torch.tensor([[1.0, 2.0, 100.0], [1.0, 2.0, 3.0]])
        done = torch.tensor([[False, True, False], [False, False, False]])
        returns, discounts = MRQPID._n_step_return(values, done, 0.5)
        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([2.0, 2.75])))
        self.assertTrue(torch.allclose(discounts[:, 0], torch.tensor([0.0, 0.125])))

    def test_learn_updates_encoder_reward_cost_and_actor(self) -> None:
        cfg = get_default_kwargs_yaml('MRQPID', 'SafetyPointGoal1-v0', 'off-policy')
        cfg.recurisve_update({'exp_name': 'mrq-pid-test', 'algo': 'MRQPID', 'env_id': 'SafetyPointGoal1-v0'})
        cfg.train_cfgs.total_steps = 12
        cfg.train_cfgs.eval_episodes = 0
        cfg.algo_cfgs.steps_per_epoch = 6
        cfg.algo_cfgs.start_learning_steps = 4
        cfg.algo_cfgs.size = 64
        cfg.algo_cfgs.batch_size = 4
        cfg.algo_cfgs.enc_horizon = 2
        cfg.algo_cfgs.Q_horizon = 2
        cfg.algo_cfgs.target_update_freq = 2
        cfg.algo_cfgs.zs_dim = 16
        cfg.algo_cfgs.za_dim = 8
        cfg.algo_cfgs.zsa_dim = 16
        cfg.algo_cfgs.enc_hdim = 16
        cfg.algo_cfgs.value_hdim = 16
        cfg.algo_cfgs.policy_hdim = 16
        cfg.algo_cfgs.num_bins = 9
        cfg.algo_cfgs.cost_prediction_weight = 0.1
        cfg.lagrange_cfgs.cost_limit = 0.0
        cfg.lagrange_cfgs.pid_kp = 0.1
        cfg.lagrange_cfgs.pid_ki = 0.1
        cfg.lagrange_cfgs.pid_kd = 0.0
        cfg.logger_cfgs.use_tensorboard = False
        cfg.logger_cfgs.save_model_freq = 1
        with tempfile.TemporaryDirectory() as log_dir:
            cfg.logger_cfgs.log_dir = log_dir

            def init_env(algo: MRQPID) -> None:
                algo._env = _TinyAdapter()
                algo._epochs = 2
                algo._epoch = 0
                algo._update_cycle = 1
                algo._samples_per_epoch = 6

            with patch.object(MRQPID, '_init_env', init_env):
                algo = MRQPID('SafetyPointGoal1-v0', cfg)
                self.assertIsInstance(algo._lagrange, PIDLagrangian)
                encoder_updates = 0
                original_update_encoder = algo._update_encoder

                def count_encoder_update(batch) -> None:
                    nonlocal encoder_updates
                    encoder_updates += 1
                    original_update_encoder(batch)

                algo._update_encoder = count_encoder_update
                before = {
                    name: next(module.parameters()).detach().clone()
                    for name, module in (
                        ('encoder', algo._actor_critic.encoder),
                        ('reward', algo._actor_critic.reward_critic),
                        ('cost', algo._actor_critic.cost_critic),
                        ('policy', algo._actor_critic.actor),
                    )
                }
                result = algo.learn()
                checkpoints = sorted(Path(algo._logger._log_dir, 'torch_save').glob('epoch-*.pt'))
                saved = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
        self.assertEqual(algo._environment_steps, 12)
        self.assertEqual(algo._training_steps, 8)
        self.assertEqual(encoder_updates, 8)  # Four target snapshots, two encoder steps each.
        self.assertEqual(result[2], 3.0)
        self.assertGreater(algo._lagrange.lagrangian_multiplier, 0.0)
        self.assertEqual(saved['lagrangian_multiplier'], algo._lagrange.lagrangian_multiplier)
        self.assertTrue(bool(algo._buf.valid_starts.any()))
        for name, module in (
            ('encoder', algo._actor_critic.encoder),
            ('reward', algo._actor_critic.reward_critic),
            ('cost', algo._actor_critic.cost_critic),
            ('policy', algo._actor_critic.actor),
        ):
            self.assertFalse(torch.equal(before[name], next(module.parameters()).detach()), name)


if __name__ == '__main__':
    unittest.main()
