"""Native rollout boundaries, bootstrapping, logging and GAIL reward replacement."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from gymnasium.spaces import Box
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from ICRL.constraint_models.constraint_net import GailDiscriminator
from ICRL.gail_callback import GailCallback, GailRolloutBuffer


class GailCallbackTests(unittest.TestCase):
    def buffer(self, cls=GailRolloutBuffer, estimator='gae'):
        return cls(
            obs_space=Box(-1., 1., (3,)), act_space=Box(-1., 1., (2,)),
            size=4, gamma=0.9, lam=0.8, lam_c=0.7, advantage_estimator=estimator,
            penalty_coefficient=0.2, standardized_adv_r=False, standardized_adv_c=False,
            num_envs=2,
        )

    def collect(self, buffer, rewards=None):
        for step in range(4):
            buffer.store(
                obs=torch.full((2, 3), float(step)), act=torch.full((2, 2), 0.1 * step),
                reward=torch.tensor([step + 1., step + 2.]) if rewards is None else rewards[step],
                cost=torch.tensor([0.2, 0.3]), value_r=torch.tensor([0.5, 0.7]),
                value_c=torch.tensor([0.1, 0.4]), logp=torch.tensor([-0.1, -0.2]),
            )
            # Termination, then truncation and epoch cutoffs with nonzero bootstrap.
            if step == 1:
                buffer.finish_path(torch.zeros(1), torch.zeros(1), idx=0)
            if step == 2:
                buffer.finish_path(torch.tensor([1.2]), torch.tensor([0.3]), idx=1)
        buffer.finish_path(torch.tensor([1.1]), torch.tensor([0.2]), idx=0)
        buffer.finish_path(torch.tensor([1.3]), torch.tensor([0.5]), idx=1)

    def discriminator(self):
        return GailDiscriminator(
            obs_dim=3, acs_dim=2, hidden_sizes=(8,), batch_size=None,
            lr_schedule=lambda _: 0.001, expert_obs=np.zeros((8, 3), dtype=np.float32),
            expert_acs=np.zeros((8, 2), dtype=np.float32), is_discrete=False,
        )

    def arrays(self, buffer):
        return (torch.stack([b.data['obs'] for b in buffer.buffers], dim=1),
                torch.stack([b.data['act'] for b in buffer.buffers], dim=1))

    def test_both_reward_modes_match_fresh_native_targets_for_all_estimators(self):
        for estimator in ('gae', 'gae-rtg', 'vtrace', 'plain'):
            for add in (False, True):
                with self.subTest(estimator=estimator, add=add):
                    buffer = self.buffer(estimator=estimator)
                    self.collect(buffer)
                    obs, acs = self.arrays(buffer)
                    original_rewards = torch.stack([b.data['reward'].clone() for b in buffer.buffers], 1)
                    disc = self.discriminator()
                    callback = GailCallback(disc, learn_cost=add,
                                            true_cost_function=lambda s, a: np.full(len(s), 0.25))
                    logger = Mock(spec=Logger)
                    metrics = callback.on_rollout_end(buffer=buffer, observations=obs, actions=acs,
                                                       logger=logger, global_step=8)
                    self.assertEqual(metrics['eval/mean_cost'], 0.25)
                    logger.write_scalars.assert_called_once_with(metrics, global_step=8)
                    rewards = torch.from_numpy(disc.reward_function(
                        obs.numpy().reshape(-1, 3), acs.numpy().reshape(-1, 2),
                    ).reshape(4, 2))
                    if add:
                        rewards += original_rewards
                    expected = self.buffer(VectorOnPolicyBuffer, estimator)
                    self.collect(expected, rewards)
                    for native, actual in zip(expected.buffers, buffer.buffers):
                        self.assertEqual((actual.ptr, actual.path_start_idx), (4, 4))
                        for key in native.data:
                            torch.testing.assert_close(actual.data[key], native.data[key], rtol=0, atol=0)
                    expected_data, actual_data = expected.get(), buffer.get()
                    for key in expected_data:
                        torch.testing.assert_close(actual_data[key], expected_data[key], rtol=0, atol=0)
                    self.collect(buffer)
                    self.assertEqual(buffer.validate_rollout(), (4, 2))

    def test_update_frequency_and_invalid_rollouts(self):
        disc = self.discriminator()
        original_fit = disc.train_nn
        disc.train_nn = Mock(wraps=original_fit)
        callback = GailCallback(disc, learn_cost=False, update_freq=2)
        buffer, logger = self.buffer(), Mock(spec=Logger)
        obs = torch.zeros((4, 2, 3))
        acs = torch.zeros((4, 2, 2))
        with self.assertRaises(RuntimeError):
            callback.on_rollout_end(buffer=buffer, observations=obs, actions=acs,
                                    logger=logger, global_step=0)
        disc.train_nn.assert_not_called()
        for index in range(3):
            self.collect(buffer)
            callback.on_rollout_end(buffer=buffer, observations=obs, actions=acs,
                                    logger=logger, global_step=8 * (index + 1))
            with self.assertRaises(RuntimeError):
                callback.on_rollout_end(buffer=buffer, observations=obs, actions=acs,
                                        logger=logger, global_step=8 * (index + 1))
            buffer.get()
        self.assertEqual(disc.train_nn.call_count, 2)
        self.assertEqual(callback.disc_itr, 3)

    def test_metrics_reach_real_omnisafe_tensorboard_logger(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(directory, 'gail-port', use_tensorboard=True)
            try:
                buffer = self.buffer()
                self.collect(buffer)
                obs, acs = self.arrays(buffer)
                metrics = GailCallback(self.discriminator(), False).on_rollout_end(
                    buffer=buffer, observations=obs, actions=acs, logger=logger, global_step=8,
                )
                event = next(Path(directory).rglob('events.out.tfevents.*'))
                data = EventAccumulator(str(event.parent)).Reload()
                self.assertIn('discriminator/disc_loss', data.Tags()['scalars'])
                logged = data.Scalars('discriminator/disc_loss')[0]
                self.assertEqual(logged.step, 8)
                self.assertAlmostEqual(logged.value, metrics['discriminator/disc_loss'], places=6)
            finally:
                logger.close()


if __name__ == '__main__':
    unittest.main()
