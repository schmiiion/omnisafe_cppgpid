"""Tests for MR.Q replay sampling and OmniSafe rollout integration."""

import unittest
from types import SimpleNamespace

import torch
from gymnasium.spaces import Box

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorMRQReplayBuffer, VectorOffPolicyBuffer


def make_buffer(size: int = 6, horizon: int = 3) -> VectorMRQReplayBuffer:
    return VectorMRQReplayBuffer(
        obs_space=Box(low=-1000, high=1000, shape=(1,)),
        act_space=Box(low=-1, high=1, shape=(1,)),
        size=size,
        batch_size=8,
        num_envs=2,
        horizon=horizon,
    )


def store_step(
    buffer: VectorMRQReplayBuffer,
    t: int,
    terminated: tuple[bool, bool] = (False, False),
    truncated: tuple[bool, bool] = (False, False),
) -> None:
    obs = torch.tensor([[float(t)], [float(t + 100)]])
    buffer.store(
        obs=obs,
        act=torch.zeros((2, 1)),
        reward=torch.tensor([float(t), float(t + 10)]),
        cost=torch.tensor([float(t + 20), float(t + 30)]),
        next_obs=obs + 1,
        terminated=torch.tensor(terminated),
        truncated=torch.tensor(truncated),
    )


class TestVectorMRQReplayBuffer(unittest.TestCase):
    def test_sequences_preserve_environment_and_terminal_mask(self) -> None:
        buffer = make_buffer()
        store_step(buffer, 0)
        store_step(buffer, 1, terminated=(True, False))
        self.assertFalse(bool(buffer.valid_starts.any()))
        store_step(buffer, 2)

        batch = buffer.sample_sequences(3, batch_size=50)
        self.assertTrue(bool((batch['indices'][:, 0] == 0).all()))
        self.assertTrue(
            torch.equal(batch['obs'][:, :, 0].diff(dim=1), torch.ones((50, 2))),
        )
        self.assertTrue(buffer.env_terminates)
        self.assertAlmostEqual(buffer.reward_scale(), 6.0)
        env_zero = batch['indices'][:, 1] == 0
        self.assertTrue(bool(env_zero.any()))
        self.assertTrue(
            torch.equal(batch['loss_mask'][env_zero][0], torch.tensor([1.0, 1.0, 0.0])),
        )

    def test_truncation_and_overwrite_invalidate_sequence_starts(self) -> None:
        buffer = make_buffer(size=5, horizon=3)
        store_step(buffer, 0)
        store_step(buffer, 1, truncated=(True, False))
        store_step(buffer, 2)
        self.assertFalse(bool(buffer.valid_starts[0, 0]))
        self.assertTrue(bool(buffer.valid_starts[0, 1]))

        store_step(buffer, 3)
        store_step(buffer, 4)
        store_step(buffer, 5)  # Replaces slot zero.
        self.assertFalse(bool(buffer.valid_starts[0].any()))
        self.assertTrue(bool(buffer.valid_starts[3].all()))
        batch = buffer.sample_sequences(3, batch_size=50)
        self.assertFalse(bool((batch['indices'][:, 0] == 0).any()))
        self.assertFalse(bool(batch['truncated'].any()))

    def test_lap_priority_and_one_step_api(self) -> None:
        torch.manual_seed(0)
        buffer = make_buffer(size=4, horizon=2)
        store_step(buffer, 0)
        store_step(buffer, 1)
        buffer.update_priorities(
            torch.tensor([[0, 0], [0, 1]]),
            torch.tensor([[9.0, 2.0], [0.2, 0.1]]),
        )
        self.assertTrue(torch.isclose(buffer.priority[0, 0], torch.tensor(9.0**0.4)))
        self.assertEqual(float(buffer.priority[0, 1]), 1.0)

        batch = buffer.sample_batch(batch_size=2000)
        high_priority_count = int((batch['indices'][:, 1] == 0).sum())
        self.assertGreater(high_priority_count, 1200)
        self.assertLess(high_priority_count, 1600)
        self.assertEqual(batch['obs'].shape, (2000, 1))

    def test_rollout_passes_episode_flags_only_to_mrq_buffer(self) -> None:
        class FakeAdapter:
            _current_obs = torch.tensor([[0.0], [100.0]])
            _env = SimpleNamespace(num_envs=2)
            _device = torch.device('cpu')
            action_space = Box(low=-1, high=1, shape=(1,))

            def step(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
                assert action.shape == (2, 1)
                return (
                    torch.tensor([[10.0], [110.0]]),
                    torch.ones(2),
                    torch.zeros(2),
                    torch.tensor([True, False]),
                    torch.tensor([False, True]),
                    {'final_observation': torch.tensor([[1.0], [101.0]])},
                )

            def _log_value(self, **kwargs: torch.Tensor) -> None:
                del kwargs

            def _log_metrics(self, logger: object, idx: int) -> None:
                del logger, idx

            def _reset_log(self, idx: int) -> None:
                del idx

        class FakeAgent:
            def step(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
                del obs, deterministic
                return torch.zeros((2, 1))

        mrq_buffer = make_buffer(size=4, horizon=1)
        OffPolicyAdapter.rollout(FakeAdapter(), 1, FakeAgent(), mrq_buffer, object(), False)
        self.assertTrue(
            torch.equal(mrq_buffer.data['next_obs'][0, :, 0], torch.tensor([1.0, 101.0])),
        )
        self.assertTrue(
            torch.equal(mrq_buffer.data['terminated'][0], torch.tensor([True, False])),
        )
        self.assertTrue(
            torch.equal(mrq_buffer.data['truncated'][0], torch.tensor([False, True])),
        )

        ordinary_buffer = VectorOffPolicyBuffer(
            obs_space=Box(low=-1000, high=1000, shape=(1,)),
            act_space=Box(low=-1, high=1, shape=(1,)),
            size=4,
            batch_size=2,
            num_envs=2,
        )
        OffPolicyAdapter.rollout(FakeAdapter(), 1, FakeAgent(), ordinary_buffer, object(), False)
        self.assertTrue(
            torch.equal(ordinary_buffer.data['done'][0], torch.tensor([1.0, 0.0])),
        )

        random_buffer = make_buffer(size=4, horizon=1)
        OffPolicyAdapter.rollout(FakeAdapter(), 1, FakeAgent(), random_buffer, object(), True)
        self.assertEqual(random_buffer.data['act'][0].shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
