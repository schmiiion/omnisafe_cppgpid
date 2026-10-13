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
"""Vector replay with MR.Q-style multi-step sampling and LAP priorities."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.vector_offpolicy_buffer import VectorOffPolicyBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class VectorMRQReplayBuffer(VectorOffPolicyBuffer):
    """Keep vectorized transitions and sample valid sequences by Q-error priority.

    ``horizon`` is the largest horizon requested by the encoder or Q update.
    Shorter samples use the same valid starts, as in the MR.Q replay buffer.
    A termination may appear inside a sampled sequence; ``loss_mask`` marks
    subsequent transitions as unusable. A truncation invalidates the sequence.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        horizon: int,
        alpha: float = 0.4,
        min_priority: float = 1.0,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        if not 1 <= horizon <= size:
            raise ValueError('horizon must be between 1 and buffer size')
        if alpha < 0 or min_priority <= 0:
            raise ValueError('alpha must be non-negative and min_priority must be positive')
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            size=size,
            batch_size=batch_size,
            num_envs=num_envs,
            penalty_coefficient=0.0,
            device=device,
        )
        self.horizon = horizon
        self.alpha = alpha
        self.min_priority = min_priority
        self.max_priority = min_priority**alpha
        self.env_terminates = False
        self.add_field('terminated', (), torch.bool)
        self.add_field('truncated', (), torch.bool)
        self.priority = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.valid_starts = torch.zeros((size, num_envs), dtype=torch.bool, device=device)

    def store(  # pylint: disable=too-many-arguments
        self,
        *,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one vector step, preserving the true final observation at a reset."""
        write_idx = self._ptr
        terminated = terminated.to(device=self._device, dtype=torch.bool).reshape(self.num_envs)
        truncated = truncated.to(device=self._device, dtype=torch.bool).reshape(self.num_envs)
        if bool(terminated.any()):
            self.env_terminates = True

        # Overwriting one slot invalidates every old window containing that slot.
        for offset in range(self.horizon):
            self.valid_starts[(write_idx - offset) % self.max_size] = False

        super().store(
            obs=obs,
            act=act,
            reward=reward,
            cost=cost,
            done=terminated & ~truncated,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )
        self.priority[write_idx] = self.max_priority

        if self.size >= self.horizon:
            start = (write_idx - self.horizon + 1) % self.max_size
            window = (start + torch.arange(self.horizon, device=self._device)) % self.max_size
            self.valid_starts[start] = ~self.data['truncated'][window].any(dim=0)

    def sample_sequences(
        self,
        horizon: int,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw ``(time, environment)`` starts proportionally to LAP priority.

        Every stored transition field has shape ``(batch, horizon, ...)``.
        ``indices`` has shape ``(batch, 2)`` and identifies the sampled starts
        for :meth:`update_priorities`. ``loss_mask`` is zero after a termination.
        """
        if not 1 <= horizon <= self.horizon:
            raise ValueError('sample horizon must be between 1 and configured horizon')
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')

        weights = (self.priority * self.valid_starts).reshape(-1)
        if not bool(weights.any()):
            raise ValueError('no valid replay sequences are available yet')
        flat_indices = torch.multinomial(weights, batch_size, replacement=True)
        start_idx = flat_indices // self.num_envs
        env_idx = flat_indices % self.num_envs
        time_idx = (start_idx[:, None] + torch.arange(horizon, device=self._device)) % self.max_size
        batch = {key: value[time_idx, env_idx[:, None]] for key, value in self.data.items()}
        not_done = (~batch['terminated']).to(dtype=torch.float32)
        batch['loss_mask'] = torch.cat(
            (torch.ones_like(not_done[:, :1]), torch.cumprod(not_done[:, :-1], dim=1)),
            dim=1,
        )
        batch['indices'] = torch.stack((start_idx, env_idx), dim=1)
        return batch

    def sample_batch(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Return one-step LAP samples for callers expecting the OmniSafe API."""
        batch = self.sample_sequences(horizon=1, batch_size=batch_size)
        return {key: value if key == 'indices' else value[:, 0] for key, value in batch.items()}

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """Set sampled-start priorities from one or more critic TD errors."""
        indices = indices.to(device=self._device, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError('indices must have shape (batch, 2)')
        errors = td_errors.detach().to(self._device).reshape(indices.shape[0], -1)
        priorities = errors.abs().amax(dim=1).clamp(min=self.min_priority).pow(self.alpha)
        self.priority[indices[:, 0], indices[:, 1]] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def reward_scale(self, eps: float = 1e-8) -> float:
        """Return the mean absolute reward in the populated replay slots."""
        if self.size == 0:
            return eps
        return float(self.data['reward'][: self.size].abs().mean().clamp(min=eps))
