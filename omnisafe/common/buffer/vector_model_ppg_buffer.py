"""Phasic PPO buffer with contiguous, nonterminal model-training windows."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.auxiliary_buffer import AuxiliaryBuffer
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.common.buffer.vector_ppg_buffer import VectorPPGBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class ModelAuxiliaryBuffer(AuxiliaryBuffer):
    """Retain transition data and phase-stable critic predictions for auxiliary learning."""

    def __init__(
        self,
        size: int,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        super().__init__(size, obs_space, act_space, device)
        for key in ('target_value_r', 'target_value_c'):
            del self.data[key]
        self.data.update({
            'act': torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device),
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'model_valid': torch.zeros(size, dtype=torch.bool, device=device),
            'critic_value_r': torch.zeros(size, dtype=torch.float32, device=device),
            'critic_value_c': torch.zeros(size, dtype=torch.float32, device=device),
        })


class VectorModelPPGBuffer(VectorPPGBuffer):
    """Keep PPO data unchanged and add valid model windows to the auxiliary buffer."""

    def __init__(
        self,
        policy_buffer: VectorOnPolicyBuffer,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        rollout_size: int,
        N_pi: int,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int,
        model_horizon: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        super().__init__(
            policy_buffer=policy_buffer,
            obs_space=obs_space,
            act_space=act_space,
            rollout_size=rollout_size,
            N_pi=N_pi,
            standardized_adv_r=standardized_adv_r,
            standardized_adv_c=standardized_adv_c,
            num_envs=num_envs,
            device=device,
            auxiliary_buffer_cls=ModelAuxiliaryBuffer,
        )
        if not 1 <= model_horizon < rollout_size:
            raise ValueError('model_horizon must be between 1 and the per-env rollout length')
        self._model_horizon = model_horizon
        self._segment_ends: list[list[int]] = [[] for _ in range(num_envs)]

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """Record the boundary so model windows never cross resets or rollouts."""
        end = self._policy_buffer.buffers[idx].ptr - 1
        if end >= 0:
            self._segment_ends[idx].append(end)
        super().finish_path(last_value_r, last_value_c, idx)

    def get(self) -> dict[str, torch.Tensor]:
        """Retain raw rewards/costs before the PPO buffer resets for the next rollout."""
        rewards = []
        costs = []
        valid_starts = []
        for idx, buffer in enumerate(self._policy_buffer.buffers):
            size = buffer.ptr
            if size != buffer.max_size or self._segment_ends[idx][-1:] != [size - 1]:
                raise RuntimeError('Finish every rollout path before collecting model data')
            rewards.append(buffer.data['reward'][:size])
            costs.append(buffer.data['cost'][:size])
            valid = torch.zeros(size, dtype=torch.bool, device=buffer.data['obs'].device)
            valid[: size - self._model_horizon] = True
            for end in self._segment_ends[idx]:
                valid[max(0, end - self._model_horizon + 1) : end + 1] = False
            valid_starts.append(valid)

        data = self._policy_buffer.get()  # Standard GAE and separate value targets.
        self._aux_buffer.store(
            **data,
            reward=torch.cat(rewards),
            cost=torch.cat(costs),
            model_valid=torch.cat(valid_starts),
        )
        self._segment_ends = [[] for _ in self._segment_ends]
        return data
