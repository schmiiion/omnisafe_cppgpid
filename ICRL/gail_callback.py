"""Explicit GAIL rollout hook for OmniSafe's on-policy buffers.

OmniSafe has no SB3 BaseCallback API. Call this hook after collecting a rollout
and before buffer.get()/the policy update. Keep raw observations/actions alongside
the policy's potentially normalized/scaled buffer inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger


@dataclass(frozen=True)
class _Path:
    start: int
    end: int
    last_value_r: torch.Tensor | None
    last_value_c: torch.Tensor | None


class GailRolloutBuffer(VectorOnPolicyBuffer):
    """Native buffer retaining the path boundaries needed for reward relabeling.

    Accepts exactly the VectorOnPolicyBuffer constructor arguments. Its store,
    finish_path and get methods can be used by OmniSafe's on-policy adapter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paths: list[list[_Path]] = [[] for _ in self.buffers]
        self._relabeled = False

    def finish_path(self, last_value_r=None, last_value_c=None, idx=0):
        buffer = self.buffers[idx]
        start, end = buffer.path_start_idx, buffer.ptr
        super().finish_path(last_value_r, last_value_c, idx)
        if end > start:
            self._paths[idx].append(_Path(
                start, end,
                None if last_value_r is None else last_value_r.detach().clone(),
                None if last_value_c is None else last_value_c.detach().clone(),
            ))

    def validate_rollout(self) -> tuple[int, int]:
        """Reject stale, unfinished or already relabeled data before any fitting."""
        if self._relabeled:
            raise RuntimeError('This rollout was already relabeled; consume it with get() first.')
        for buffer, paths in zip(self.buffers, self._paths):
            if buffer.ptr != buffer.max_size or buffer.path_start_idx != buffer.ptr:
                raise RuntimeError('Collect a full rollout and finish all paths before relabeling.')
            end = 0
            for path in paths:
                if path.start != end:
                    raise RuntimeError('Missing rollout path boundaries.')
                end = path.end
            if end != buffer.ptr:
                raise RuntimeError('Missing rollout path boundaries.')
        return self.buffers[0].ptr, self.num_buffers

    def relabel_rewards(self, rewards, *, add: bool = False) -> None:
        """Replace/add rewards and rerun native return/advantage computation.

        Retains the rollout's critic predictions and terminal/truncation bootstrap
        values. Cost signals and cost targets keep their native meaning.
        """
        shape = self.validate_rollout()
        reference = self.buffers[0].data['reward']
        rewards = torch.as_tensor(rewards, dtype=reference.dtype, device=reference.device)
        if tuple(rewards.shape) != shape or not torch.isfinite(rewards).all():
            raise ValueError(f'Expected finite rewards with shape {shape}.')
        for idx, (buffer, paths) in enumerate(zip(self.buffers, self._paths)):
            if add:
                buffer.data['reward'][:buffer.ptr].add_(rewards[:, idx])
            else:
                buffer.data['reward'][:buffer.ptr].copy_(rewards[:, idx])
            old_ptr, old_start = buffer.ptr, buffer.path_start_idx
            try:
                for path in paths:
                    buffer.path_start_idx, buffer.ptr = path.start, path.end
                    buffer.finish_path(path.last_value_r, path.last_value_c)
            finally:
                buffer.ptr, buffer.path_start_idx = old_ptr, old_start
        self._relabeled = True

    def get(self):
        data = super().get()
        self._paths = [[] for _ in self.buffers]
        self._relabeled = False
        return data


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class GailCallback:
    """OmniSafe replacement for the original SB3 rollout callback.

    ``learn_cost=False`` replaces rewards with log(D + eps); ``True`` adds
    log(D + eps) to the existing rewards, as in the source GACL callback. Neither
    mode installs D as an OmniSafe cost function. Use a learned-cost environment
    wrapper for that separate integration.
    """

    def __init__(self, discriminator, learn_cost: bool, true_cost_function=None,
                 save_dir: str | None = None, plot_disc: bool = False,
                 update_freq: int = 1, verbose: int = 1):
        if update_freq < 1:
            raise ValueError('update_freq must be at least one.')
        if plot_disc and save_dir is None:
            raise ValueError('save_dir is required when plot_disc is enabled.')
        self.discriminator = discriminator
        self.learn_cost = learn_cost
        self.true_cost_function = true_cost_function
        self.update_freq = update_freq
        self.verbose = verbose
        self.plot_save_dir = save_dir
        self.plot_disc = plot_disc
        self.disc_itr = 0
        if plot_disc:
            self.plot_folder = Path(save_dir) / 'discriminator'
            self.plot_folder.mkdir(parents=True, exist_ok=True)

    def on_rollout_end(self, *, buffer: GailRolloutBuffer, observations, actions,
                       logger: Logger, global_step: int,
                       obs_mean=None, obs_var=None) -> dict[str, float]:
        """Fit, relabel, recompute targets, then log through OmniSafe.

        Inputs have shape [steps, environments, features], in exactly the same
        coordinates as the expert demonstrations and true_cost_function. Supply
        pre-action observations and corresponding executed actions explicitly:
        clipped observation normalization cannot reliably be inverted afterward.
        """
        if not isinstance(buffer, GailRolloutBuffer):
            raise TypeError('Use GailRolloutBuffer to retain episode boundaries and bootstraps.')
        shape = buffer.validate_rollout()
        obs, acs = _numpy(observations), _numpy(actions)
        if obs.ndim != 3 or acs.ndim != 3 or obs.shape[:2] != shape or acs.shape[:2] != shape:
            raise ValueError(f'Expected observations/actions with leading dimensions {shape}.')
        # The legacy discriminator's continuous-action preprocessing expects 2D
        # actions. Flatten both arrays in the same time/environment order.
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_acs = acs.reshape(-1, acs.shape[-1])
        metrics = {}
        if self.disc_itr % self.update_freq == 0:
            metrics.update(self.discriminator.train_nn(
                1, flat_obs, flat_acs, obs_mean=obs_mean, obs_var=obs_var,
            ))
            if self.plot_disc:
                self.discriminator.plot_HC(
                    str(self.plot_folder / f'{self.disc_itr}.png'), nominal_obs=flat_obs,
                )
        if self.true_cost_function is not None:
            metrics['eval/mean_cost'] = float(np.mean(self.true_cost_function(flat_obs, flat_acs)))
        rewards = self.discriminator.reward_function(flat_obs, flat_acs).reshape(shape)
        buffer.relabel_rewards(rewards, add=self.learn_cost)
        logger.write_scalars(metrics, global_step=global_step)
        self.disc_itr += 1
        return metrics
