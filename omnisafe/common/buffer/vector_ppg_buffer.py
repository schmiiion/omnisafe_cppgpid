import torch
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.common.buffer.auxiliary_buffer import AuxiliaryBuffer

from omnisafe.typing import OmnisafeSpace, DEVICE_CPU, AdvatageEstimator

class VectorPPGBuffer(VectorOnPolicyBuffer):
    """
    A wrapper buffer for Phasic Policy Gradient (PPG).
    It manages a standard VectorOnPolicyBuffer for the Policy Phase
    and siphons data into an AuxiliaryBuffer for the Auxiliary Phase.
    """
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
        device: torch.device = DEVICE_CPU,
    ) -> None:
        self._policy_buffer = policy_buffer

        total_aux_size = rollout_size * N_pi * num_envs
        self._aux_buffer = AuxiliaryBuffer(
            obs_space=obs_space,
            act_space=act_space,
            size=total_aux_size,
            device=device,
        )

        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c

    @property
    def num_buffers(self) -> int:
        """Number of buffers."""
        return self._policy_buffer.num_buffers

    @property
    def buffers(self):
        return self._policy_buffer.buffers

    def store(self, **data: torch.Tensor) -> None:
        """
        Store data during environment interaction in the policy buffer.
        """
        self._policy_buffer.store(**data)

    def store_aux_predictions(self, **data):
        self._aux_buffer.store(data)

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """
        Called when an episode ends or the epoch ends to bootstrap values.
        """
        self._policy_buffer.finish_path(last_value_r, last_value_c, idx)

    def get(self) -> dict[str, torch.Tensor]:
        """
        Called at the end of a policy phase. Computes advantages/targets,
        siphons the necessary flattened data into the auxiliary buffer,
        and returns the data to train the PPO policy.
        """
        data = self._policy_buffer.get()

        self._aux_buffer.store(**data)

        return data

    def get_aux_data(self) -> dict[str, torch.Tensor]:
        """
        Called when it is time to perform the Auxiliary Phase update.
        """
        return self._aux_buffer.get()
