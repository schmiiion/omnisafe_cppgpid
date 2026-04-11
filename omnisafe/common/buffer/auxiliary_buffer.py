import torch
from omnisafe.typing import OmnisafeSpace, DEVICE_CPU


class AuxiliaryBuffer:
    """Buffer to accumulate data across multiple policy phases for PPG's auxiliary phase."""

    def __init__(self, size: int, obs_space: OmnisafeSpace, act_space: OmnisafeSpace, device: torch.device = DEVICE_CPU):
        self._device = device
        self.max_size = size
        self.ptr = 0

        self.data = {
            'obs': torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device),
            'target_value_r': torch.zeros((size,), dtype=torch.float32, device=device),
            'target_value_c': torch.zeros((size,), dtype=torch.float32, device=device),
            'mean_old': torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device),
            'std_old': torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device),
        }

    def store(self, **batch_data: torch.Tensor) -> None:
        batch_size = batch_data['obs'].shape[0]
        assert self.ptr + batch_size <= self.max_size, 'No more space in the auxiliary buffer!'

        slice_idx = slice(self.ptr, self.ptr + batch_size)
        for key, value in batch_data.items():
            if key in self.data:
                self.data[key][slice_idx] = value

        self.ptr += batch_size

    def get(self):
        """Returns the auxiliary data and resets the pointer."""
        self.ptr = 0
        return self.data
