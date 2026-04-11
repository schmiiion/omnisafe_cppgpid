import torch
import torch.nn as nn
from torch.distributions import Normal
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace

class GaussianLearningPPGActor(GaussianLearningActor):

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:

        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.shared_backbone = self.mean[:-2]
        last_hidden_dim = hidden_sizes[-1]

        #rebuilt mean
        self.mean = nn.Linear(last_hidden_dim, self._act_dim)

        #built aux heads
        self.aux_r_head = nn.Linear(last_hidden_dim, 1)
        self.aux_c_head = nn.Linear(last_hidden_dim, 1)

        nn.init.kaiming_uniform_(self.aux_r_head.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.aux_c_head.weight, nonlinearity='linear')


    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        features = self.shared_backbone(obs)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def auxiliary_forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass strictly for the Auxiliary Phase.
        Returns the predictions from the reward and cost value heads of the actor.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            reward value preciction for the observation,
            cost value preciction for the observation.
        """
        features = self.shared_backbone(obs)
        reward = self.aux_r_head(features).flatten()
        cost = self.aux_c_head(features).flatten()
        return reward, cost
