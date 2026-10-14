"""Gaussian PPG actor with an action-conditioned latent dynamics model."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class GaussianLearningModelPPGActor(GaussianLearningActor):
    """Share one encoder across policy, dynamics, and auxiliary value heads."""

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
        )
        # The original Gaussian actor constructs the policy MLP in ``mean``.
        # Reuse its hidden layers as the shared encoder.
        self.shared_backbone = self.mean[:-2]
        latent_dim = hidden_sizes[-1]
        self.mean = nn.Linear(latent_dim, self._act_dim)
        self.model_backbone = build_mlp_network(
            sizes=[latent_dim + self._act_dim, latent_dim, latent_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.next_latent_head = nn.Linear(latent_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.cost_head = nn.Linear(latent_dim, 1)
        self.value_reward_head = nn.Linear(latent_dim, 1)
        self.value_cost_head = nn.Linear(latent_dim, 1)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        latent = self.shared_backbone(obs)
        return Normal(self.mean(latent), torch.exp(self.log_std))

    def predict_values(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict reward and cost values from the policy's shared latent state."""
        latent = self.shared_backbone(obs)
        return (
            self.value_reward_head(latent).squeeze(-1),
            self.value_cost_head(latent).squeeze(-1),
        )

    def predict_model(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the next latent, immediate reward and immediate cost."""
        features = self.model_backbone(torch.cat((latent, action), dim=-1))
        return (
            latent + self.next_latent_head(features),
            self.reward_head(features).squeeze(-1),
            self.cost_head(features).squeeze(-1),
        )
