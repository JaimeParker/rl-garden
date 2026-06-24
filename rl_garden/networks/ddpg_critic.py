"""DrQ-v2 double-Q critic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class DrQv2Critic(nn.Module):
    """Reference DrQ-v2 critic with a shared feature trunk and two Q heads."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.trunk = nn.Sequential(
            nn.Linear(features_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.q1 = self._build_head(feature_dim + action_dim, hidden_dim)
        self.q2 = self._build_head(feature_dim + action_dim, hidden_dim)
        self._reset_parameters()

    @staticmethod
    def _build_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        hidden_action = torch.cat([hidden, actions], dim=-1)
        return self.q1(hidden_action), self.q2(hidden_action)

    def forward_all(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(self(features, actions), dim=0)
