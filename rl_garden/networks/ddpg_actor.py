"""DDPG / DrQ-v2 deterministic actor.

Ported from ``3rd_party/drqv2/drqv2.py:70-93``.  Produces a ``TruncatedNormal``
whose mean is the network output and whose standard deviation is controlled
entirely by an external schedule — there are no learnable log-std parameters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.distributions import TruncatedNormal


class DDPGActor(nn.Module):
    """DrQ-v2 deterministic actor with external std.

    Architecture (matches ``3rd_party/drqv2/drqv2.py:70-93``)::

        trunk:  Linear(features_dim, feature_dim) → LayerNorm → Tanh
        policy: Linear → ReLU → Linear → ReLU → Linear(action_dim)
        output: tanh(mu) → TruncatedNormal(mu, ones_like(mu) * std)

    The standard deviation ``std`` is passed into ``forward()`` — it is **not**
    learned.  This is the fundamental difference from SAC's
    ``SquashedGaussianActor``, which outputs a learned per-dimension log-std.

    All actions are in ``[-1, 1]`` (DrQ-v2 convention).  No action scaling or
    bias is applied.
    """

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))

        self.trunk = nn.Sequential(
            nn.Linear(features_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, act_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, std: float) -> TruncatedNormal:
        h = self.trunk(features)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std_t = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std_t)

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        h = self.trunk(features)
        mu = self.policy(h)
        return torch.tanh(mu)
