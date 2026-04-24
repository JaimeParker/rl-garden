"""Identity-ish extractor for flat Box observations (state-only SAC)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor


class FlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box) -> None:
        features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.flatten(obs)
