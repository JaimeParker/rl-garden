"""Identity-ish extractor for flat Box observations (state-only SAC)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_normalization import RunningObsNormalizer
from rl_garden.encoders.base import BaseFeaturesExtractor


class FlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, normalize_obs: bool = False) -> None:
        features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.normalizer = RunningObsNormalizer(features_dim) if normalize_obs else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(obs)
        return self.normalizer(flat) if self.normalizer is not None else flat

    def update_normalizer(self, obs: torch.Tensor) -> None:
        if self.normalizer is not None:
            self.normalizer.update(self.flatten(obs))
