"""Base class for feature extractors.

API-compatible with Stable-Baselines3 ``BaseFeaturesExtractor``: exposes a
``features_dim`` property that the policy heads rely on to size their MLPs.
"""
from __future__ import annotations

import gymnasium as gym
import torch.nn as nn


class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim
