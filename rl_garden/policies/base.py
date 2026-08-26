"""Policy abstraction: owns the features_extractor and exposes predict."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor


class BasePolicy(nn.Module, ABC):
    features_extractor: BaseFeaturesExtractor

    @abstractmethod
    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor: ...

    def _extract_features(
        self, obs: Obs, stop_gradient: bool = False
    ) -> torch.Tensor:
        return self.features_extractor.extract(obs, stop_gradient=stop_gradient)

    def update_obs_normalizer(self, obs: Obs) -> None:
        """Update any running obs-normalization statistics from ``obs``.

        Default no-op so rollout code (``OnPolicyAlgorithm``) can call this
        uniformly on any policy, whether or not it supports running
        normalization. Override in policies whose extractor(s) do (e.g.
        ``PPOPolicy``).
        """
