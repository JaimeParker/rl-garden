"""Policy abstraction: owns the features_extractor and exposes predict."""
from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature

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
        """Call the extractor with hil-serl style gradient control when available."""
        supports_stop_gradient = getattr(
            self, "_features_extractor_supports_stop_gradient", None
        )
        if supports_stop_gradient is None:
            forward_params = signature(self.features_extractor.forward).parameters
            supports_stop_gradient = "stop_gradient" in forward_params
            self._features_extractor_supports_stop_gradient = supports_stop_gradient

        if supports_stop_gradient:
            return self.features_extractor(obs, stop_gradient=stop_gradient)

        features = self.features_extractor(obs)
        if stop_gradient:
            features = features.detach()
        return features
