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
