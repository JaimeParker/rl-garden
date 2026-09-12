"""Identity-ish extractor for flat Box observations (state-only SAC)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_normalization import RunningObsNormalizer
from rl_garden.encoders.base import BaseFeaturesExtractor


class FlattenExtractor(BaseFeaturesExtractor):
    """Flattens a Box observation, or the ``"state"`` entry of a
    ``Dict({"state": Box})`` observation (the schema-normalized shape of a
    state-only env/dataset -- see ``rl_garden.observations.normalize_observation_space``).
    """

    def __init__(
        self, observation_space: "spaces.Box | spaces.Dict", normalize_obs: bool = False
    ) -> None:
        if isinstance(observation_space, spaces.Dict):
            if set(observation_space.spaces) != {"state"}:
                raise ValueError(
                    "FlattenExtractor's Dict observation_space must contain "
                    f"exactly {{'state'}}, got {sorted(observation_space.spaces)}"
                )
            self._state_key: Optional[str] = "state"
            state_space = observation_space.spaces["state"]
        else:
            self._state_key = None
            state_space = observation_space
        features_dim = int(np.prod(state_space.shape))
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.normalizer = RunningObsNormalizer(features_dim) if normalize_obs else None

    def _unwrap(self, obs):
        return obs[self._state_key] if self._state_key is not None else obs

    def forward(self, obs) -> torch.Tensor:
        flat = self.flatten(self._unwrap(obs))
        return self.normalizer(flat) if self.normalizer is not None else flat

    def update_normalizer(self, obs) -> None:
        if self.normalizer is not None:
            self.normalizer.update(self.flatten(self._unwrap(obs)))
