"""Identity-ish extractor for flat Box observations (state-only SAC)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_normalization import RunningObsNormalizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.observations.schema import Modality, key_modality


class FlattenExtractor(BaseFeaturesExtractor):
    """Flattens a Box observation, or all ``state``/``state_<name>`` entries
    of a ``Dict`` observation containing only state keys (the schema-
    normalized shape of a state-only env/dataset -- see
    ``rl_garden.observations.normalize_observation_space``), concatenated in
    dict (schema) order.
    """

    def __init__(
        self,
        observation_space: "spaces.Box | spaces.Dict",
        normalize_obs: bool = False,
        *,
        state_keys: Optional[tuple[str, ...]] = None,
    ) -> None:
        """``state_keys`` fixes the concatenation order explicitly (the
        factory passes ``schema.state_keys`` -- "state" first, then the
        rest in space order, per ``ObservationSchema``); when omitted, it
        defaults to ``observation_space.spaces.keys()``'s own order, which
        for a bare ``gymnasium.spaces.Dict`` happens to already be
        alphabetical (and so already "state" first for this key family) but
        is not a guaranteed API contract to rely on."""
        if isinstance(observation_space, spaces.Dict):
            if state_keys is None:
                state_keys = tuple(observation_space.spaces.keys())
            for key in state_keys:
                if key_modality(key) != Modality.STATE:
                    raise ValueError(
                        "FlattenExtractor's Dict observation_space must contain "
                        f"only state keys, got {key!r} in {sorted(observation_space.spaces)}"
                    )
            self._state_keys: Optional[tuple[str, ...]] = state_keys
            features_dim = sum(
                int(np.prod(observation_space.spaces[k].shape)) for k in state_keys
            )
        else:
            self._state_keys = None
            features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.normalizer = RunningObsNormalizer(features_dim) if normalize_obs else None

    def _flatten_obs(self, obs) -> torch.Tensor:
        if self._state_keys is None:
            return self.flatten(obs)
        if len(self._state_keys) == 1:
            return self.flatten(obs[self._state_keys[0]])
        return torch.cat([self.flatten(obs[k]) for k in self._state_keys], dim=-1)

    def forward(self, obs) -> torch.Tensor:
        flat = self._flatten_obs(obs)
        return self.normalizer(flat) if self.normalizer is not None else flat

    def update_normalizer(self, obs) -> None:
        if self.normalizer is not None:
            self.normalizer.update(self._flatten_obs(obs))
