"""Explicit state-key routing; no image encoder or running normalization."""
from __future__ import annotations

from gymnasium import spaces
import numpy as np
import torch

from .base import BaseFeaturesExtractor


class SelectedStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, *, keys):
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("SelectedStateExtractor requires Dict observations")
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Select at least one distinct state key")
        dims = []
        for key in keys:
            if key != "state" and not key.startswith("state_"):
                raise ValueError(f"Not an explicit state key: {key}")
            space = observation_space.spaces.get(key)
            if (not isinstance(space, spaces.Box) or len(space.shape) != 1
                    or not np.issubdtype(space.dtype, np.floating)):
                raise ValueError(f"{key} must be a floating point vector")
            dims.append(space.shape[0])
        super().__init__(observation_space, sum(dims))
        self.keys = keys
        self.dims = tuple(dims)

    def forward(self, obs):
        parts = [obs[key] for key in self.keys]
        if any(x.ndim != 2 or x.shape[-1] != dim or x.shape[0] != parts[0].shape[0]
               for x, dim in zip(parts, self.dims)):
            raise ValueError("State batches do not match the declared schema")
        return torch.cat([x.float() for x in parts], dim=-1)
