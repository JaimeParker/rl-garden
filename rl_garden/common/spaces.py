"""Shared Gymnasium space normalization helpers."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def canonicalize_floating_observation_space(space: spaces.Space) -> spaces.Space:
    """Return rl-garden's runtime observation-space view.

    Floating observations are represented as float32 tensors in the training
    path. Keep non-floating spaces (for example uint8 images) unchanged.
    """
    if isinstance(space, spaces.Dict):
        return spaces.Dict(
            {
                key: canonicalize_floating_observation_space(subspace)
                for key, subspace in space.spaces.items()
            }
        )
    if isinstance(space, spaces.Box) and np.issubdtype(space.dtype, np.floating):
        return spaces.Box(
            low=space.low.astype(np.float32),
            high=space.high.astype(np.float32),
            shape=space.shape,
            dtype=np.float32,
        )
    return space
