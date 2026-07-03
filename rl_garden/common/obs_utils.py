"""Shared helpers for flattening/indexing ``Obs`` values with leading (T, N) axes."""
from __future__ import annotations

import torch

from rl_garden.common.types import Obs


def flatten_leading_dims(obs: Obs) -> Obs:
    """Collapse the leading ``(T, N, ...)`` axes of ``obs`` into a single batch axis."""
    # Deferred import: rl_garden.buffers.dict_buffer's package (rl_garden.buffers)
    # imports rollout_buffer, which imports this module -- importing DictArray at
    # module level here would be circular.
    from rl_garden.buffers.dict_buffer import DictArray

    if isinstance(obs, DictArray):
        return {key: flatten_leading_dims(value) for key, value in obs.data.items()}
    if isinstance(obs, dict):
        return {key: flatten_leading_dims(value) for key, value in obs.items()}
    return obs.reshape((-1,) + obs.shape[2:])


def index_obs(obs: Obs, indices: torch.Tensor) -> Obs:
    """Index a (possibly nested-dict) ``obs`` value along its leading batch axis."""
    if isinstance(obs, dict):
        return {key: index_obs(value, indices) for key, value in obs.items()}
    return obs[indices]
