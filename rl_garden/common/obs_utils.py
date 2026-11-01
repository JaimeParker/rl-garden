"""Shared helpers for flattening/indexing/merging ``Obs`` values (nested-dict trees of tensors)."""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.common.types import Obs


def flatten_leading_dims(obs: Obs) -> Obs:
    """Collapse the leading ``(T, N, ...)`` axes of ``obs`` into a single batch axis."""
    # Deferred import: rl_garden.buffers.replay_buffer's package (rl_garden.buffers)
    # imports rollout_buffer, which imports this module -- importing DictArray at
    # module level here would be circular.
    from rl_garden.buffers.replay_buffer import DictArray

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


def tree_where(mask: torch.Tensor, new: Obs, old: Obs) -> Obs:
    if isinstance(new, dict):
        return {key: tree_where(mask, new[key], old[key]) for key in new}
    view_shape = (mask.shape[0],) + (1,) * (new.dim() - 1)
    return torch.where(mask.view(view_shape), new, old)


def tree_clone(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: tree_clone(value) for key, value in tree.items()}
    return tree.clone()


def tree_structure(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: tree_structure(value) for key, value in tree.items()}
    return None
