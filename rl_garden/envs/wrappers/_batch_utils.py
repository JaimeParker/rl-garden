"""Small helpers shared by batched-vector-env wrappers that need to freeze
per-slot data at the sub-step a slot actually terminates on (``action_chunk.py``,
``skill_action_wrapper.py``).
"""
from __future__ import annotations

import torch

from rl_garden.common.types import Obs


def _tree_where(mask: torch.Tensor, new: Obs, old: Obs) -> Obs:
    if isinstance(new, dict):
        return {key: _tree_where(mask, new[key], old[key]) for key in new}
    view_shape = (mask.shape[0],) + (1,) * (new.dim() - 1)
    return torch.where(mask.view(view_shape), new, old)
