# rl_garden/envs/wrappers/obs_space_utils.py
"""Small Dict-observation-space helpers shared by dual-encoder policies."""
from __future__ import annotations

from typing import Iterable

from gymnasium import spaces


def drop_dict_keys(space: spaces.Dict, keys: Iterable[str]) -> spaces.Dict:
    """Return a new ``spaces.Dict`` with ``keys`` removed.

    Keys not present in ``space`` are silently ignored. Never mutates
    ``space``.
    """
    drop = set(keys)
    return spaces.Dict({k: v for k, v in space.spaces.items() if k not in drop})
