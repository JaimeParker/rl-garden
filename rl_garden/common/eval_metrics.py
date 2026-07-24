"""Canonical naming for per-env episode stats collected during evaluation.

Vectorized env backends report completed-episode stats in
``infos["final_info"]["episode"]``, using different raw key conventions per
backend (gymnasium.wrappers.RecordEpisodeStatistics's classic ``"r"``/``"l"``/
``"t"``, or a backend's own canonical names like ``"return"``/
``"success_at_end"``). This module normalizes them to one set of names so
eval logging and downstream metric lookups (``_first_metric``,
``_log_eval_stdout``) behave the same way regardless of which env backend
produced them.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch

EVAL_METRIC_ALIASES: dict[str, str] = {
    "r": "return",
    "l": "episode_len",
    "success": "success_at_end",
    "is_success": "success_at_end",
}

# Raw keys with no RL-meaningful signal (wall-clock profiling, etc.).
EVAL_METRIC_DROP: frozenset[str] = frozenset({"t"})


def append_masked_episode_metrics(
    metrics: dict[str, list[torch.Tensor]],
    episode: Mapping[str, torch.Tensor],
    done_mask: Optional[torch.Tensor],
) -> None:
    """Append this step's completed-episode stats from ``episode`` into ``metrics``.

    ``episode`` is ``infos["final_info"]["episode"]`` from a vectorized env
    step. ``done_mask`` (typically ``infos["_final_info"]``) selects which
    sub-envs actually completed this step; entries for sub-envs that haven't
    finished are stale full-width values and must be filtered out.
    """
    for raw_key, value in episode.items():
        if raw_key.startswith("_") or raw_key in EVAL_METRIC_DROP:
            continue
        key = EVAL_METRIC_ALIASES.get(raw_key, raw_key)
        done_values = value[done_mask] if done_mask is not None else value
        if done_values.numel() == 0:
            continue
        metrics.setdefault(key, []).append(done_values)
