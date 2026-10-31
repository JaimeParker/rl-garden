"""Canonical naming for per-env episode stats collected during evaluation.

Vectorized env backends report completed-episode stats in
``infos["final_info"]["episode"]``, using different raw key conventions per
backend (gymnasium.wrappers.RecordEpisodeStatistics's classic ``"r"``/``"l"``/
``"t"``, or a backend's own canonical names like ``"return"``/
``"success_at_end"``); see the episode boundary contract in
``.agents/rules/adding-env-backend.md``. This module normalizes them to one
set of names so eval logging and downstream metric lookups
(``_first_metric``, ``_log_eval_stdout``) behave the same way regardless of
which env backend produced them.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

EVAL_METRIC_ALIASES: dict[str, str] = {
    "r": "return",
    "l": "episode_len",
    "success": "success_at_end",
    "is_success": "success_at_end",
}

# Raw keys with no RL-meaningful signal (wall-clock profiling, etc.).
EVAL_METRIC_DROP: frozenset[str] = frozenset({"t"})


def canonical_episode_metric_key(raw_key: str) -> Optional[str]:
    """Return the canonical metric name for ``raw_key``, or None if it is skipped.

    Underscore-prefixed keys (RecordEpisodeStatistics' ``"_r"`` masks) and
    ``EVAL_METRIC_DROP`` keys yield None.
    """
    if raw_key.startswith("_") or raw_key in EVAL_METRIC_DROP:
        return None
    return EVAL_METRIC_ALIASES.get(raw_key, raw_key)


def masked_episode_metrics(infos: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return this step's completed-episode stats, keyed by canonical name.

    Reads ``infos["final_info"]["episode"]`` from a vectorized env step and
    keeps only the rows selected by ``infos["_final_info"]``; rows of sub-envs
    that did not finish this step hold stale full-width values. Each value is
    the tensor of done-row entries, left on the env device. Keys whose
    done-row selection is empty are omitted.

    Returns ``{}`` when the step carries no ``final_info`` or its
    ``final_info`` has no ``episode`` (e.g. ManiSkill with
    ``record_metrics=False``).

    Raises:
        ValueError: ``final_info`` is present without its ``_final_info`` mask.
    """
    final_info = infos.get("final_info")
    if final_info is None or "episode" not in final_info:
        return {}
    if "_final_info" not in infos:
        raise ValueError(
            "infos has 'final_info' but no '_final_info' mask; the env must emit "
            "'_final_info' whenever it emits 'final_info'."
        )
    done_mask = infos["_final_info"]
    metrics: dict[str, torch.Tensor] = {}
    for raw_key, value in final_info["episode"].items():
        key = canonical_episode_metric_key(raw_key)
        if key is None:
            continue
        done_values = value[done_mask]
        if done_values.numel() == 0:
            continue
        metrics[key] = done_values
    return metrics


def append_masked_episode_metrics(
    metrics: dict[str, list[torch.Tensor]],
    infos: Mapping[str, Any],
) -> None:
    """Append this step's ``masked_episode_metrics(infos)`` into ``metrics``."""
    for key, done_values in masked_episode_metrics(infos).items():
        metrics.setdefault(key, []).append(done_values)
