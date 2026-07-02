"""Utilities for loading Minari offline datasets into replay buffers."""

from __future__ import annotations

from typing import Any, Optional
import warnings

import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import (
    _add_flat_transitions,
    _concat,
    _load_success,
    _mc_returns,
    _slice,
    _to_tensor,
)
from rl_garden.buffers.base import BaseReplayBuffer


def _require_minari():
    try:
        import minari  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without minari.
        raise ImportError(
            "Loading Minari datasets requires the `minari` package. Install "
            "rl-garden with the `minari` extra or install minari directly."
        ) from exc
    return minari


def infer_specs_from_minari(dataset_id: str) -> tuple[spaces.Space, spaces.Space]:
    """Return the observation/action spaces stored on a Minari dataset."""
    minari = _require_minari()
    dataset = minari.load_dataset(dataset_id, download=True)
    return dataset.observation_space, dataset.action_space


def load_minari_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    dataset_id: str,
    *,
    num_episodes: Optional[int] = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Load a Minari dataset's episodes into an existing replay buffer.

    ``done`` is set to each episode's ``terminations`` only -- ``truncations``
    (timeouts) are intentionally excluded so the Bellman target keeps
    bootstrapping through artificial episode cutoffs, the offline-RL-standard
    convention D4RL's ``timeouts`` field was introduced to support. This
    differs from ``maniskill_h5.py``'s ``_transition_done``, which ORs every
    terminal-like field together.

    ``reward_scale``/``reward_bias`` apply ``r := scale * r + bias`` to each
    loaded reward, matching ``load_maniskill_h5_to_replay_buffer``.
    """
    minari = _require_minari()
    storage_device = buffer.storage_device
    dataset = minari.load_dataset(dataset_id, download=True)
    sparse_reward_mc = bool(getattr(buffer, "sparse_reward_mc", False))

    obs_parts: list[Any] = []
    next_obs_parts: list[Any] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    mc_parts: list[torch.Tensor] = []
    success_parts: list[torch.Tensor] = []

    episode_indices = None
    if num_episodes is not None:
        episode_indices = list(range(min(num_episodes, dataset.total_episodes)))
    warned_success_fallback = False

    for episode in dataset.iterate_episodes(episode_indices):
        actions = _to_tensor(episode.actions, storage_device).float()
        rewards = _to_tensor(episode.rewards, storage_device).float()
        if reward_scale != 1.0 or reward_bias != 0.0:
            rewards = rewards * reward_scale + reward_bias
        dones = _to_tensor(episode.terminations, storage_device).float()
        length = actions.shape[0]

        obs = _to_tensor(_slice(episode.observations, 0, length), storage_device)
        next_obs = _to_tensor(
            _slice(episode.observations, 1, length + 1), storage_device
        )

        if sparse_reward_mc:
            success, inferred = _load_success(
                {"infos": episode.infos or {}},
                length,
                storage_device,
                success_key=success_key,
                rewards=rewards,
                success_threshold=float(getattr(buffer, "success_threshold", 0.5)),
            )
            success_parts.append(success)
            if inferred and not warned_success_fallback:
                warnings.warn(
                    "sparse_reward_mc=True but no success field was found in the "
                    "Minari episode infos; inferring success from "
                    "reward >= success_threshold.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_success_fallback = True
        elif hasattr(buffer, "_mc_table") and hasattr(buffer, "gamma"):
            mc_parts.append(_mc_returns(rewards, dones, float(buffer.gamma)))

        obs_parts.append(obs)
        next_obs_parts.append(next_obs)
        action_parts.append(actions)
        reward_parts.append(rewards)
        done_parts.append(dones)

    if not action_parts:
        raise ValueError(f"No episodes found in Minari dataset: {dataset_id!r}")

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
    actions_all = torch.cat(action_parts, dim=0)
    rewards_all = torch.cat(reward_parts, dim=0)
    dones_all = torch.cat(done_parts, dim=0)
    mc_all = torch.cat(mc_parts, dim=0) if mc_parts else None
    success_all = torch.cat(success_parts, dim=0) if success_parts else None
    return _add_flat_transitions(
        buffer,
        obs_all,
        next_obs_all,
        actions_all,
        rewards_all,
        dones_all,
        mc_all,
        success_all,
    )
