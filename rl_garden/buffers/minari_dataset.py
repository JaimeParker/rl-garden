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
from rl_garden.common.spaces import canonicalize_floating_observation_space


def _sparse_mc_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    success: torch.Tensor,
    gamma: float,
    sparse_negative_reward: float,
) -> torch.Tensor:
    if bool((success > 0.5).any()):
        return _mc_returns(rewards, dones, gamma)
    return torch.full_like(rewards, sparse_negative_reward / (1.0 - gamma))


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
    return (
        canonicalize_floating_observation_space(dataset.observation_space),
        dataset.action_space,
    )


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
    differs from ``h5.py``'s ``_transition_done``, which ORs every
    terminal-like field together.

    ``reward_scale``/``reward_bias`` apply ``r := scale * r + bias`` to each
    loaded reward, matching ``load_h5_dataset_to_replay_buffer``.
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
    episode_end_parts: list[torch.Tensor] = []
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
        # Unlike `dones` (terminations only, for TD bootstrap-through-timeout),
        # the MC recursion boundary is the true episode end: this episode's
        # last stored step, regardless of whether it closed via termination or
        # a `max_episode_steps` truncation.
        episode_end = torch.zeros_like(dones, dtype=torch.bool)
        episode_end[-1] = True
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
            mc_parts.append(
                _sparse_mc_returns(
                    rewards,
                    dones,
                    success,
                    float(buffer.gamma),
                    float(getattr(buffer, "sparse_negative_reward", 0.0)),
                )
            )
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
        episode_end_parts.append(episode_end)

    if not action_parts:
        raise ValueError(f"No episodes found in Minari dataset: {dataset_id!r}")

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
    actions_all = torch.cat(action_parts, dim=0)
    rewards_all = torch.cat(reward_parts, dim=0)
    dones_all = torch.cat(done_parts, dim=0)
    episode_end_all = torch.cat(episode_end_parts, dim=0)
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
        episode_ends=episode_end_all,
    )
