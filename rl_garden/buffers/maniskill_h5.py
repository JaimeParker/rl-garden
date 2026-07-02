"""Utilities for loading ManiSkill trajectory H5 files into replay buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import (
    _add_flat_transitions,
    _concat,
    _first_existing,
    _length,
    _load_success,
    _mc_returns,
    _slice,
    _to_tensor,
    _transition_done,
)
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import Obs


def infer_specs_from_h5(
    path: str | Path,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> tuple[spaces.Box | spaces.Dict, spaces.Box]:
    """Infer Box or Dict observation/action spaces from a trajectory H5 file."""
    h5py = _require_h5py()

    def _box_from_dataset(dataset: Any) -> spaces.Box:
        shape = tuple(dataset.shape[1:])
        dtype = np.dtype(dataset.dtype)
        if dtype == np.dtype(np.uint8):
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return spaces.Box(low=info.min, high=info.max, shape=shape, dtype=dtype)
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def _space_from_node(node: Any) -> spaces.Box | spaces.Dict:
        if isinstance(node, h5py.Dataset):
            return _box_from_dataset(node)
        if isinstance(node, h5py.Group):
            return spaces.Dict(
                {key: _space_from_node(node[key]) for key in node.keys()}
            )
        raise TypeError(f"Unsupported H5 node type: {type(node)!r}")

    path = Path(path)
    with h5py.File(path, "r") as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("traj_")])
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {path}.")
        traj = f[traj_keys[0]]

        if "obs" in traj:
            obs_node = traj["obs"]
        elif "observations" in traj:
            obs_node = traj["observations"]
        else:
            raise ValueError(f"No obs/observations field in {traj_keys[0]}.")
        obs_space = _space_from_node(obs_node)

        if "actions" in traj:
            action_node = traj["actions"]
        elif "action" in traj:
            action_node = traj["action"]
        else:
            raise ValueError(f"No actions/action field in {traj_keys[0]}.")
        action_shape = tuple(action_node.shape[1:])

    action_space = spaces.Box(
        low=action_low,
        high=action_high,
        shape=action_shape,
        dtype=np.float32,
    )
    return obs_space, action_space


def infer_box_specs_from_h5(
    path: str | Path,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> tuple[spaces.Box, spaces.Box]:
    """Infer flat Box observation/action spaces from a trajectory H5 file."""
    obs_space, action_space = infer_specs_from_h5(
        path,
        action_low=action_low,
        action_high=action_high,
    )
    if isinstance(obs_space, spaces.Dict):
        raise NotImplementedError(
            "Dict observations detected. infer_box_specs_from_h5 supports flat "
            "Box observations only. Use infer_specs_from_h5 for RGBD/offline IQL."
        )
    return obs_space, action_space


def _load_traj_transitions(
    traj: dict[str, Any],
    device: torch.device,
) -> tuple[Obs, Obs, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = _to_tensor(_first_existing(traj, ("actions", "action")), device).float()
    rewards = _to_tensor(_first_existing(traj, ("rewards", "reward")), device).float()
    obs_raw = _first_existing(traj, ("obs", "observations"))
    length = min(actions.shape[0], rewards.shape[0])

    if "next_obs" in traj:
        next_obs_raw = traj["next_obs"]
        obs = _to_tensor(_slice(obs_raw, 0, length), device)
        next_obs = _to_tensor(_slice(next_obs_raw, 0, length), device)
    elif "next_observations" in traj:
        next_obs_raw = traj["next_observations"]
        obs = _to_tensor(_slice(obs_raw, 0, length), device)
        next_obs = _to_tensor(_slice(next_obs_raw, 0, length), device)
    elif _length(obs_raw) >= length + 1:
        obs = _to_tensor(_slice(obs_raw, 0, length), device)
        next_obs = _to_tensor(_slice(obs_raw, 1, length + 1), device)
    else:
        raise ValueError(
            "ManiSkill H5 trajectory must contain obs with length actions+1 "
            "or explicit next_obs/next_observations."
        )

    return (
        obs,
        next_obs,
        actions[:length],
        rewards[:length],
        _transition_done(traj, length, device),
    )


def load_maniskill_h5_to_replay_buffer(
    buffer: BaseReplayBuffer,
    path: str | Path,
    *,
    num_traj: Optional[int] = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Load ManiSkill trajectory H5 transitions into an existing replay buffer.

    The loader supports common ManiSkill layouts with top-level ``traj_*`` groups
    containing ``obs``/``observations``, ``actions``, ``rewards``, and either
    terminal flags or explicit ``next_obs``. Transitions are inserted in
    full ``buffer.num_envs`` chunks to preserve the existing ``(T, N, ...)``
    replay layout.

    ``reward_scale`` and ``reward_bias`` apply ``r := scale * r + bias`` to each
    loaded reward. Use the same values as ``RewardScaleBiasWrapper`` so that
    offline and online rewards live on the same scale.
    """
    h5py = _require_h5py()
    path = Path(path)
    storage_device = buffer.storage_device

    obs_parts: list[Obs] = []
    next_obs_parts: list[Obs] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    mc_parts: list[torch.Tensor] = []
    success_parts: list[torch.Tensor] = []
    sparse_reward_mc = bool(getattr(buffer, "sparse_reward_mc", False))
    warned_success_fallback = False

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        traj_keys = [key for key in keys if key.startswith("traj_")]
        if traj_keys:
            keys = sorted(traj_keys, key=lambda key: int(key.split("_")[-1]))
        if num_traj is not None:
            keys = keys[:num_traj]
        for key in keys:
            traj = _read_node(f[key])
            if not isinstance(traj, dict):
                continue
            obs, next_obs, actions, rewards, dones = _load_traj_transitions(
                traj, storage_device
            )
            if reward_scale != 1.0 or reward_bias != 0.0:
                rewards = rewards * reward_scale + reward_bias
            if sparse_reward_mc:
                success, inferred = _load_success(
                    traj,
                    rewards.shape[0],
                    storage_device,
                    success_key=success_key,
                    rewards=rewards,
                    success_threshold=float(getattr(buffer, "success_threshold", 0.5)),
                )
                success_parts.append(success)
                if inferred and not warned_success_fallback:
                    warnings.warn(
                        "sparse_reward_mc=True but no success field was found in the H5 "
                        "trajectory; inferring success from reward >= success_threshold.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_success_fallback = True
            obs_parts.append(obs)
            next_obs_parts.append(next_obs)
            action_parts.append(actions)
            reward_parts.append(rewards)
            done_parts.append(dones)
            if (
                (not sparse_reward_mc)
                and hasattr(buffer, "_mc_table")
                and hasattr(buffer, "gamma")
            ):
                mc_parts.append(_mc_returns(rewards, dones, float(buffer.gamma)))

    if not action_parts:
        raise ValueError(f"No trajectories found in ManiSkill H5 file: {path}")

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
