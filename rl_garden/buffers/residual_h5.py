"""Utilities for loading rl-garden's residual-offline H5 datasets into replay buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.buffers._dataset_common import (
    _concat,
    _first_existing,
    _length,
    _slice,
    _to_tensor,
    _transition_done,
)
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import Obs


def _residual_done(
    traj: dict[str, Any],
    length: int,
    device: torch.device,
    *,
    bootstrap_at_done: str,
) -> torch.Tensor:
    if bootstrap_at_done not in ("always", "never", "truncated"):
        raise ValueError(f"Unknown bootstrap_at_done mode: {bootstrap_at_done!r}")

    def _flag(*names: str) -> torch.Tensor:
        out = torch.zeros(length, device=device, dtype=torch.bool)
        for name in names:
            if name in traj:
                out |= torch.as_tensor(traj[name][:length], device=device).bool()
        return out

    terminated = _flag("terminated", "terminations")
    truncated = _flag("truncated", "truncations")
    if not terminated.any() and not truncated.any():
        done = _transition_done(traj, length, device).bool()
        return done.float()
    if bootstrap_at_done == "always":
        return torch.zeros(length, device=device, dtype=torch.float32)
    if bootstrap_at_done == "never":
        return (terminated | truncated).float()
    return terminated.float()


def _load_residual_traj_transitions(
    traj: dict[str, Any],
    device: torch.device,
    *,
    bootstrap_at_done: str,
) -> tuple[
    Obs,
    Obs,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    obs_raw = _first_existing(traj, ("obs", "observations"))
    actions = _to_tensor(_first_existing(traj, ("actions", "action")), device).float()
    base_actions = _to_tensor(_first_existing(traj, ("base_actions",)), device).float()
    next_base_actions = _to_tensor(
        _first_existing(traj, ("next_base_actions",)), device
    ).float()
    rewards = _to_tensor(_first_existing(traj, ("rewards", "reward")), device).float()
    length = min(
        actions.shape[0],
        base_actions.shape[0],
        next_base_actions.shape[0],
        rewards.shape[0],
    )
    if _length(obs_raw) < length + 1:
        raise ValueError(
            "Residual H5 trajectory must contain obs with length actions+1."
        )
    obs = _to_tensor(_slice(obs_raw, 0, length), device)
    next_obs = _to_tensor(_slice(obs_raw, 1, length + 1), device)
    return (
        obs,
        next_obs,
        actions[:length],
        rewards[:length],
        _residual_done(traj, length, device, bootstrap_at_done=bootstrap_at_done),
        base_actions[:length],
        next_base_actions[:length],
    )


def _add_residual_transitions(
    buffer: BaseReplayBuffer,
    obs: Obs,
    next_obs: Obs,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    base_actions: torch.Tensor,
    next_base_actions: torch.Tensor,
) -> int:
    total = actions.shape[0]
    usable = (total // buffer.num_envs) * buffer.num_envs
    if usable <= 0:
        raise ValueError(
            f"Offline dataset has {total} transitions, fewer than num_envs={buffer.num_envs}."
        )
    for start in range(0, usable, buffer.num_envs):
        end = start + buffer.num_envs
        buffer.add(
            _slice(obs, start, end),
            _slice(next_obs, start, end),
            actions[start:end],
            rewards[start:end],
            dones[start:end],
            base_actions=base_actions[start:end],
            next_base_actions=next_base_actions[start:end],
        )
    return usable


def count_residual_h5_transitions(
    path: str | Path,
    *,
    num_traj: Optional[int] = None,
    num_envs: Optional[int] = None,
) -> int:
    """Count transitions loadable from a residual-offline H5 file.

    When ``num_envs`` is provided, the returned count matches replay-buffer
    insertion semantics. Residual offline buffers pass ``num_envs=1`` so every
    transition is loadable.
    """
    h5py = _require_h5py()
    path = Path(path)
    total = 0
    with h5py.File(path, "r") as f:
        dataset_type = f.attrs.get("dataset_type")
        if dataset_type not in (None, "rl_garden_residual_offline"):
            raise ValueError(
                f"Expected residual offline H5 dataset, got dataset_type={dataset_type!r}."
            )
        keys = sorted(
            [key for key in f.keys() if key.startswith("traj_")],
            key=lambda key: int(key.split("_")[-1]),
        )
        if num_traj is not None:
            keys = keys[:num_traj]
        for key in keys:
            traj = f[key]
            if not isinstance(traj, h5py.Group):
                continue
            if not all(
                field in traj
                for field in ("actions", "base_actions", "next_base_actions", "rewards")
            ):
                continue
            total += min(
                int(traj["actions"].shape[0]),
                int(traj["base_actions"].shape[0]),
                int(traj["next_base_actions"].shape[0]),
                int(traj["rewards"].shape[0]),
            )
    if num_envs is None:
        return total
    return (total // num_envs) * num_envs


def load_residual_h5_to_replay_buffer(
    buffer: BaseReplayBuffer,
    path: str | Path,
    *,
    num_traj: Optional[int] = None,
    bootstrap_at_done: str = "always",
) -> int:
    """Load residual-offline H5 transitions into a residual replay buffer."""
    h5py = _require_h5py()
    path = Path(path)
    storage_device = buffer.storage_device

    obs_parts: list[Obs] = []
    next_obs_parts: list[Obs] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    base_action_parts: list[torch.Tensor] = []
    next_base_action_parts: list[torch.Tensor] = []

    with h5py.File(path, "r") as f:
        dataset_type = f.attrs.get("dataset_type")
        if dataset_type not in (None, "rl_garden_residual_offline"):
            raise ValueError(
                f"Expected residual offline H5 dataset, got dataset_type={dataset_type!r}."
            )
        keys = sorted(
            [key for key in f.keys() if key.startswith("traj_")],
            key=lambda key: int(key.split("_")[-1]),
        )
        if num_traj is not None:
            keys = keys[:num_traj]
        for key in keys:
            traj = _read_node(f[key])
            if not isinstance(traj, dict):
                continue
            (
                obs,
                next_obs,
                actions,
                rewards,
                dones,
                base_actions,
                next_base_actions,
            ) = _load_residual_traj_transitions(
                traj, storage_device, bootstrap_at_done=bootstrap_at_done
            )
            obs_parts.append(obs)
            next_obs_parts.append(next_obs)
            action_parts.append(actions)
            reward_parts.append(rewards)
            done_parts.append(dones)
            base_action_parts.append(base_actions)
            next_base_action_parts.append(next_base_actions)

    if not action_parts:
        raise ValueError(f"No residual trajectories found in H5 file: {path}")

    return _add_residual_transitions(
        buffer,
        _concat(obs_parts),
        _concat(next_obs_parts),
        torch.cat(action_parts, dim=0),
        torch.cat(reward_parts, dim=0),
        torch.cat(done_parts, dim=0),
        torch.cat(base_action_parts, dim=0),
        torch.cat(next_base_action_parts, dim=0),
    )
