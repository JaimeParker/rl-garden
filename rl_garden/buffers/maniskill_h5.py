"""Utilities for loading ManiSkill trajectory H5 files into replay buffers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import Obs


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without h5py.
        raise ImportError(
            "Loading ManiSkill H5 datasets requires h5py. Install rl-garden with "
            "the ManiSkill/offline dependencies or install h5py directly."
        ) from exc
    return h5py


def _read_node(node: Any) -> Any:
    h5py = _require_h5py()
    if isinstance(node, h5py.Dataset):
        return node[()]
    if isinstance(node, h5py.Group):
        return {key: _read_node(node[key]) for key in node.keys()}
    raise TypeError(f"Unsupported H5 node type: {type(node)!r}")


def _first_existing(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of the expected keys exist: {names}")


def _length(x: Any) -> int:
    if isinstance(x, dict):
        first = next(iter(x.values()))
        return _length(first)
    return int(x.shape[0])


def _slice(x: Any, start: int, end: int) -> Any:
    if isinstance(x, dict):
        return {key: _slice(value, start, end) for key, value in x.items()}
    return x[start:end]


def _concat(xs: list[Any]) -> Any:
    if isinstance(xs[0], dict):
        return {key: _concat([x[key] for x in xs]) for key in xs[0].keys()}
    return torch.cat(xs, dim=0)


def _mc_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for idx in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[idx] + gamma * running * (1.0 - dones[idx])
        returns[idx] = running
    return returns


def _to_tensor(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {key: _to_tensor(value, device) for key, value in x.items()}
    tensor = torch.as_tensor(x, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _transition_done(traj: dict[str, Any], length: int, device: torch.device) -> torch.Tensor:
    done_parts = []
    for key in ("dones", "done", "terminated", "terminations", "truncated", "truncations"):
        if key in traj:
            value = torch.as_tensor(traj[key][:length], device=device).bool()
            done_parts.append(value)
    if not done_parts:
        done = torch.zeros(length, device=device, dtype=torch.bool)
        done[-1] = True
        return done.float()
    done = done_parts[0]
    for part in done_parts[1:]:
        done = done | part
    return done.float()


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


def _add_flat_transitions(
    buffer: BaseReplayBuffer,
    obs: Obs,
    next_obs: Obs,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    mc_returns: torch.Tensor | None = None,
) -> int:
    total = actions.shape[0]
    usable = (total // buffer.num_envs) * buffer.num_envs
    if usable <= 0:
        raise ValueError(
            f"Offline dataset has {total} transitions, fewer than num_envs={buffer.num_envs}."
        )

    mc_table = None
    if mc_returns is not None and hasattr(buffer, "_mc_table"):
        mc_table = torch.zeros_like(buffer.rewards)

    for start in range(0, usable, buffer.num_envs):
        end = start + buffer.num_envs
        pos = buffer.pos
        buffer.add(
            _slice(obs, start, end),
            _slice(next_obs, start, end),
            actions[start:end],
            rewards[start:end],
            dones[start:end],
        )
        if mc_table is not None:
            mc_table[pos] = mc_returns[start:end].to(buffer.storage_device)

    if mc_table is not None:
        buffer._mc_table = mc_table
    return usable


def load_maniskill_h5_to_replay_buffer(
    buffer: BaseReplayBuffer,
    path: str | Path,
    *,
    num_traj: Optional[int] = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
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
            obs_parts.append(obs)
            next_obs_parts.append(next_obs)
            action_parts.append(actions)
            reward_parts.append(rewards)
            done_parts.append(dones)
            if hasattr(buffer, "_mc_table") and hasattr(buffer, "gamma"):
                mc_parts.append(_mc_returns(rewards, dones, float(buffer.gamma)))

    if not action_parts:
        raise ValueError(f"No trajectories found in ManiSkill H5 file: {path}")

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
    actions_all = torch.cat(action_parts, dim=0)
    rewards_all = torch.cat(reward_parts, dim=0)
    dones_all = torch.cat(done_parts, dim=0)
    mc_all = torch.cat(mc_parts, dim=0) if mc_parts else None
    return _add_flat_transitions(
        buffer, obs_all, next_obs_all, actions_all, rewards_all, dones_all, mc_all
    )
