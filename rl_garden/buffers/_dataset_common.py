"""Buffer-format-agnostic helpers shared by offline dataset loaders.

These operate on already-parsed tensors/dicts and know nothing about the
source format (ManiSkill H5, Minari, ...); see ``maniskill_h5.py`` and
``minari_dataset.py`` for the format-specific parsing that feeds into them.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import Obs


def _first_existing(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of the expected keys exist: {names}")


def _transition_done(
    traj: dict[str, Any], length: int, device: torch.device
) -> torch.Tensor:
    done_parts = []
    for key in (
        "dones",
        "done",
        "terminated",
        "terminations",
        "truncated",
        "truncations",
    ):
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


def _find_nested(data: dict[str, Any], key: str) -> Any | None:
    """Find ``key`` in a dataset dict; supports slash paths and one-level common nests."""
    if "/" in key:
        cur: Any = data
        for part in key.split("/"):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur
    if key in data:
        return data[key]
    for parent in ("infos", "info", "episode", "metrics"):
        child = data.get(parent)
        if isinstance(child, dict) and key in child:
            return child[key]
    return None


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


def _to_tensor(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {key: _to_tensor(value, device) for key, value in x.items()}
    tensor = torch.as_tensor(x, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _mc_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for idx in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[idx] + gamma * running * (1.0 - dones[idx])
        returns[idx] = running
    return returns


def _load_success(
    traj: dict[str, Any],
    length: int,
    device: torch.device,
    *,
    success_key: str | None,
    rewards: torch.Tensor,
    success_threshold: float,
) -> tuple[torch.Tensor, bool]:
    """Load per-transition success flags or infer them from rewards.

    Returns ``(success, inferred)``. Scalar or one-element episode-level success
    values are broadcast to the full trajectory.
    """
    keys: tuple[str, ...]
    if success_key is not None:
        keys = (success_key,)
    else:
        keys = ("success", "success_once", "success_at_end", "is_success")

    for key in keys:
        raw = _find_nested(traj, key)
        if raw is None:
            continue
        success = torch.as_tensor(raw, device=device).float()
        if success.numel() == 1:
            return success.reshape(1).expand(length), False
        success = success.reshape(-1)
        if success.numel() < length:
            raise ValueError(
                f"Success field {key!r} has {success.numel()} entries, "
                f"but trajectory has {length} transitions."
            )
        return success[:length], False

    return (rewards >= success_threshold).float(), True


def _add_flat_transitions(
    buffer: BaseReplayBuffer,
    obs: Obs,
    next_obs: Obs,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    mc_returns: torch.Tensor | None = None,
    successes: torch.Tensor | None = None,
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
        add_kwargs = {}
        if successes is not None:
            add_kwargs["success"] = successes[start:end]
        buffer.add(
            _slice(obs, start, end),
            _slice(next_obs, start, end),
            actions[start:end],
            rewards[start:end],
            dones[start:end],
            **add_kwargs,
        )
        if mc_table is not None:
            mc_table[pos] = mc_returns[start:end].to(buffer.storage_device)

    if mc_table is not None:
        buffer._mc_table = mc_table
    return usable
