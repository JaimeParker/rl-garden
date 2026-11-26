"""Buffer-format-agnostic helpers shared by offline dataset loaders.

These operate on already-parsed tensors/dicts and know nothing about the
source format (trajectory H5, Minari, ...); see ``h5_dataset.py`` and
``minari_dataset.py`` for the format-specific parsing that feeds into them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs
from rl_garden.observations.schema import validate_observation_space


def first_existing(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of the expected keys exist: {names}")


def transition_done(
    traj: dict[str, Any], length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(done, truncated)`` for a loaded trajectory's transitions.

    ``done`` is the TD/Bellman stop-bootstrap flag: the OR of every
    *terminal* field present (``dones``/``done``/``terminated``/
    ``terminations``). ``truncated`` is a separate OR of every *truncation*
    field present (``truncated``/``truncations``), kept out of ``done``
    because an artificial ``max_episode_steps`` cutoff must not stop TD
    bootstrapping -- online rollouts never fold truncation into the
    bootstrap-stop flag either (``off_policy.py``'s ``_compute_done_masks``,
    default ``bootstrap_at_done="truncated"``, keeps bootstrapping through
    it). ``truncated`` is ``None`` when the trajectory has no truncation
    field at all.

    If the trajectory has no terminal field, ``done`` falls back to marking
    only the final transition (the pre-existing single-signal convention) --
    but only when there is no truncation field either. A truncation-only
    trajectory already carries a real boundary signal (just not one that
    should stop bootstrapping), so in that case ``done`` is all-``False``;
    see ``_load_traj_transitions``'s ``episode_end`` fallback for where that
    boundary is actually recorded.
    """
    done_parts = [
        torch.as_tensor(traj[key][:length], device=device).bool()
        for key in ("dones", "done", "terminated", "terminations")
        if key in traj
    ]
    truncated_parts = [
        torch.as_tensor(traj[key][:length], device=device).bool()
        for key in ("truncated", "truncations")
        if key in traj
    ]
    truncated: torch.Tensor | None = None
    for part in truncated_parts:
        truncated = part if truncated is None else (truncated | part)

    if done_parts:
        done = done_parts[0]
        for part in done_parts[1:]:
            done = done | part
    elif truncated is not None:
        done = torch.zeros(length, device=device, dtype=torch.bool)
    else:
        done = torch.zeros(length, device=device, dtype=torch.bool)
        done[-1] = True
    return done.float(), truncated


def find_nested(data: dict[str, Any], key: str) -> Any | None:
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


def tree_length(x: Any) -> int:
    if isinstance(x, dict):
        first = next(iter(x.values()))
        return tree_length(first)
    return int(x.shape[0])


def tree_slice(x: Any, start: int, end: int) -> Any:
    if isinstance(x, dict):
        return {key: tree_slice(value, start, end) for key, value in x.items()}
    return x[start:end]


def tree_concat(xs: list[Any]) -> Any:
    if isinstance(xs[0], dict):
        return {key: tree_concat([x[key] for x in xs]) for key in xs[0].keys()}
    return torch.cat(xs, dim=0)


def to_tensor(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {key: to_tensor(value, device) for key, value in x.items()}
    tensor = torch.as_tensor(x, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def mc_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for idx in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[idx] + gamma * running * (1.0 - dones[idx])
        returns[idx] = running
    return returns


def load_success(
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
        raw = find_nested(traj, key)
        if raw is None:
            continue
        success = torch.as_tensor(raw, device=device).float()
        if success.numel() == 1:
            return success.reshape(1).expand(length), False
        success = success.reshape(-1)
        if success.numel() == length + 1:
            return success[1 : length + 1], False
        if success.numel() < length:
            raise ValueError(
                f"Success field {key!r} has {success.numel()} entries, "
                f"but trajectory has {length} transitions."
            )
        return success[:length], False

    return (rewards >= success_threshold).float(), True


def finalize_dataset_obs_space(space: "spaces.Box | spaces.Dict") -> "spaces.Dict":
    """Coerce a dataset loader's inferred observation space into the strict
    rl-garden observation contract (``rl_garden.observations.schema``) and
    validate it.

    A bare (state-only) ``Box`` becomes ``Dict({"state": Box(float32)})`` --
    every state-only dataset loader's own ``infer_specs_from_*`` ends with
    this call. An already-``Dict`` space (built by a loader that maps its own
    producer-specific field names onto ``state``/``rgb_<cam>``/``depth_<cam>``
    itself, e.g. RLBench/robomimic) is validated as-is: this function does no
    renaming of its own.
    """
    if isinstance(space, spaces.Box):
        space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.asarray(space.low, dtype=np.float32),
                    high=np.asarray(space.high, dtype=np.float32),
                    shape=space.shape,
                    dtype=np.float32,
                )
            }
        )
    validate_observation_space(space)
    return space


def match_obs_to_buffer(buffer: BaseReplayBuffer, obs: Obs, next_obs: Obs) -> tuple[Obs, Obs]:
    """Wrap flat state ``obs``/``next_obs`` tensors into ``{"state": ...}``
    for every state-only loader's data path (which produces plain tensors
    regardless of what ``infer_specs_from_*`` reports) -- a no-op when
    ``obs`` is already a dict (image-bearing sources). Every replay buffer
    is Dict-observation now (the strict rl-garden contract), so this is
    purely about the loader's own output shape, not the buffer's.
    """
    if not isinstance(obs, dict):
        return {"state": obs}, {"state": next_obs}
    return obs, next_obs


def add_flat_transitions(
    buffer: BaseReplayBuffer,
    obs: Obs,
    next_obs: Obs,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    mc_returns: torch.Tensor | None = None,
    successes: torch.Tensor | None = None,
    episode_ends: torch.Tensor | None = None,
) -> int:
    """``episode_ends`` marks each episode's true final transition (termination
    *or* truncation), independent of ``dones`` -- callers whose source data
    has no way to distinguish the two (e.g. a single combined terminal
    field) can pass the same tensor for both.

    The rows are written ``buffer.num_envs`` at a time in dataset order. A
    buffer whose samples read consecutive rows (``rows_form_trajectories``)
    instead gets one contiguous stretch of the dataset per env. A trajectory
    that continues in the next stretch is cut there: a buffer that samples
    windows (one with an ``episode_ends`` column) ends it (truncated), the
    others stop reading the next row there (``cut_running_episodes``), and a
    Monte Carlo buffer gets the returns of the dataset order.

    Each call starts with ``cut_running_episodes()``: the loaded rows never
    continue an episode the buffer already holds, so a trajectory split over two
    calls is two, the first cut.
    """
    total = actions.shape[0]
    usable = (total // buffer.num_envs) * buffer.num_envs
    if usable <= 0:
        raise ValueError(
            f"Offline dataset has {total} transitions, fewer than num_envs={buffer.num_envs}."
        )

    windowed = hasattr(buffer, "episode_ends")
    if windowed and episode_ends is None:
        raise ValueError(
            f"{type(buffer).__name__} samples windows of consecutive transitions, so its "
            "dataset must say where episodes end; this one does not (episode_ends=None)."
        )
    stretched = windowed or buffer.rows_form_trajectories
    mc_table = None
    written_mask = None
    mc_known = None
    if hasattr(buffer, "_mc_table"):
        if mc_returns is None and stretched and buffer.num_envs > 1:
            mc_returns, mc_known = buffer.stream_mc_returns(
                rewards, episode_ends if episode_ends is not None else dones, successes
            )
        if mc_returns is not None:
            mc_table = torch.zeros_like(buffer.rewards)
            written_mask = torch.zeros_like(buffer.rewards, dtype=torch.bool)
    has_episode_end = episode_ends is not None and (windowed or hasattr(buffer, "_episode_end"))
    per_env = usable // buffer.num_envs
    stretch_starts = torch.arange(buffer.num_envs, device=actions.device) * per_env

    buffer.cut_running_episodes()
    for row in range(per_env):
        start, end = row * buffer.num_envs, (row + 1) * buffer.num_envs
        selected = stretch_starts + row if stretched else slice(start, end)
        pos = buffer.pos
        add_kwargs = {}
        if successes is not None:
            add_kwargs["success"] = successes[selected]
        if has_episode_end:
            ends = episode_ends[selected]
            last_of_stretch = windowed and row == per_env - 1
            add_kwargs["episode_end"] = torch.ones_like(ends) if last_of_stretch else ends
        buffer.add(
            index_obs(obs, selected),
            index_obs(next_obs, selected),
            actions[selected],
            rewards[selected],
            dones[selected],
            **add_kwargs,
        )
        if mc_table is not None:
            mc_table[pos] = mc_returns[selected].to(buffer.storage_device)
            written_mask[pos] = True if mc_known is None else mc_known[selected].to(buffer.storage_device)
    if stretched and not windowed and buffer.num_envs > 1:
        buffer.cut_running_episodes()

    if mc_table is not None:
        # Loader-provided MC values are trusted as-is (see mc_buffer.py's
        # _externally_valid mechanism): this buffer must not recompute them
        # via its own backward-sweep recursion, which cannot see a trajectory
        # continue in the next column. add() already cleared _externally_valid
        # at each written position; restore it here, after the full load, so
        # it isn't immediately wiped by the loop's own add() calls.
        buffer._external_mc = torch.where(written_mask, mc_table, buffer._external_mc)
        buffer._externally_valid = buffer._externally_valid | written_mask
        buffer._mc_table = None
        buffer._valid_table = None
        buffer._valid_indices_cache = None
        buffer._sampleable_size_cache = None
    return usable
