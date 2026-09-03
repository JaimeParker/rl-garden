"""robomimic offline dataset loader.

robomimic (github.com/ARISE-Initiative/robomimic) HDF5 datasets store each
demo under ``data/demo_N``, with a Dict observation (``obs``/``next_obs``
groups keyed by named low-dim sensors). ``ROBOMIMIC_LOW_DIM_OBS_KEYS`` and
``flatten_robomimic_obs`` below are the single source of truth both this
loader and the online env (``rl_garden/envs/robomimic/env.py``) flatten
against -- robomimic's live env force-activates extra observables (any
``joint_pos``/``eef_vel`` match) that are never recorded into the offline
HDF5's ``obs``/``next_obs`` groups, so flattening must select exactly the
keys below, in this order, and raise if one is missing from either side.
If the two sides ever disagreed on the key set or order, RLPD's
offline/online mixed batches would be silently misaligned.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._dataset_common import (
    _add_flat_transitions,
    _concat,
    _load_success,
    _mc_returns,
    _to_tensor,
)
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.base import BaseReplayBuffer

# robot0_eef_quat_site is deliberately excluded: float32 (vs float64 for
# every other key) and redundant with robot0_eef_quat (site vs body-frame
# quaternion of the same end-effector).
ROBOMIMIC_LOW_DIM_OBS_KEYS: tuple[str, ...] = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)


def flatten_robomimic_obs(obs: dict[str, Any]) -> np.ndarray:
    """Concatenate ``obs[key]`` for each key in ``ROBOMIMIC_LOW_DIM_OBS_KEYS``.

    Works on either a single observation (per-key shape ``(dim_k,)``) or a
    whole demo's worth of observations (per-key shape ``(T, dim_k)``), since
    concatenation is always along the last axis. Raises ``KeyError`` naming
    the missing key rather than silently dropping it -- this is the
    parity-enforcing failure point shared by the online env and this loader.
    """
    parts = []
    for key in ROBOMIMIC_LOW_DIM_OBS_KEYS:
        if key not in obs:
            raise KeyError(
                f"robomimic observation is missing expected key {key!r}. "
                f"Available keys: {sorted(obs.keys())}"
            )
        parts.append(np.asarray(obs[key], dtype=np.float64))
    return np.concatenate(parts, axis=-1)


def _require_data_group(f: Any, path: Path) -> Any:
    if "data" not in f:
        raise ValueError(f"No top-level 'data' group found in robomimic hdf5: {path}")
    return f["data"]


def _sorted_demo_keys(data: Any) -> list[str]:
    demo_keys = [key for key in data if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[-1]))


def infer_specs_from_robomimic(path: str | Path) -> tuple[spaces.Box, spaces.Box]:
    """Infer flat Box observation/action spaces from a robomimic HDF5 file."""
    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        data = _require_data_group(f, path)
        demo_keys = _sorted_demo_keys(data)
        if not demo_keys:
            raise ValueError(f"No demo_* groups found in {path}.")
        demo = data[demo_keys[0]]
        obs_dim = int(sum(demo["obs"][key].shape[-1] for key in ROBOMIMIC_LOW_DIM_OBS_KEYS))
        action_dim = int(demo["actions"].shape[-1])

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return obs_space, action_space


def load_robomimic_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    path: str | Path,
    *,
    num_traj: int | None = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Load robomimic ``demo_N`` transitions into an existing replay buffer.

    ``dones`` passed to the buffer is always all-zero: the online rollout
    env (``rl_garden/envs/robomimic/env.py``) never sets ``terminated=True``
    -- robosuite's ``ignore_done=True`` is forced unconditionally and
    ``EnvRobosuite.is_done()`` always returns ``False`` (confirmed by
    reading robomimic's own source) -- so a demo's raw end-of-episode
    ``dones`` marker would otherwise cut TD/MC bootstrap on the offline
    half of every RLPD batch while the online half never does, corrupting
    bootstrap targets. The raw ``dones`` array is instead passed only as
    ``episode_ends`` (trajectory-boundary bookkeeping, and this loader's own
    per-demo MC-return boundary) -- it must never be used to cut bootstrap
    here. If a caller enables ``terminate_on_success`` on the online env,
    this loader's ``dones`` must change together (derive from ``reward >=
    success_threshold`` instead of the raw ``dones`` array), not be left as
    zeros.
    """
    h5py = _require_h5py()
    path = Path(path)
    storage_device = buffer.storage_device
    gamma = float(getattr(buffer, "gamma", 0.99))
    sparse_reward_mc = bool(getattr(buffer, "sparse_reward_mc", False))
    success_threshold = float(getattr(buffer, "success_threshold", 0.5))

    obs_parts = []
    next_obs_parts = []
    action_parts = []
    reward_parts = []
    done_parts = []
    episode_end_parts = []
    mc_parts = []
    success_parts = []
    warned_success_fallback = False

    with h5py.File(path, "r") as f:
        data = _require_data_group(f, path)
        demo_keys = _sorted_demo_keys(data)
        if not demo_keys:
            raise ValueError(f"No demo_* groups found in {path}.")
        if num_traj is not None:
            demo_keys = demo_keys[:num_traj]

        for key in demo_keys:
            demo = _read_node(data[key])
            actions = _to_tensor(demo["actions"], storage_device).float()
            rewards = _to_tensor(demo["rewards"], storage_device).float()
            raw_dones = _to_tensor(demo["dones"], storage_device).bool()
            length = actions.shape[0]

            if reward_scale != 1.0 or reward_bias != 0.0:
                rewards = rewards * reward_scale + reward_bias

            obs = _to_tensor(flatten_robomimic_obs(demo["obs"]), storage_device)
            next_obs = _to_tensor(flatten_robomimic_obs(demo["next_obs"]), storage_device)
            zero_dones = torch.zeros(length, device=storage_device, dtype=torch.float32)

            if sparse_reward_mc:
                success, inferred = _load_success(
                    demo,
                    length,
                    storage_device,
                    success_key=success_key,
                    rewards=rewards,
                    success_threshold=success_threshold,
                )
                success_parts.append(success)
                if inferred and not warned_success_fallback:
                    warnings.warn(
                        "sparse_reward_mc=True but no success field was found "
                        "in the robomimic demo; inferring success from "
                        "reward >= success_threshold.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_success_fallback = True

            obs_parts.append(obs)
            next_obs_parts.append(next_obs)
            action_parts.append(actions)
            reward_parts.append(rewards)
            done_parts.append(zero_dones)
            episode_end_parts.append(raw_dones)
            if (not sparse_reward_mc) and hasattr(buffer, "_mc_table"):
                mc_parts.append(_mc_returns(rewards, raw_dones.float(), gamma))

    if not action_parts:
        raise ValueError(f"No trajectories found in robomimic hdf5 file: {path}")

    obs_all = _concat(obs_parts)
    next_obs_all = _concat(next_obs_parts)
    actions_all = torch.cat(action_parts, dim=0)
    rewards_all = torch.cat(reward_parts, dim=0)
    dones_all = torch.cat(done_parts, dim=0)
    episode_ends_all = torch.cat(episode_end_parts, dim=0)
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
        episode_ends=episode_ends_all,
    )
