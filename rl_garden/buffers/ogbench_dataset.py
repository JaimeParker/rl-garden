"""Utilities for loading OGBench offline datasets into replay buffers.

OGBench's own ``relabel_dataset()`` (invoked internally by
``ogbench.make_env_and_datasets`` for any ``*-singletask*`` dataset name)
already computes exactly the two fields this loader needs, so no
per-family reward-recomputation code is needed here (unlike
``d4rl_legacy_dataset.py``):

- ``masks`` (0 only when the task is complete, else 1) maps directly to
  ``done`` for TD bootstrap-cut: ``done = 1 - masks``.
- ``terminals`` (1 at each fixed-length trajectory's last step, regardless
  of task completion -- a real dataset trajectory boundary) maps directly
  to ``episode_end``.

For these sparse-reward singletask tasks, success is algebraically
identical to ``1 - masks`` (success is the only thing that flips a mask to
0), so ``successes = dones`` -- the same convention
``d4rl_legacy_dataset.py``'s antmaze branch already uses.

Non-singletask (goal-conditioned) dataset names carry no ``rewards``/
``masks`` at all; loading one raises rather than silently loading a
rewardless dataset into a reward-based replay buffer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from rl_garden.buffers._dataset_common import _add_flat_transitions, _mc_returns, _to_tensor
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.spaces import canonicalize_floating_observation_space

DEFAULT_DATASET_DIR = "~/.ogbench/data"


def _require_ogbench():
    try:
        import ogbench  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without ogbench.
        raise ImportError(
            "Loading an OGBench dataset requires the `ogbench` package. Install "
            "rl-garden with the `ogbench` extra or install ogbench directly."
        ) from exc
    return ogbench


def infer_specs_from_ogbench(
    dataset_name: str, *, dataset_dir: str = DEFAULT_DATASET_DIR
) -> tuple[spaces.Box, spaces.Box]:
    """Return the observation/action spaces for an OGBench dataset's env."""
    ogbench = _require_ogbench()
    env = ogbench.make_env_and_datasets(dataset_name, dataset_dir=dataset_dir, env_only=True)
    try:
        return (
            canonicalize_floating_observation_space(env.observation_space),
            env.action_space,
        )
    finally:
        env.close()


def _truncate_to_num_traj(dataset: dict[str, Any], num_traj: int | None) -> dict[str, Any]:
    if num_traj is None:
        return dataset
    terminal_idx = np.flatnonzero(dataset["terminals"])
    if len(terminal_idx) < num_traj:
        return dataset
    cutoff = int(terminal_idx[num_traj - 1]) + 1
    return {key: value[:cutoff] for key, value in dataset.items()}


def load_ogbench_dataset_to_replay_buffer(
    buffer: BaseReplayBuffer,
    dataset_name: str,
    *,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    num_traj: int | None = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    success_key: str | None = None,
) -> int:
    """Load an OGBench singletask dataset's transitions into a replay buffer.

    ``reward_scale``/``reward_bias`` apply ``r := scale * r + bias`` to each
    loaded reward, matching every other loader's convention.
    """
    del success_key  # ogbench's own masks/terminals fully determine done/episode_end/success.
    ogbench = _require_ogbench()
    storage_device = buffer.storage_device

    env, train_dataset, _val_dataset = ogbench.make_env_and_datasets(
        dataset_name, dataset_dir=dataset_dir
    )
    try:
        if "rewards" not in train_dataset or "masks" not in train_dataset:
            raise ValueError(
                f"OGBench dataset {dataset_name!r} has no 'rewards'/'masks' fields -- "
                "only *-singletask-* dataset names carry reward labels (goal-conditioned "
                "datasets do not)."
            )
        dataset = _truncate_to_num_traj(train_dataset, num_traj)

        rewards = _to_tensor(dataset["rewards"], storage_device).float()
        if reward_scale != 1.0 or reward_bias != 0.0:
            rewards = rewards * reward_scale + reward_bias
        dones = 1.0 - _to_tensor(dataset["masks"], storage_device).float()
        episode_ends = _to_tensor(dataset["terminals"], storage_device).bool()
        obs = _to_tensor(dataset["observations"], storage_device)
        next_obs = _to_tensor(dataset["next_observations"], storage_device)
        actions = _to_tensor(dataset["actions"], storage_device).float()

        successes = dones if hasattr(buffer, "_step_success") else None
        mc_returns = (
            _mc_returns(rewards, dones, float(buffer.gamma))
            if hasattr(buffer, "_mc_table") and hasattr(buffer, "gamma")
            else None
        )

        return _add_flat_transitions(
            buffer,
            obs,
            next_obs,
            actions,
            rewards,
            dones,
            mc_returns,
            successes,
            episode_ends=episode_ends,
        )
    finally:
        env.close()
