"""Shared offline dataset specification and replay-loading helpers.

Used by both the ``offline`` and ``off2on`` training packages so neither
depends on the other's internals.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from rl_garden.algorithms import infer_specs_from_h5
from rl_garden.buffers import (
    infer_specs_from_minari,
    load_maniskill_h5_to_replay_buffer,
    load_minari_dataset_to_replay_buffer,
)


def infer_offline_dataset_specs(args: Any) -> tuple[spaces.Space, spaces.Box]:
    """Infer observation and action spaces for the configured dataset source."""
    if args.dataset_source == "minari":
        obs_space, action_space = infer_specs_from_minari(args.offline_dataset_path)
        if not isinstance(action_space, spaces.Box):
            raise ValueError(
                f"Minari dataset {args.offline_dataset_path!r} has a "
                f"{type(action_space).__name__} action space; only continuous "
                "(Box) actions are supported."
            )
        return obs_space, action_space
    if args.dataset_source == "maniskill_h5":
        return infer_specs_from_h5(
            args.offline_dataset_path,
            action_low=args.action_low,
            action_high=args.action_high,
        )
    raise ValueError(f"Unsupported offline dataset source: {args.dataset_source!r}")


def load_offline_dataset(replay_buffer: Any, args: Any) -> int:
    """Load the configured offline dataset into ``replay_buffer``."""
    common_kwargs = {
        "reward_scale": args.reward_scale,
        "reward_bias": args.reward_bias,
        "success_key": args.success_key,
    }
    if args.dataset_source == "minari":
        return load_minari_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset_path,
            num_episodes=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_source == "maniskill_h5":
        return load_maniskill_h5_to_replay_buffer(
            replay_buffer,
            args.offline_dataset_path,
            num_traj=args.offline_num_traj,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported offline dataset source: {args.dataset_source!r}")
