"""Shared offline dataset specification and replay-loading helpers."""

from __future__ import annotations

from typing import Any

from rl_garden.algorithms import infer_specs_from_h5
from rl_garden.buffers import (
    infer_specs_from_minari,
    load_maniskill_h5_to_replay_buffer,
    load_minari_dataset_to_replay_buffer,
)


def infer_offline_dataset_specs(args: Any):
    """Infer observation and action spaces for the configured dataset source."""
    if args.dataset_source == "minari":
        return infer_specs_from_minari(args.offline_dataset_path)
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
