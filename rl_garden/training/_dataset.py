"""Shared offline dataset specification and replay-loading helpers.

Used by both the ``offline`` and ``off2on`` training packages so neither
depends on the other's internals.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from rl_garden.buffers import (
    infer_specs_from_h5,
    infer_specs_from_d4rl_legacy,
    infer_specs_from_minari,
    infer_specs_from_ogbench,
    infer_specs_from_rlbench,
    infer_specs_from_robomimic,
    load_d4rl_legacy_dataset_to_replay_buffer,
    load_h5_dataset_to_replay_buffer,
    load_minari_dataset_to_replay_buffer,
    load_ogbench_dataset_to_replay_buffer,
    load_rlbench_dataset_to_replay_buffer,
    load_robomimic_dataset_to_replay_buffer,
)


def infer_offline_dataset_specs(args: Any) -> tuple[spaces.Space, spaces.Box]:
    """Infer observation and action spaces for the configured dataset backend."""
    if args.dataset_backend == "d4rl_legacy":
        return infer_specs_from_d4rl_legacy(args.offline_dataset)
    if args.dataset_backend == "minari":
        obs_space, action_space = infer_specs_from_minari(args.offline_dataset)
        if not isinstance(action_space, spaces.Box):
            raise ValueError(
                f"Minari dataset {args.offline_dataset!r} has a "
                f"{type(action_space).__name__} action space; only continuous "
                "(Box) actions are supported."
            )
        return obs_space, action_space
    if args.dataset_backend == "h5":
        return infer_specs_from_h5(
            args.offline_dataset,
            action_low=args.action_low,
            action_high=args.action_high,
        )
    if args.dataset_backend == "robomimic":
        return infer_specs_from_robomimic(args.offline_dataset)
    if args.dataset_backend == "ogbench":
        return infer_specs_from_ogbench(args.offline_dataset)
    if args.dataset_backend == "rlbench":
        return infer_specs_from_rlbench(args.offline_dataset)
    raise ValueError(f"Unsupported offline dataset backend: {args.dataset_backend!r}")


def load_offline_dataset(replay_buffer: Any, args: Any) -> int:
    """Load the configured offline dataset into ``replay_buffer``."""
    common_kwargs = {
        "reward_scale": args.reward_scale,
        "reward_bias": args.reward_bias,
        "success_key": args.success_key,
    }
    if args.dataset_backend == "d4rl_legacy":
        return load_d4rl_legacy_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_episodes=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_backend == "minari":
        return load_minari_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_episodes=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_backend == "h5":
        return load_h5_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_traj=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_backend == "robomimic":
        return load_robomimic_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_traj=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_backend == "ogbench":
        return load_ogbench_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_traj=args.offline_num_traj,
            **common_kwargs,
        )
    if args.dataset_backend == "rlbench":
        return load_rlbench_dataset_to_replay_buffer(
            replay_buffer,
            args.offline_dataset,
            num_traj=args.offline_num_traj,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported offline dataset backend: {args.dataset_backend!r}")
