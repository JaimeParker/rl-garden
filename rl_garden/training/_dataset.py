"""Shared offline dataset specification and replay-loading helpers.

Used by both the ``offline`` and ``off2on`` training packages so neither
depends on the other's internals. Backend-agnostic: dispatch to the actual
per-format loader goes through ``rl_garden.buffers.dataset_backend_registry``
(same registry pattern as ``rl_garden.envs.backend_registry``'s
``EnvBackend``), so adding a new ``--dataset_backend`` never requires
touching this file.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from rl_garden.buffers import DatasetRequest, infer_dataset_specs, load_dataset


def _dataset_request(args: Any, *, num_traj: int | None = None) -> DatasetRequest:
    return DatasetRequest(
        path=args.offline_dataset,
        num_traj=num_traj,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        success_key=args.success_key,
        action_low=args.action_low,
        action_high=args.action_high,
        obs_mode=getattr(args, "obs_mode", None),
        # Per-backend CLI config (e.g. RLBenchConfig), keyed directly off
        # dataset_backend -- correct even when --env_backend differs from
        # --dataset_backend, unlike args.resolve_backend_config() (which is
        # keyed off env_backend).
        backend_config=getattr(args, args.dataset_backend, None),
    )


def infer_offline_dataset_specs(args: Any) -> tuple[spaces.Space, spaces.Box]:
    """Infer observation and action spaces for the configured dataset backend."""
    return infer_dataset_specs(_dataset_request(args), backend_name=args.dataset_backend)


def load_offline_dataset(replay_buffer: Any, args: Any) -> int:
    """Load the configured offline dataset into ``replay_buffer``."""
    return load_dataset(
        replay_buffer,
        _dataset_request(args, num_traj=args.offline_num_traj),
        backend_name=args.dataset_backend,
    )
