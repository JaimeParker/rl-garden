"""Shared args for human-in-the-loop split actor/learner methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence


@dataclass
class HITLArgs:
    role: Literal["actor", "learner", "eval"] = "actor"

    # Actor: address of the learner's sync server. Learner: bind address.
    sync_host: str = "127.0.0.1"
    sync_port: int = 6000
    sync_monitor_interval_s: float = 5.0

    control_hz: float = 10.0
    deterministic_actor: bool = False
    actor_show_rgb_window: bool = True
    actor_rgb_window_name: str = "residual_hil_serl_actor"
    vis_camera_width: Optional[int] = 256
    vis_camera_height: Optional[int] = 256
    train_freq: int = 1
    publish_freq: int = 100

    teleop_device: Literal["pico", "spacemouse"] = "spacemouse"
    teleop_record_gripper: bool = True
    teleop_init_timeout_s: float = 120.0

    demo_buffer_size: int = 100_000
    demo_data_ratio: float = 0.5
    buffer_period: int = 1000
    demo_dataset_paths: Sequence[str] = field(default_factory=tuple)

    eval_n_trajs: int = 10
