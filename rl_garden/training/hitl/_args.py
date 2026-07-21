"""Shared args for human-in-the-loop split actor/learner methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


@dataclass
class HITLArgs:
    role: Literal["actor", "learner", "eval"] = "actor"

    # Actor: address of the learner's sync server. Learner: bind address.
    sync_host: str = "127.0.0.1"
    sync_port: int = 6000

    control_hz: float = 10.0
    deterministic_actor: bool = False
    train_freq: int = 1
    publish_freq: int = 100

    teleop_device: Literal["pico", "spacemouse"] = "pico"
    teleop_record_gripper: bool = True

    demo_buffer_size: int = 100_000
    demo_data_ratio: float = 0.5
    buffer_period: int = 1000
    demo_dataset_paths: Sequence[str] = field(default_factory=tuple)

    eval_n_trajs: int = 10
