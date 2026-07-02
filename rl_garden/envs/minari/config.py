"""Full env-factory config for Minari-backed environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MinariEnvConfig:
    dataset_id: str
    num_envs: int
    eval_env: bool
    device: str = "cpu"
    download: bool = True
