"""``CustomEnvConfig``: the concrete task config, translated from
``EnvRequest`` by ``rl_garden.envs.backends.custom.CustomBackend``."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CustomEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
