from __future__ import annotations

from dataclasses import dataclass


@dataclass
class D4RLLegacyEnvConfig:
    env_id: str
    num_envs: int = 1
    device: str = "cpu"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
