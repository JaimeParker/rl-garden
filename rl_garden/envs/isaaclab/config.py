from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsaacLabEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    headless: bool = True
    sim_device: str = "cuda:0"
    obs_mode: str = "state"
    frame_stack: int = 1
    env_kwargs: dict = field(default_factory=dict)
