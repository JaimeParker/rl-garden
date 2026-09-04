"""Implementation-layer config for the OGBench env backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OGBenchEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    # Per-task kwargs forwarded verbatim to gym.make (rarely needed -- every
    # OGBench task variant, including obs modality, is already encoded in
    # env_id itself).
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # "sync": single-process gymnasium.vector.SyncVectorEnv (default).
    # "async": one OS process per env -- recommended for visual-* env ids,
    # since each instance owns its own MuJoCo renderer/GL context (same
    # reasoning as rl_garden.envs.mujoco.env's vectorization knob).
    vectorization: str = "sync"
