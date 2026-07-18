"""Implementation-layer config for the MuJoCo env backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MujocoEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    # Per-task kwargs forwarded verbatim to gym.make (e.g. forward_reward_weight,
    # ctrl_cost_weight, reset_noise_scale) — task-specific, not rl-garden fields.
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # "sync": single-process gymnasium.vector.SyncVectorEnv (default, matches
    # the CPU v1 Gymnasium-benchmark-task path). "async": one OS process per
    # env via AsyncVectorEnv — required for custom tasks with camera obs,
    # since MujocoRenderer's OpenGL context isn't verified safe to share
    # across env instances in one process.
    vectorization: str = "sync"
