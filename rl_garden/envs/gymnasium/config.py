"""Implementation-layer config for the generic ``gymnasium`` env backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GymnasiumEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    # Imported inside every env thunk before gym.make, so registration also
    # happens in AsyncVectorEnv workers that don't inherit parent imports.
    import_modules: tuple[str, ...] = ()
    # Forwarded verbatim to gym.make.
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    # "sync": single-process SyncVectorEnv (default); "async": one OS process
    # per env (AsyncVectorEnv).
    vectorization: str = "sync"
    # rgb_<cam>/depth_<cam> keys the env emits itself; stacked when
    # frame_stack > 1.
    image_keys: tuple[str, ...] = ()
    frame_stack: int = 1
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Env observation key -> rl-garden contract key, applied at the vector
    # level after the numpy-to-torch adapter.
    obs_key_map: dict[str, str] = field(default_factory=dict)
    # "pkg.module:function" replacing gym.make + Sync/AsyncVectorEnv.
    vector_env_factory: str = ""
    vector_env_returns_torch: bool = False
