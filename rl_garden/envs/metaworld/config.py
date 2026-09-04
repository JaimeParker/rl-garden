"""Implementation-layer config for the Meta-World env backend."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetaWorldEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # "sync": single-process gymnasium.vector.SyncVectorEnv (default).
    # "async": one OS process per env.
    vectorization: str = "sync"
    # Appends a one-hot task id to the observation. Only consulted when
    # env_id is "MT10"/"MT50" -- ignored for a single-task env_id.
    use_one_hot: bool = True
    # "state" (default) or "rgb". "rgb" only supported for a single-task
    # env_id -- MT10/MT50 build their sub-envs internally with no per-env
    # construction hook to attach a camera renderer to.
    obs_mode: str = "state"
    # Fixed camera name; every Meta-World v3 task scene defines the same 6
    # ("corner", "corner2", "corner3", "corner4", "behindGripper",
    # "gripperPOV"). Only consulted when obs_mode == "rgb".
    camera: str = "corner2"
    image_size: tuple[int, int] = (84, 84)
