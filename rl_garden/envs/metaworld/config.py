"""Implementation-layer config for the Meta-World env backend."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    # From EnvRequest.observation: which cameras (every Meta-World v3 task
    # scene defines the same 6: "corner", "corner2", "corner3", "corner4",
    # "behindGripper", "gripperPOV") render rgb / depth, whether to keep the
    # flat proprioceptive "state" key, and the render resolution. Only
    # consulted for a single-task env_id -- MT10/MT50 build their sub-envs
    # internally with no per-env construction hook to attach a camera
    # renderer to.
    rgb_cameras: tuple[str, ...] = field(default_factory=tuple)
    depth_cameras: tuple[str, ...] = field(default_factory=tuple)
    state: bool = True
    image_size: tuple[int, int] = (84, 84)
    # Stacks each rgb_<cam>/depth_<cam> key into a leading time dimension via
    # ImageFrameStackWrapper (see env.py) -- only valid together with
    # rgb_cameras/depth_cameras, checked at the backend (envs/backends/
    # metaworld.py).
    frame_stack: int = 1
