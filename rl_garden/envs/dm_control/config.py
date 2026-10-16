"""Implementation-layer config for the dm_control env backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DMControlEnvConfig:
    # "<domain>-<task>[-hard]" (NCRL TASK_SET naming, see tasks.py).
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # "sync": single-process gymnasium.vector.SyncVectorEnv (default).
    # "async": one OS process per env (each owns its own GL context).
    vectorization: str = "sync"
    # Physics frames per policy step; rewards are summed over the repeat.
    action_repeat: int = 2
    # Physics frames per episode (truncation); NCRL's MAX_EPISODE_LENGTH_DMC.
    episode_frames: int = 1000
    # Sparse-reward + action-penalty variant (NCRL HardTaskWrapper). Also
    # switched on by a "-hard" env_id suffix.
    hard: bool = False
    # MuJoCo camera index rendered for the "default" camera; None -> 2 for
    # quadruped, 0 otherwise (NCRL's choice).
    camera_id: Optional[int] = None
    # From EnvRequest.observation.
    state: bool = True
    rgb_cameras: tuple[str, ...] = field(default_factory=tuple)
    image_size: tuple[int, int] = (64, 64)
    frame_stack: int = 1
