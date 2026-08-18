"""``PointReach-v0``: a minimal single-instance ``gymnasium.Env`` task.

This is the part a new environment author actually writes: standard
Gymnasium authoring (``reset``/``step`` on plain numpy arrays, no torch, no
vectorization) -- exactly the pattern documented at
https://gymnasium.farama.org/introduction/create_custom_env/, current as of
gymnasium 1.3 (``reset(*, seed, options) -> (obs, info)``,
``step(action) -> (obs, reward, terminated, truncated, info)``, the
post-0.26 5-tuple API this whole repo already targets).

Task: a 2D point mass under force control must reach a randomly placed
target. Reward is negative distance to target; episode terminates on
success, truncates via the ``max_episode_steps`` passed to
``gymnasium.register`` in this module (see bottom of file) -- vectorization
and torch conversion are handled centrally by
``rl_garden.envs.custom.env.make_custom_env``, not here.

Observation convention: rl-garden's encoders expect a dict observation even
for state-only tasks (see ``rl_garden.envs.mujoco.custom_mujoco_env``'s
module docstring for the same rule) -- ``"state"`` holds the flat
proprioceptive vector: ``[pos_x, pos_y, vel_x, vel_y, target_x, target_y]``.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

_ARENA_HALF_EXTENT = 5.0
_SUCCESS_THRESHOLD = 0.2
_FORCE_SCALE = 5.0
_DT = 0.05
_DAMPING = 0.98


class PointReachEnv(gym.Env):
    """2D point-mass reach task. Copy this file to author your own task."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                )
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        self._pos = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._target = self.np_random.uniform(
            -_ARENA_HALF_EXTENT, _ARENA_HALF_EXTENT, size=2
        ).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._vel = (self._vel + action * _FORCE_SCALE * _DT) * _DAMPING
        self._pos = np.clip(
            self._pos + self._vel * _DT, -_ARENA_HALF_EXTENT, _ARENA_HALF_EXTENT
        )

        distance = float(np.linalg.norm(self._target - self._pos))
        reward = -distance
        terminated = distance < _SUCCESS_THRESHOLD
        if terminated:
            reward += 10.0

        return self._get_obs(), reward, terminated, False, {"distance": distance}

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "state": np.concatenate([self._pos, self._vel, self._target]).astype(
                np.float32
            )
        }


gym.register(
    id="PointReach-v0",
    entry_point="rl_garden.envs.custom.point_reach_env:PointReachEnv",
    max_episode_steps=200,
)
