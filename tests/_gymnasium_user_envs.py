"""Stand-in for a user's own env module, for ``tests/test_gymnasium_backend.py``.

Importing this module registers two gymnasium ids, the way a user's package
would when listed in ``--gymnasium.import_modules``:

- ``RlgTestPixels-v0``: Dict observation already keyed by the rl-garden
  contract (``rgb_front`` uint8 16x16x3 + ``state`` float32 (2,)), reward 1.0
  per step, truncated after 3 steps.
- ``RlgTestPreWrapped-v0``: the same env, whose entry point already applies
  ``RecordEpisodeStatistics`` outside its own 3-step ``TimeLimit`` (so the
  statistics wrapper sees the truncation).
"""
from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit


class PixelsStateEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "rgb_front": spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 1.0, False, False, {}

    def _observation(self):
        return {
            "rgb_front": self.np_random.integers(0, 256, size=(16, 16, 3), dtype=np.uint8),
            "state": self.np_random.standard_normal(2).astype(np.float32),
        }


def make_pre_wrapped_env() -> gym.Env:
    return RecordEpisodeStatistics(TimeLimit(PixelsStateEnv(), max_episode_steps=3))


if "RlgTestPixels-v0" not in gym.registry:
    gym.register("RlgTestPixels-v0", entry_point=PixelsStateEnv, max_episode_steps=3)
if "RlgTestPreWrapped-v0" not in gym.registry:
    gym.register("RlgTestPreWrapped-v0", entry_point=make_pre_wrapped_env)
