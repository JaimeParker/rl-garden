"""``CustomMujocoEnv`` reimplementation of Gymnasium's ``InvertedPendulum-v4``.

Logic (reward, termination condition, reset noise) is copied from
``gymnasium.envs.mujoco.inverted_pendulum_v4.InvertedPendulumEnv``. This
exists to verify parity: ``CustomMujocoEnv``'s generic ``step``/
``reset_model`` should behave identically to the official hand-written
version under the same seed and action sequence — only the observation
container (rl-garden's ``{"state": ...}`` dict instead of a bare array) and,
for the camera variant, the added image keys differ.

Reuses Gymnasium's bundled ``inverted_pendulum.xml`` model on purpose (this
is a parity-verification task, not a template for a real custom task — a
real custom task would point ``model_path`` at its own MJCF file).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.envs.mujoco.custom_mujoco_env import CustomMujocoEnv


class InvertedPendulumCustomEnv(CustomMujocoEnv):
    def __init__(self, **kwargs):
        # Dict, not a bare Box: _get_obs() always returns rl-garden's
        # {"state": ...} convention, and the declared observation_space must
        # match that shape exactly -- AsyncVectorEnv's shared-memory
        # transport sizes/types its buffers from this declaration, so it is
        # not just a placeholder the way it is for some other backends.
        observation_space = spaces.Dict(
            {"state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)}
        )
        super().__init__(
            "inverted_pendulum.xml", frame_skip=2, observation_space=observation_space, **kwargs
        )

    def _flat_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def _apply_action(self, action) -> None:
        self.do_simulation(action, self.frame_skip)

    def _get_obs(self) -> dict:
        return {"state": self._flat_state()}

    def _get_reward(self, action) -> float:
        return 1.0

    def _get_terminated(self) -> bool:
        state = self._flat_state()
        return bool(not np.isfinite(state).all() or abs(state[1]) > 0.2)

    def _sample_reset_state(self, qpos: np.ndarray, qvel: np.ndarray):
        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        return qpos, qvel


class InvertedPendulumCameraCustomEnv(InvertedPendulumCustomEnv):
    """Adds a free-camera (the model has no named camera) rgb+depth observation,
    to verify the CustomMujocoEnv._render_cameras() path end to end. Depth
    semantics are unverified (see custom_mujoco_env.py module docstring) --
    this task exists to observe and record what MujocoRenderer actually
    returns, not to claim a specific depth convention.
    """

    _CAMERA_SIZE = 64

    def __init__(self, **kwargs):
        super().__init__(
            camera_configs=[
                {
                    "camera_id": -1,
                    "width": self._CAMERA_SIZE,
                    "height": self._CAMERA_SIZE,
                    "rgb": True,
                    "depth": True,
                }
            ],
            **kwargs,
        )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
                "rgb": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._CAMERA_SIZE, self._CAMERA_SIZE, 3),
                    dtype=np.uint8,
                ),
                "depth": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._CAMERA_SIZE, self._CAMERA_SIZE),
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self) -> dict:
        return {"state": self._flat_state(), **self._render_cameras()}


gym.register(
    id="RlGarden-InvertedPendulum-Custom-v0",
    entry_point=f"{__name__}:InvertedPendulumCustomEnv",
    max_episode_steps=1000,
    disable_env_checker=True,  # avoid the default checker's assumptions about
    # custom mujoco tasks / Dict observation spaces (camera variant).
)

gym.register(
    id="RlGarden-InvertedPendulum-Camera-Custom-v0",
    entry_point=f"{__name__}:InvertedPendulumCameraCustomEnv",
    max_episode_steps=1000,
    disable_env_checker=True,
)
