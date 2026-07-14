"""Converts the quaternion slice of a Franka real-robot env's ``state``
observation into a 3D rotation vector, matching the rotation representation
the action side already uses (``FrankaRealEnv.step()``'s rotvec deltas).
Optional and off by default -- HIL-SERL converts to Euler via
``Quat2EulerWrapper`` instead, but rotvec avoids Euler's gimbal-lock/wraparound
special-casing and stays consistent with this repo's own action convention.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from scipy.spatial.transform import Rotation

# FrankaRealEnv's "state" layout: pose (xyz[0:3] + quat_xyzw[3:7]) + vel(6) +
# gripper_pos(1) + force(3) + torque(3) -- see rl_garden/envs/franka_real/env.py.
_QUAT_START = 3
_QUAT_END = 7


class RotvecObsWrapper(gym.ObservationWrapper):
    """Shrinks ``obs["state"]`` from ``_STATE_DIM`` (20) to 19 by replacing
    the 4D quaternion at indices [3:7] with its 3D rotation vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        state_space = env.single_observation_space["state"]
        new_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_space.shape[0] - 1,),
            dtype=state_space.dtype,
        )
        new_spaces = dict(env.single_observation_space.spaces)
        new_spaces["state"] = new_state_space
        self.single_observation_space = spaces.Dict(new_spaces)
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)

    def __getattr__(self, name: str):
        # See RewardClassifierWrapper -- gymnasium.Wrapper (>=1.0) no longer
        # forwards arbitrary attributes to self.env, but this repo's env
        # backend contract (num_envs, single_observation_space, ...) relies
        # on direct attribute access.
        return getattr(self.env, name)

    def observation(self, obs: dict) -> dict:
        state = obs["state"]
        xyz = state[..., :_QUAT_START]
        quat = state[..., _QUAT_START:_QUAT_END]
        rest = state[..., _QUAT_END:]
        rotvec = torch.as_tensor(
            Rotation.from_quat(quat.detach().cpu().numpy()).as_rotvec(),
            dtype=state.dtype,
            device=state.device,
        )
        return {**obs, "state": torch.cat([xyz, rotvec, rest], dim=-1)}
