"""``CustomMujocoWarpEnv`` reimplementation of Gymnasium's ``InvertedPendulum-v4``.

Logic (reward, termination condition, reset noise) is copied from
``gymnasium.envs.mujoco.inverted_pendulum_v4.InvertedPendulumEnv`` (the same
ground truth the CPU ``InvertedPendulumCustomEnv`` parity task uses), batched
over ``nworld`` with torch tensor ops instead of per-env scalars. This exists
to verify parity against both the official Gymnasium task and the CPU
``CustomMujocoEnv`` scaffold.

State-only variant reuses Gymnasium's bundled ``inverted_pendulum.xml``
verbatim (same physics as the CPU parity task). The camera variant needs its
own local asset with an explicit ``<camera>`` added -- see
``assets/inverted_pendulum_camera.xml`` and the
``custom_mujoco_warp_env.py`` module docstring for why.
"""
from __future__ import annotations

import os

import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env as _gym_mujoco_env
from gymnasium.vector.utils import batch_space

from rl_garden.envs.mujoco_warp.custom_mujoco_warp_env import CustomMujocoWarpEnv
from rl_garden.envs.mujoco_warp.env import register_mujoco_warp_task

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Gymnasium's own asset-path resolver; reused so the state-only variant can
# point directly at the bundled model, matching the CPU parity task's model.
_GYM_PENDULUM_PATH = _gym_mujoco_env.expand_model_path("inverted_pendulum.xml")
_CAMERA_PENDULUM_PATH = os.path.join(_ASSETS_DIR, "inverted_pendulum_camera.xml")

_MAX_EPISODE_STEPS = 1000


class InvertedPendulumWarpEnv(CustomMujocoWarpEnv):
    _MODEL_PATH = _GYM_PENDULUM_PATH

    def __init__(self, nworld: int, device: str, **kwargs) -> None:
        observation_space = spaces.Dict(
            {"state": spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,))}
        )
        # frame_skip=2 matches gymnasium's InvertedPendulumEnv/the CPU
        # InvertedPendulumCustomEnv parity task -- without this, mjw.step()
        # (one physics timestep per call) advances only half as much
        # simulated time per action as the CPU version, and trajectories
        # diverge sharply within a handful of steps (verified: divergence of
        # ~0.7 rad in pole angle by step 4 with frame_skip=1).
        kwargs.setdefault("frame_skip", 2)
        super().__init__(self._MODEL_PATH, nworld, device, observation_space, **kwargs)
        self._nq = self._mjm.nq
        self._nv = self._mjm.nv
        self._elapsed = torch.zeros(nworld, dtype=torch.int64, device=self.device)

    def _flat_state(self) -> torch.Tensor:
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)
        return torch.cat([qpos, qvel], dim=-1)

    def _apply_action(self, actions: torch.Tensor) -> None:
        wp.to_torch(self.data.ctrl)[:] = actions

    def _get_obs(self) -> dict:
        return {"state": self._flat_state()}

    def _get_reward(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _get_terminated(self) -> torch.Tensor:
        state = self._flat_state()
        finite = torch.isfinite(state).all(dim=-1)
        pole_ok = state[:, 1].abs() <= 0.2
        return ~(finite & pole_ok)

    def _get_truncated(self) -> torch.Tensor:
        self._elapsed += 1
        return self._elapsed >= _MAX_EPISODE_STEPS

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return
        noise_qpos = (torch.rand((n, self._nq), device=self.device) * 2 - 1) * 0.01
        noise_qvel = (torch.rand((n, self._nv), device=self.device) * 2 - 1) * 0.01
        wp.to_torch(self.data.qpos)[env_ids] = noise_qpos
        wp.to_torch(self.data.qvel)[env_ids] = noise_qvel
        mjw.forward(self.model, self.data)  # required after manual qpos/qvel writes.
        self._elapsed[env_ids] = 0


class InvertedPendulumCameraWarpEnv(InvertedPendulumWarpEnv):
    """Adds an rgb+depth camera observation to verify
    CustomMujocoWarpEnv._render_cameras() end to end."""

    _MODEL_PATH = _CAMERA_PENDULUM_PATH

    def __init__(self, nworld: int, device: str, **kwargs) -> None:
        kwargs.setdefault("camera_width", 64)
        kwargs.setdefault("camera_height", 64)
        kwargs.setdefault("render_rgb", True)
        kwargs.setdefault("render_depth", True)
        super().__init__(nworld, device, **kwargs)
        # Must match _get_obs()'s actual keys exactly: _render_cameras() only
        # includes "rgb"/"depth" when the corresponding render_rgb/
        # render_depth flag is set (an unallocated buffer can't be read at
        # all, so this isn't a placeholder-vs-real-shape nuance like on some
        # other backends -- a declared-but-absent key here means _get_obs()
        # would KeyError or the caller silently expects a key that never
        # arrives).
        obs_spaces = {"state": spaces.Box(low=-float("inf"), high=float("inf"), shape=(4,))}
        if self._render_rgb:
            obs_spaces["rgb"] = spaces.Box(
                low=0, high=255, shape=(self._camera_height, self._camera_width, 3), dtype="uint8"
            )
        if self._render_depth:
            obs_spaces["depth"] = spaces.Box(
                low=0, high=float("inf"), shape=(self._camera_height, self._camera_width)
            )
        self.single_observation_space = spaces.Dict(obs_spaces)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _get_obs(self) -> dict:
        return {"state": self._flat_state(), **self._render_cameras()}


register_mujoco_warp_task("RlGarden-InvertedPendulum-Warp-v0", InvertedPendulumWarpEnv)
register_mujoco_warp_task(
    "RlGarden-InvertedPendulum-Camera-Warp-v0", InvertedPendulumCameraWarpEnv
)
