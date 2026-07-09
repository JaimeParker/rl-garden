"""FrankaRealEnv: gym.Env talking to robot_infra's Franka bridge over HTTP.

Action/observation conventions are copied directly from SERL: a 7D action
(delta EE position xyz + delta rotation as axis-angle (3) + gripper), and a
dict observation with ``state`` (pose 7 + vel 6 + gripper 1 + force 3 +
torque 3, flattened to one 20-D vector) plus one entry per configured camera
key. State field names/HTTP shape match SERL's ``franka_server.py``
``/getstate`` response exactly (``pose``/``vel``/``force``/``torque``/
``gripper_pos``) -- see ``bridge_client.py``.

Camera frames are *not* proxied through the bridge: SERL's own reference
architecture captures cameras directly in the process that owns the env
(e.g. via a RealSense SDK), not on the robot-side Flask server, so
``camera_capture`` here is an optional, lazily-supplied callable rather than
another bridge endpoint -- this keeps the bridge itself a pure pose/state
forwarder and keeps camera SDK dependencies out of this module entirely.

Exposed as a batch-of-1 "vectorized" env (``num_envs = 1``) so it satisfies
the same contract every other rl-garden env backend does
(``.agents/rules/adding-env-backend.md``): ``step``/``reset`` return torch
tensors with a leading ``(1, ...)`` batch dim.

Control-loop pacing (holding a fixed Hz) is *not* this env's job --
``ActorLoop`` (``rl_garden/real_world/actor_loop.py``) owns pacing so there
is a single source of truth for the real-time budget. ``step()`` here simply
executes as fast as the bridge responds.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from scipy.spatial.transform import Rotation

from rl_garden.envs.franka_real.bridge_client import FrankaBridgeClient
from rl_garden.envs.franka_real.config import FrankaRealEnvConfig

_STATE_DIM = 7 + 6 + 1 + 3 + 3  # pose + vel + gripper_pos + force + torque

CameraCapture = Callable[[], dict[str, np.ndarray]]


class FrankaRealEnv(gym.Env):
    num_envs = 1

    def __init__(
        self,
        cfg: FrankaRealEnvConfig,
        bridge_client: Optional[FrankaBridgeClient] = None,
        camera_capture: Optional[CameraCapture] = None,
    ) -> None:
        if cfg.camera_keys and camera_capture is None:
            raise ValueError(
                "cfg.camera_keys is non-empty but no camera_capture callable was "
                "provided; FrankaRealEnv does not talk to cameras through the "
                "bridge (see module docstring)."
            )
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.bridge = (
            bridge_client if bridge_client is not None else FrankaBridgeClient(cfg.bridge_url)
        )
        self.camera_capture = camera_capture

        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(_STATE_DIM,), dtype=np.float32),
        }
        for key in cfg.camera_keys:
            obs_spaces[key] = spaces.Box(
                low=0,
                high=255,
                shape=(cfg.camera_height, cfg.camera_width, 3),
                dtype=np.uint8,
            )
        self.single_observation_space = spaces.Dict(obs_spaces)
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)

        self._safety_low = np.array(cfg.safety_box_low, dtype=np.float64)
        self._safety_high = np.array(cfg.safety_box_high, dtype=np.float64)
        self._t = 0
        self._last_pose: Optional[np.ndarray] = None  # xyz(3) + quat_xyzw(4)

    def _obs_from_state(self, state: dict[str, Any]) -> dict[str, torch.Tensor]:
        flat = np.concatenate(
            [
                np.asarray(state["pose"], dtype=np.float32),
                np.asarray(state["vel"], dtype=np.float32),
                np.asarray([state["gripper_pos"]], dtype=np.float32),
                np.asarray(state["force"], dtype=np.float32),
                np.asarray(state["torque"], dtype=np.float32),
            ]
        )
        obs: dict[str, torch.Tensor] = {
            "state": torch.as_tensor(flat, device=self.device).unsqueeze(0)
        }
        if self.camera_capture is not None:
            frames = self.camera_capture()
            for key in self.cfg.camera_keys:
                image = np.asarray(frames[key], dtype=np.uint8)
                obs[key] = torch.as_tensor(image, device=self.device).unsqueeze(0)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del seed, options
        self.bridge.reset_joints()
        state = self.bridge.get_state()
        self._t = 0
        self._last_pose = np.asarray(state["pose"], dtype=np.float64)
        return self._obs_from_state(state), {}

    def step(self, action: torch.Tensor):
        if self._last_pose is None:
            raise RuntimeError("FrankaRealEnv.step() called before reset().")

        action_np = (
            action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        ).reshape(-1)
        assert action_np.shape == (7,), (
            f"FrankaRealEnv is a batch-of-1 env; expected a (1, 7) or (7,) action, "
            f"got array of shape {action_np.shape}."
        )

        pos_scale, rot_scale = self.cfg.action_scale
        delta_pos = action_np[:3].astype(np.float64) * pos_scale
        delta_rotvec = action_np[3:6].astype(np.float64) * rot_scale
        gripper_action = float(action_np[6])

        target_pos = np.clip(
            self._last_pose[:3] + delta_pos, self._safety_low, self._safety_high
        )
        current_rot = Rotation.from_quat(self._last_pose[3:7])
        target_rot = Rotation.from_rotvec(delta_rotvec) * current_rot
        target_pose = np.concatenate([target_pos, target_rot.as_quat()])

        self.bridge.send_gripper(gripper_action > self.cfg.gripper_threshold)
        self.bridge.send_pose(target_pose)

        state = self.bridge.get_state()
        self._last_pose = np.asarray(state["pose"], dtype=np.float64)
        self._t += 1

        obs = self._obs_from_state(state)
        # Task reward is intentionally not computed here -- see
        # rl_garden/envs/wrappers/reward_classifier.py, which wraps this env
        # and replaces this placeholder with a learned success signal.
        reward = torch.zeros(1, device=self.device)
        terminated = torch.zeros(1, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(
            [self._t >= self.cfg.max_episode_steps], dtype=torch.bool, device=self.device
        )
        return obs, reward, terminated, truncated, {}


def make_franka_real_env(
    cfg: FrankaRealEnvConfig,
    bridge_client: Optional[FrankaBridgeClient] = None,
    camera_capture: Optional[CameraCapture] = None,
) -> gym.Env:
    env: gym.Env = FrankaRealEnv(cfg, bridge_client=bridge_client, camera_capture=camera_capture)
    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper

        env = RewardScaleBiasWrapper(env, cfg.reward_scale, cfg.reward_bias)
    return env
