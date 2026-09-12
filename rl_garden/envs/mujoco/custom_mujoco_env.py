"""rl-garden's custom MuJoCo task authoring base class.

Subclasses ``gymnasium.envs.mujoco.MujocoEnv`` (which already bridges the raw
mujoco Python API -- MjModel/MjData, ``mj_step``, renderer, action space
derived from ``actuator_ctrlrange`` -- to the standard ``gym.Env`` shape) and
decomposes its monolithic ``step()``/``reset_model()`` into rl-garden's
4-method task-authoring contract: subclass this and implement
``_apply_action``, ``_get_obs``, ``_get_reward``, ``_get_terminated``.

Vectorization: a ``CustomMujocoEnv`` subclass is always a SINGLE environment
(``MjModel``/``MjData`` has no native batching). N-way parallelism is handled
by wrapping N instances in ``gymnasium.vector.AsyncVectorEnv`` (one OS
process per env -- see ``rl_garden.envs.mujoco.env.make_mujoco_env``), not by
this class.

Observation key convention: ``_get_obs()`` should return rl-garden's own
cross-backend keys -- ``"state"`` for the flat proprioceptive tensor,
``"rgb_<camera>"``/``"depth_<camera>"`` for images, always named even for a
single camera (the observation contract has no bare rgb/depth keys),
matching the convention already used for the IsaacLab custom-task
scaffold. Since ``_get_obs()`` always returns a dict,
the ``observation_space`` passed to ``__init__`` must be a matching
``gym.spaces.Dict`` -- not a bare ``Box`` -- even for a single ``"state"``
key. Unlike some other rl-garden backends where the declared space is only a
placeholder, ``gymnasium.vector.AsyncVectorEnv``'s shared-memory transport
sizes and types its buffers directly from this declaration, so a mismatch
crashes at the first ``reset()``/``step()`` under ``vectorization="async"``
(see ``rl_garden.envs.mujoco.tasks.inverted_pendulum_custom`` for the
worked example).

Camera rendering: use ``_render_cameras()``, built from the ``camera_configs``
passed to ``__init__``. Each entry gets its own
``gymnasium.envs.mujoco.mujoco_rendering.MujocoRenderer`` instance -- a
``MujocoRenderer`` is bound to exactly one camera and one resolution at
construction, so heterogeneous cameras need separate renderer instances.
``MujocoEnv``'s own ``self.mujoco_renderer`` (built from the
``render_mode``/``camera_id``/... constructor kwargs) is unrelated: it serves
the generic ``env.render()`` gym API for human viewing/video capture, not
observations.

RGB is raw uint8 ``[0, 255]`` (no normalization -- matches
``rl_garden.encoders.base.image_needs_normalization``'s convention). Depth is
whatever ``mujoco.mjr_readPixels``'s raw depth buffer contains: nonlinear
normalized device coordinates, NOT linear/metric depth -- confirmed both by
reading Gymnasium's ``OffScreenViewer.render()`` source
(``gymnasium/envs/mujoco/mujoco_rendering.py``, no znear/zfar linearization
applied) and empirically (``rl_garden.envs.mujoco.tasks.
inverted_pendulum_custom.InvertedPendulumCameraCustomEnv``, a free-camera
64x64 render, returned values in roughly ``[0.987, 1.0]`` float32 -- the
compressed-near-1.0 range expected of raw NDC depth for a mostly-background
scene, not metric distance). If a task needs linear depth, apply the
znear/zfar conversion yourself; do not assume this output is already linear.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


class CustomMujocoEnv(MujocoEnv):
    """Base class for hand-authored MuJoCo tasks.

    Subclass this and implement the four task-specific methods:
    ``_apply_action``, ``_get_obs``, ``_get_reward``, ``_get_terminated``.
    Everything else (MjModel/MjData init, physics stepping, rendering,
    action space) comes from ``MujocoEnv``.
    """

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: gym.Space,
        camera_configs: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        self._camera_configs = camera_configs or []
        super().__init__(model_path, frame_skip, observation_space, **kwargs)
        self._camera_renderers = {
            self._camera_key(cfg): MujocoRenderer(
                self.model,
                self.data,
                width=cfg["width"],
                height=cfg["height"],
                camera_id=cfg.get("camera_id"),
                camera_name=cfg.get("camera_name"),
            )
            for cfg in self._camera_configs
        }

    @staticmethod
    def _camera_key(cfg: dict[str, Any]) -> str:
        """Observation-key name for this camera. ``"key"`` overrides when the
        camera has no MuJoCo ``camera_name`` (e.g. a free/unnamed camera
        selected by ``camera_id``) but still needs a stable, descriptive
        observation key."""
        key = cfg.get("key", cfg.get("camera_name", cfg.get("camera_id")))
        if key is None:
            raise ValueError(
                "camera_configs entries need 'key', 'camera_name', or 'camera_id'"
            )
        return str(key)

    def step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        reward = self._get_reward(action)
        terminated = self._get_terminated()
        return obs, reward, terminated, False, {}  # truncation via registered TimeLimit

    def reset_model(self):
        qpos, qvel = self._sample_reset_state(self.init_qpos.copy(), self.init_qvel.copy())
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _sample_reset_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """Optional hook for reset-time randomization. Default: no-op."""
        return qpos, qvel

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Renders every configured camera into rl-garden's
        ``rgb_<cam>``/``depth_<cam>`` key convention (always named, even for a
        single camera -- the observation contract has no bare ``rgb``/
        ``depth`` keys). Call this from your own ``_get_obs()``; it is not
        invoked automatically."""
        out: dict[str, np.ndarray] = {}
        for cfg in self._camera_configs:
            key = self._camera_key(cfg)
            renderer = self._camera_renderers[key]
            if cfg.get("enable_rgb", True):
                out[f"rgb_{key}"] = renderer.render("rgb_array")
            if cfg.get("enable_depth", False):
                # Contract requires depth as (H, W, 1); MujocoRenderer returns
                # a bare (H, W) buffer.
                out[f"depth_{key}"] = renderer.render("depth_array")[..., None].astype(
                    np.float32
                )
        return out

    def close(self) -> None:
        for renderer in self._camera_renderers.values():
            renderer.close()
        super().close()

    # --- task authors implement these four ---

    def _apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def _get_obs(self) -> dict[str, Any]:
        raise NotImplementedError

    def _get_reward(self, action: np.ndarray) -> float:
        raise NotImplementedError

    def _get_terminated(self) -> bool:
        raise NotImplementedError
