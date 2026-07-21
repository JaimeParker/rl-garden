"""rl-garden's custom mujoco_warp GPU task authoring base class.

Subclass this and implement ``_apply_action``/``_get_obs``/``_get_reward``/
``_get_terminated``/``_get_truncated``/``_reset_idx``. Physics is driven by
``mujoco_warp`` (``mjw.put_model``/``make_data``/``step``) directly on the
GPU. Unlike the CPU ``CustomMujocoEnv`` scaffold, ``mujoco_warp``'s ``Data``
is natively batched (one ``Data`` object holds all ``nworld`` envs at once),
so this class *is* the final rl-garden env shape itself -- there is no
separate adapter layer and no ``gymnasium.vector`` wrapping.

Termination contract: mirrors ``ManiSkillVectorEnv.step()``
(``mani_skill/vector/wrappers/gymnasium.py``), not the CPU MuJoCo backend's
Gymnasium-autoreset bridging -- there is no Gymnasium vector env underneath
to bridge from. This class subclasses ``gymnasium.vector.VectorEnv``
directly (matching ``ManiSkillVectorEnv``'s own precedent), not ``gym.Env``
-- it already produces ``(nworld, ...)``-shaped torch tensors natively, so
there is nothing for an adapter to translate. On the step where an env
terminates/truncates, this class clones the true terminal observation into
``infos["final_observation"]``, resets that env's state in place, and
returns the *post-reset* observation from ``step()`` -- written directly in
rl-garden's own convention from the start, no key-name translation needed.

**Required initialization order (verified empirically, not obvious from the
API surface):** ``mjw.make_data()`` only allocates GPU buffers -- it does
NOT run forward kinematics. ``xpos``/``xmat``/etc (and therefore rendering
and any observation reading world-frame quantities) are left at their
zero-initialized default until ``mjw.forward(model, data)`` is called
explicitly. This class calls it once in ``__init__`` after ``make_data()``,
and again in ``_reset_idx()`` after writing ``qpos``/``qvel`` for the reset
envs (mirrors Gymnasium's own ``MujocoEnv.set_state()``, which calls
``mujoco.mj_forward`` right after writing state for the same reason). Omitting
this call doesn't raise -- it silently produces degenerate all-zero-ish
observations/renders, so don't skip it when overriding ``_reset_idx``.

Camera rendering: v1 supports exactly one camera (no multi-camera
``rgb_<cam>``/``depth_<cam>`` support yet -- ``mujoco_warp``'s render buffers
pack multiple cameras via ``rgb_adr``/``depth_adr`` offset arrays, which
hasn't been exercised/verified; extend ``_render_cameras`` once that's
actually tested, don't guess at the slicing). The model must define an
explicit ``<camera>`` element -- unlike the CPU path's ``MujocoRenderer``,
there is no automatic free/"track" camera fallback; a camera-less model
renders pure background with no error.

RGB: ``rc.rgb_data`` is a flat packed ``0xRRGGBB`` ``uint32`` array (not
``wp.vec3f`` -- confirmed by reading ``mujoco_warp/_src/render_test.py``'s
``_unpack_rgb`` helper after an earlier wrong-dtype guess crashed with a real
CUDA illegal memory access). PyTorch doesn't support bitwise ops on
``uint32`` CUDA tensors, so unpacking upcasts to ``int64`` first. Output is
raw uint8 ``[0, 255]`` per channel, matching
``rl_garden.encoders.base.image_needs_normalization``'s convention.

Depth: ``rc.depth_data`` is "planar depth" (distance along the camera's
viewing axis, confirmed against ``mujoco.Renderer`` with
``enable_depth_rendering()`` via ``render_test.py``'s
``test_depth_matches_mujoco``) -- ``0.0`` means no geometry hit (background),
positive values are real distances in model units. This is a genuinely
different, more directly usable convention than the CPU MuJoCo backend's raw
NDC depth (see ``rl_garden.envs.mujoco.custom_mujoco_env`` module docstring)
-- do not assume the two backends' depth outputs mean the same thing.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


def _unpack_rgb(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.to(torch.int64)
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return torch.stack([r, g, b], dim=-1).to(torch.uint8)


def _clone_obs(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: value.clone() for key, value in obs.items()}
    return obs.clone()


class CustomMujocoWarpEnv(VectorEnv):
    """Base class for hand-authored mujoco_warp GPU tasks. See module
    docstring for the full contract."""

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(
        self,
        model_path: str,
        nworld: int,
        device: str,
        observation_space: gym.Space,
        frame_skip: int = 1,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        render_rgb: bool = True,
        render_depth: bool = False,
        **kwargs: Any,
    ) -> None:
        self._mjm = mujoco.MjModel.from_xml_path(model_path)
        self.model = mjw.put_model(self._mjm)
        self.data = mjw.make_data(self._mjm, nworld=nworld)
        mjw.forward(self.model, self.data)  # see module docstring: required, not automatic.
        # mjw.step() advances exactly one physics timestep per call, unlike
        # CPU MujocoEnv.do_simulation(action, frame_skip) which internally
        # loops mj_step for frame_skip substeps -- step() below loops
        # mjw.step() the same number of times so an "env step" advances the
        # same amount of simulated time per action as the CPU scaffold.
        self.frame_skip = frame_skip

        self.num_envs = nworld
        self.device = torch.device(device)
        self.single_observation_space = observation_space
        self.observation_space = batch_space(self.single_observation_space, nworld)
        self.single_action_space = self._make_action_space()
        # off_policy.py's random-exploration phase reads the batched
        # action_space.shape directly (not single_action_space).
        low = np.tile(self.single_action_space.low, (nworld, 1))
        high = np.tile(self.single_action_space.high, (nworld, 1))
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._has_camera = camera_width is not None
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._render_rgb = render_rgb
        self._render_depth = render_depth
        self._render_ctx = (
            mjw.create_render_context(
                self._mjm,
                nworld=nworld,
                cam_res=(camera_width, camera_height),
                render_rgb=render_rgb,
                render_depth=render_depth,
            )
            if self._has_camera
            else None
        )

    def _make_action_space(self) -> gym.spaces.Box:
        # Same derivation as gymnasium.envs.mujoco.MujocoEnv._set_action_space.
        bounds = self._mjm.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        env_ids = options.get("env_idx") if options else None
        self._reset_idx(env_ids)
        return self._get_obs(), {}

    def step(self, actions: torch.Tensor):
        self._apply_action(actions)  # sets ctrl once; held constant across substeps below.
        for _ in range(self.frame_skip):
            mjw.step(self.model, self.data)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        done_mask = terminated | truncated

        infos: dict[str, Any] = {}
        if done_mask.any():
            infos["final_observation"] = _clone_obs(obs)
            infos["_final_observation"] = done_mask
            infos["_final_info"] = done_mask
            env_ids = torch.nonzero(done_mask, as_tuple=True)[0]
            self._reset_idx(env_ids)
            obs = self._get_obs()

        return obs, reward, terminated, truncated, infos

    def _render_cameras(self) -> dict[str, torch.Tensor]:
        """Renders the (single, v1-only) configured camera into rl-garden's
        ``rgb``/``depth`` key convention. Call this from your own
        ``_get_obs()``; it is not invoked automatically."""
        if self._render_ctx is None:
            return {}
        mjw.render(self.model, self.data, self._render_ctx)
        out: dict[str, torch.Tensor] = {}
        # rgb_data/depth_data are only allocated (non-empty) when the
        # corresponding render_rgb/render_depth flag was set at
        # create_render_context() time -- reading the disabled one gives a
        # zero-size buffer that fails to reshape, not a zero-filled one.
        if self._render_rgb:
            rgb_flat = wp.to_torch(self._render_ctx.rgb_data)
            out_shape_rgb = (self.num_envs, self._camera_height, self._camera_width, 3)
            out["rgb"] = _unpack_rgb(rgb_flat).reshape(out_shape_rgb)
        if self._render_depth:
            depth_flat = wp.to_torch(self._render_ctx.depth_data)
            out["depth"] = depth_flat.reshape(self.num_envs, self._camera_height, self._camera_width)
        return out

    def close(self) -> None:
        pass

    # --- task authors implement these six ---

    def _apply_action(self, actions: torch.Tensor) -> None:
        raise NotImplementedError

    def _get_obs(self) -> dict[str, Any]:
        raise NotImplementedError

    def _get_reward(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_terminated(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_truncated(self) -> torch.Tensor:
        raise NotImplementedError

    def _reset_idx(self, env_ids) -> None:
        raise NotImplementedError
