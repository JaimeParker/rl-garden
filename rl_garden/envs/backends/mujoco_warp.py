"""mujoco_warp GPU env backend — registered as ``"mujoco_warp"``.

Only rl-garden's own custom tasks (``CustomMujocoWarpEnv`` subclasses,
registered via ``register_mujoco_warp_task``) are supported — ``mujoco_warp``
has no bundled benchmark task suite the way Gymnasium's ``envs.mujoco`` does.
See ``rl_garden.envs.mujoco_warp.env``/``custom_mujoco_warp_env`` module
docstrings for the native-batched adapter-free design and the SAME_STEP-style
``final_observation`` contract.
"""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class MujocoWarpBackend(EnvBackend):
    config_field = "mujoco_warp"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.mujoco_warp import MujocoWarpEnvConfig

        mjw_cfg = req.backend_config  # MujocoWarpConfig or None
        env_kwargs = (
            json.loads(mjw_cfg.env_kwargs_json)
            if mjw_cfg is not None and mjw_cfg.env_kwargs_json
            else {}
        )
        return MujocoWarpEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=mjw_cfg.device if mjw_cfg is not None else "cuda:0",
            camera_width=req.camera_width,
            camera_height=req.camera_height,
            render_rgb=req.obs_mode != "state",
            render_depth=req.obs_mode == "rgbd",
            env_kwargs=env_kwargs,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco_warp import make_mujoco_warp_env

        return make_mujoco_warp_env(cls._make_cfg(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco_warp import make_mujoco_warp_env

        return make_mujoco_warp_env(cls._make_cfg(req, is_eval=True))


register_env_backend("mujoco_warp", MujocoWarpBackend)
