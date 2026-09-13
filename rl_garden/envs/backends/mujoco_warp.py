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
    api_version = 2
    config_field = "mujoco_warp"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.mujoco_warp.config import MujocoWarpEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        # v1 supports exactly one camera (CustomMujocoWarpEnv.CAMERA_NAME =
        # "main"); see that module's docstring.
        obs = req.observation
        if not obs.state:
            raise ObservationContractError(
                "mujoco_warp: state=False is not supported -- every "
                "mujoco_warp task emits a state observation unconditionally "
                "(there is no per-task hook to omit it)."
            )
        unknown = (set(obs.rgb) | set(obs.depth)) - {"main"}
        if unknown:
            raise ObservationContractError(
                f"mujoco_warp: unknown camera(s) {sorted(unknown)!r}; available "
                "cameras: ['main']"
            )
        if obs.extra_state:
            raise ObservationContractError("mujoco_warp has no extra state sources")
        render_rgb = "main" in obs.rgb
        render_depth = "main" in obs.depth
        if obs.image_size is not None:
            render_height, render_width = obs.image_size
        elif render_rgb or render_depth:
            render_height = render_width = 64
        else:
            render_height = render_width = None

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
            render_width=render_width,
            render_height=render_height,
            render_rgb=render_rgb,
            render_depth=render_depth,
            frame_stack=obs.frame_stack,
            env_kwargs=env_kwargs,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
        )

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco_warp import make_mujoco_warp_env

        return make_mujoco_warp_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco_warp import make_mujoco_warp_env

        return make_mujoco_warp_env(cls.resolve_config(req, is_eval=True))


register_env_backend("mujoco_warp", MujocoWarpBackend)
