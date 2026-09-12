"""OGBench env backend — registered as ``"ogbench"``.

Covers all 8 OGBench task families (locomotion: pointmaze/antmaze/
humanoidmaze/antsoccer; manipulation: cube/scene/puzzle), state and pixel
observations; see ``rl_garden.envs.ogbench.env`` module docstring for the
sync/async vectorization choice and pixel-obs scoping.
"""

from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class OGBenchBackend(EnvBackend):
    api_version = 2
    config_field = "ogbench"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.ogbench.config import OGBenchEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        og = req.backend_config  # OGBenchConfig or None
        env_kwargs = (
            json.loads(og.env_kwargs_json)
            if og is not None and og.env_kwargs_json
            else {}
        )
        obs = req.observation
        is_visual_id = req.env_id.startswith("visual-")
        if is_visual_id:
            # OGBench's own API exposes no camera name -- the offline dataset
            # loader (rl_garden.buffers.ogbench_dataset) fixes the single
            # pixel key as "rgb_ogbench"; this backend must emit the same key
            # so online/offline observation spaces agree.
            if obs.rgb != ("ogbench",) or obs.depth or obs.state:
                raise ObservationContractError(
                    f"ogbench: visual-* env_id={req.env_id!r} requires exactly "
                    "rgb=('ogbench',) and no depth/state (pixel-only observation, "
                    "matching the offline dataset loader's fixed key), got "
                    f"rgb={obs.rgb!r} depth={obs.depth!r} state={obs.state!r}"
                )
            pixel_camera = obs.rgb[0]
        else:
            if obs.rgb or obs.depth:
                raise ObservationContractError(
                    f"ogbench: env_id={req.env_id!r} is not a visual-* env id; cannot "
                    f"honor rgb={obs.rgb!r} depth={obs.depth!r}"
                )
            if obs.frame_stack > 1:
                raise ObservationContractError(
                    f"ogbench: env_id={req.env_id!r} is not a visual-* env id; "
                    f"frame_stack={obs.frame_stack} requires a pixel observation "
                    "(a state-only observation cannot be frame-stacked)."
                )
            pixel_camera = None
        if obs.image_size is not None:
            raise ObservationContractError(
                f"ogbench backend cannot set image_size={obs.image_size!r}; "
                "pixel resolution is fixed by env_id"
            )
        return OGBenchEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=og.device if og is not None else "cpu",
            pixel_camera=pixel_camera,
            frame_stack=obs.frame_stack,
            env_kwargs=env_kwargs,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            vectorization=og.vectorization if og is not None else "sync",
        )

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.ogbench import make_ogbench_env

        return make_ogbench_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.ogbench import make_ogbench_env

        return make_ogbench_env(cls.resolve_config(req, is_eval=True))


register_env_backend("ogbench", OGBenchBackend)
