"""dm_control env backend — registered as ``"dm_control"``.

``--env_id`` names the task as ``<domain>-<task>`` (NCRL TASK_SET
naming, e.g. ``walker-walk``, ``cheetah-run-backwards``, ``walker-walk-hard``;
see ``rl_garden.envs.dm_control.tasks``). Supports a flat ``"state"``
observation and/or one pixel camera ``rgb_default``; see
``rl_garden.envs.dm_control.env`` for the episode/action-repeat semantics.
``dm_control`` is only imported when an env is constructed.
"""

from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class DMControlBackend(EnvBackend):
    api_version = 2
    config_field = "dm_control"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.dm_control.config import DMControlEnvConfig
        from rl_garden.envs.dm_control.env import validate_cameras
        from rl_garden.envs.dm_control.tasks import parse_task_id
        from rl_garden.observations.schema import ObservationContractError

        dmc = req.backend_config  # DMControlConfig or None
        obs = req.observation
        if obs.depth:
            raise ObservationContractError(
                f"dm_control: depth cameras {obs.depth!r} are not supported; request rgb only."
            )
        if obs.extra_state:
            raise ObservationContractError("dm_control has no extra state sources")
        if obs.frame_stack > 1 and not obs.rgb:
            raise ObservationContractError(
                f"dm_control: frame_stack={obs.frame_stack} requires an rgb camera; "
                "a state-only observation cannot be frame-stacked."
            )
        validate_cameras(obs.rgb)
        _, _, hard_suffix = parse_task_id(req.env_id)
        return DMControlEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=dmc.device if dmc is not None else "cpu",
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            vectorization=dmc.vectorization if dmc is not None else "sync",
            action_repeat=dmc.action_repeat if dmc is not None else 2,
            hard=hard_suffix or (dmc.hard if dmc is not None else False),
            camera_id=dmc.camera_id if dmc is not None else None,
            state=obs.state,
            rgb_cameras=obs.rgb,
            image_size=obs.image_size if obs.image_size is not None else (64, 64),
            frame_stack=obs.frame_stack,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.dm_control import make_dm_control_env

        return make_dm_control_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.dm_control import make_dm_control_env

        return make_dm_control_env(cls.resolve_config(req, is_eval=True))


register_env_backend("dm_control", DMControlBackend)
