"""ManiSkill env backend — registered as ``"maniskill"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)
class ManiSkillBackend(EnvBackend):
    config_field = "maniskill"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.maniskill import ManiSkillEnvConfig

        ms = req.backend_config  # ManiSkillConfig or None
        sim_backend = ms.sim_backend if ms is not None else "gpu"
        render_backend = ms.render_backend if ms is not None else "gpu"
        reward_mode = ms.reward_mode if ms is not None else None
        success_reward_override = ms.success_reward_override if ms is not None else None
        return ManiSkillEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            obs_mode=req.obs_mode,
            include_state=req.include_state,
            control_mode=req.control_mode,
            camera_width=req.camera_width,
            camera_height=req.camera_height,
            render_mode=req.render_mode,
            per_camera_rgbd=req.per_camera_rgbd,
            frame_stack=req.frame_stack,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            sim_backend=sim_backend,
            render_backend=render_backend,
            reward_mode=reward_mode,
            success_reward_override=success_reward_override,
            reconfiguration_freq=1 if is_eval else 0,
            record_dir=req.eval_record_dir if is_eval else None,
            save_video=req.capture_video if is_eval else False,
            video_fps=req.video_fps if is_eval else 30,
            max_steps_per_video=req.num_eval_steps if is_eval else None,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.maniskill import make_maniskill_env

        cfg = cls._make_cfg(req, is_eval=False)
        env = make_maniskill_env(cfg)
        ms = req.backend_config
        if ms is not None and ms.success_reward_override is not None:
            if env.unwrapped.reward_mode != "normalized_dense":
                raise ValueError(
                    "--maniskill.success_reward_override requires "
                    "reward_mode='normalized_dense'; "
                    f"got {env.unwrapped.reward_mode!r}"
                )
        return env

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.maniskill import make_maniskill_env

        return make_maniskill_env(cls._make_cfg(req, is_eval=True))


register_env_backend("maniskill", ManiSkillBackend)
