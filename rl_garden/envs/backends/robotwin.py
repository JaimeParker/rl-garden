"""RoboTwin env backend — registered as ``"robotwin"``."""

from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)

_ROBOTWIN_CAMERAS = ("head", "left_wrist", "right_wrist")


class RoboTwinBackend(EnvBackend):
    api_version = 2
    config_field = "robotwin"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.robotwin.config import RoboTwinEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        obs = req.observation
        if obs.depth:
            raise ObservationContractError(
                f"robotwin backend has no depth camera output; requested "
                f"depth={obs.depth!r}"
            )
        unknown = set(obs.rgb) - set(_ROBOTWIN_CAMERAS)
        if unknown:
            raise ObservationContractError(
                f"robotwin: unknown camera(s) {sorted(unknown)!r}; available "
                f"cameras: {list(_ROBOTWIN_CAMERAS)}"
            )
        if obs.frame_stack > 1 and not obs.rgb:
            raise ObservationContractError(
                "robotwin: frame_stack > 1 requires at least one rgb camera"
            )
        if obs.extra_state:
            raise ObservationContractError("robotwin has no extra state sources")
        collect_wrist = bool({"left_wrist", "right_wrist"} & set(obs.rgb))

        rt = req.backend_config  # RoboTwinConfig or None
        head_cam = rt.head_camera_type if rt is not None else "D435"
        wrist_cam = rt.wrist_camera_type if rt is not None else "D435"
        random_light = rt.random_light if rt is not None else False
        crazy_light = rt.crazy_random_light_rate if rt is not None else 0.0
        render_every = rt.render_every_control_step if rt is not None else False
        step_cap = rt.control_step_cap if rt is not None else None
        step_lim = rt.step_lim if rt is not None else 400
        planner = rt.planner_backend if rt is not None else "mplib"
        embodiment = rt.embodiment if rt is not None else ["aloha-agilex"]
        disable_topp = rt.disable_topp if rt is not None else False

        # (H, W) order -- matches ObservationConfig.image_size's own convention.
        image_size = obs.image_size if obs.image_size is not None else (64, 64)

        task_cfg: dict = {
            "task_name": req.env_id,
            "step_lim": step_lim,
            "planner_backend": planner,
            "embodiment": embodiment,
            "render_freq": 0,
            "render_every_control_step": render_every,
            "need_topp": not disable_topp,
            "episode_num": 100,
            "use_seed": False,
            "save_freq": 15,
            "camera": {
                "head_camera_type": head_cam,
                "wrist_camera_type": wrist_cam,
                "collect_head_camera": "head" in obs.rgb,
                "collect_wrist_camera": collect_wrist,
            },
            "domain_randomization": {
                "random_background": True,
                "cluttered_table": True,
                "clean_background_rate": 0.02,
                "random_head_camera_dis": 0,
                "random_table_height": 0.03,
                "random_light": random_light,
                "crazy_random_light_rate": crazy_light,
            },
            "data_type": {"rgb": True, "qpos": True},
            "save_path": "./data",
            "collect_data": False,
            "eval_video_log": bool(is_eval and req.capture_video),
        }
        if step_cap is not None:
            task_cfg["control_step_cap"] = step_cap
        if is_eval and req.capture_video and req.eval_record_dir:
            task_cfg["eval_video_save_dir"] = req.eval_record_dir

        return RoboTwinEnvConfig(
            task_name=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            robotwin_root=rt.robotwin_root if rt is not None else None,
            assets_path=rt.assets_path if rt is not None else None,
            seeds_path=rt.seeds_path if rt is not None else None,
            step_lim=step_lim,
            max_episode_steps=step_lim,
            task_config=task_cfg,
            planner_backend=planner,
            embodiment=embodiment,
            reward_mode=rt.reward_mode if rt is not None else "dense",  # type: ignore[arg-type]
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            control_mode=req.control_mode or "delta_joint_pos",  # type: ignore[arg-type]
            joint_delta_scale=rt.joint_delta_scale if rt is not None else 0.05,
            gripper_delta_scale=rt.gripper_delta_scale if rt is not None else 0.2,
            ee_delta_pos_scale=rt.ee_delta_pos_scale if rt is not None else 0.03,
            ee_delta_rot_scale=rt.ee_delta_rot_scale if rt is not None else 0.15,
            profile_timing=rt.profile_timing if rt is not None else False,
            profile_interval=rt.profile_interval if rt is not None else 100,
            render_every_control_step=render_every,
            control_step_cap=step_cap,
            random_light=random_light,
            crazy_random_light_rate=crazy_light,
            head_camera_type=head_cam,
            wrist_camera_type=wrist_cam,
            image_size=image_size,
            rgb_cameras=obs.rgb,
            state=obs.state,
            frame_stack=obs.frame_stack,
            auto_reset=True,
            ignore_terminations=False,
            device=rt.device if rt is not None else "auto",
        )

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.robotwin.env import make_robotwin_env

        return make_robotwin_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.robotwin.env import make_robotwin_env

        return make_robotwin_env(cls.resolve_config(req, is_eval=True))


register_env_backend("robotwin", RoboTwinBackend)
