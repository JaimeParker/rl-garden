"""RoboTwin env backend — registered as ``"robotwin"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class RoboTwinBackend(EnvBackend):
    config_field = "robotwin"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.robotwin.config import RoboTwinEnvConfig

        rt = req.backend_config  # RoboTwinConfig or None
        iw = rt.include_wrist_cameras if rt is not None else True
        head_cam = rt.head_camera_type if rt is not None else "D435"
        wrist_cam = rt.wrist_camera_type if rt is not None else "D435"
        random_background = rt.random_background if rt is not None else True
        cluttered_table = rt.cluttered_table if rt is not None else True
        clean_background_rate = rt.clean_background_rate if rt is not None else 0.02
        random_head_camera_dis = rt.random_head_camera_dis if rt is not None else 0.0
        random_table_height = rt.random_table_height if rt is not None else 0.03
        random_light = rt.random_light if rt is not None else False
        crazy_light = rt.crazy_random_light_rate if rt is not None else 0.0
        render_every = rt.render_every_control_step if rt is not None else False
        step_cap = rt.control_step_cap if rt is not None else None
        step_lim = rt.step_lim if rt is not None else 400
        planner = rt.planner_backend if rt is not None else "mplib"
        embodiment = rt.embodiment if rt is not None else ["aloha-agilex"]
        agent_image_size = rt.agent_image_size if rt is not None else None
        image_resize_backend = rt.image_resize_backend if rt is not None else "pillow"

        # height first — matches (camera_height, camera_width) convention in both scripts
        image_size = (req.camera_height or 64, req.camera_width or 64)

        task_cfg: dict = {
            "task_name": req.env_id,
            "step_lim": step_lim,
            "planner_backend": planner,
            "embodiment": embodiment,
            "render_freq": 0,
            "render_every_control_step": render_every,
            "episode_num": 100,
            "use_seed": False,
            "save_freq": 15,
            "camera": {
                "head_camera_type": head_cam,
                "wrist_camera_type": wrist_cam,
                "collect_head_camera": True,
                "collect_wrist_camera": iw,
            },
            "domain_randomization": {
                "random_background": random_background,
                "cluttered_table": cluttered_table,
                "clean_background_rate": clean_background_rate,
                "random_head_camera_dis": random_head_camera_dis,
                "random_table_height": random_table_height,
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

        control_mode = req.control_mode or "delta_joint_pos"
        action_dim = 14

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
            control_mode=control_mode,  # type: ignore[arg-type]
            action_dim=action_dim,
            joint_delta_scale=rt.joint_delta_scale if rt is not None else 0.05,
            gripper_delta_scale=rt.gripper_delta_scale if rt is not None else 0.2,
            ee_delta_pos_scale=rt.ee_delta_pos_scale if rt is not None else 0.03,
            ee_delta_rot_scale=rt.ee_delta_rot_scale if rt is not None else 0.15,
            profile_timing=rt.profile_timing if rt is not None else False,
            profile_interval=rt.profile_interval if rt is not None else 100,
            render_every_control_step=render_every,
            control_step_cap=step_cap,
            random_background=random_background,
            cluttered_table=cluttered_table,
            clean_background_rate=clean_background_rate,
            random_head_camera_dis=random_head_camera_dis,
            random_table_height=random_table_height,
            random_light=random_light,
            crazy_random_light_rate=crazy_light,
            head_camera_type=head_cam,
            wrist_camera_type=wrist_cam,
            image_size=image_size,
            agent_image_size=(agent_image_size, agent_image_size)
            if agent_image_size is not None
            else None,
            image_resize_backend=image_resize_backend,
            include_wrist_cameras=iw,
            auto_reset=True,
            ignore_terminations=False,
            device=rt.device if rt is not None else "auto",
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.robotwin.env import make_robotwin_env

        return make_robotwin_env(cls._make_cfg(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.robotwin.env import make_robotwin_env

        return make_robotwin_env(cls._make_cfg(req, is_eval=True))


register_env_backend("robotwin", RoboTwinBackend)
