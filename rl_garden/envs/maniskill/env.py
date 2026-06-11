"""ManiSkill environment factory.

Wraps a ManiSkill env with the same stack ManiSkill baselines use:

    gym.make -> [FlattenRGBDObservationWrapper] -> [FlattenActionSpaceWrapper]
              -> [RecordEpisode] -> ManiSkillVectorEnv

The resulting env exposes GPU torch tensors from ``reset`` / ``step`` and is
the only env shape the rest of ``rl_garden`` targets.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_garden.envs.maniskill.config import ManiSkillEnvConfig


def _is_visual_obs_mode(mode: str) -> bool:
    return any(part in ("rgb", "rgbd", "depth") for part in mode.split("+"))


def make_maniskill_env(cfg: ManiSkillEnvConfig):
    """Build a ManiSkill vectorized env according to ``cfg``.

    Returns the wrapped ``ManiSkillVectorEnv`` instance.
    """
    # Lazy imports so the package doesn't hard-depend on mani_skill being installed.
    import mani_skill.envs  # noqa: F401  (registers envs)
    from rl_garden.envs import register_custom_envs
    from mani_skill.utils.wrappers.flatten import (
        FlattenActionSpaceWrapper,
        FlattenRGBDObservationWrapper,
    )
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env_kwargs: dict[str, Any] = dict(
        obs_mode=cfg.obs_mode,
        render_mode=cfg.render_mode,
        sim_backend=cfg.sim_backend,
        render_backend=cfg.render_backend,
        sensor_configs=dict(),
    )
    register_custom_envs()
    if cfg.control_mode is not None:
        env_kwargs["control_mode"] = cfg.control_mode
    if cfg.reward_mode is not None:
        env_kwargs["reward_mode"] = cfg.reward_mode
    if cfg.robot_uids is not None:
        env_kwargs["robot_uids"] = cfg.robot_uids
    if cfg.fix_peg_pose is not None:
        env_kwargs["fix_peg_pose"] = cfg.fix_peg_pose
    if cfg.fix_box is not None:
        env_kwargs["fix_box"] = cfg.fix_box
    if cfg.fixed_peg_xy is not None:
        env_kwargs["fixed_peg_xy"] = cfg.fixed_peg_xy
    if cfg.fixed_peg_z_rot_deg is not None:
        env_kwargs["fixed_peg_z_rot_deg"] = cfg.fixed_peg_z_rot_deg
    if cfg.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = cfg.camera_width
    if cfg.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = cfg.camera_height
    if cfg.human_render_camera_configs is not None:
        env_kwargs["human_render_camera_configs"] = cfg.human_render_camera_configs
    env_kwargs.update(cfg.env_kwargs)

    env = gym.make(
        cfg.env_id,
        num_envs=cfg.num_envs,
        reconfiguration_freq=cfg.reconfiguration_freq,
        **env_kwargs,
    )

    # RGBD dict -> flat {rgb(/depth), state} dict
    # FIXME(depth-obs-mode): `"depth" in cfg.obs_mode` is a substring test, so
    # obs_mode="rgbd" yields False ("depth" is not a substring of "rgbd") and
    # depth is silently dropped -- yet image_keys_from_obs_mode("rgbd") returns
    # ("rgb", "depth"). The two disagree on what "rgbd" means. Decide the rgbd
    # contract and make env factory + image_keys helper consistent. (Pre-existing;
    # out of scope for the encoder-registry change.)
    if _is_visual_obs_mode(cfg.obs_mode):
        if cfg.per_camera_rgbd:
            from rl_garden.envs.wrappers import PerCameraRGBDWrapper

            env = PerCameraRGBDWrapper(
                env,
                rgb=("rgb" in cfg.obs_mode),
                depth=("depth" in cfg.obs_mode),
                state=cfg.include_state,
            )
        else:
            env = FlattenRGBDObservationWrapper(
                env,
                rgb=("rgb" in cfg.obs_mode),
                depth=("depth" in cfg.obs_mode),
                state=cfg.include_state,
            )

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper
        env = RewardScaleBiasWrapper(env, scale=cfg.reward_scale, bias=cfg.reward_bias)

    if cfg.record_dir is not None and (cfg.save_video or cfg.save_trajectory):
        env = RecordEpisode(
            env,
            output_dir=cfg.record_dir,
            save_trajectory=cfg.save_trajectory,
            save_video=cfg.save_video,
            trajectory_name=cfg.trajectory_name,
            max_steps_per_video=cfg.max_steps_per_video,
            video_fps=cfg.video_fps,
        )

    env = ManiSkillVectorEnv(
        env,
        cfg.num_envs,
        ignore_terminations=cfg.ignore_terminations,
        record_metrics=cfg.record_metrics,
    )
    return env
