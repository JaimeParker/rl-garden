"""ManiSkill environment factory.

Wraps a ManiSkill env with the same stack ManiSkill baselines use:

    gym.make -> [FlattenRGBDObservationWrapper] -> [FlattenActionSpaceWrapper]
              -> [RecordEpisode] -> ManiSkillVectorEnv

The resulting env exposes GPU torch tensors from ``reset`` / ``step`` and is
the only env shape the rest of ``rl_garden`` targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym


@dataclass
class ManiSkillEnvConfig:
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    obs_mode: str = "state"  # "state" | "rgb" | "rgbd"
    include_state: bool = True  # only used when obs_mode is visual
    control_mode: Optional[str] = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    sim_backend: str = "gpu"
    reconfiguration_freq: Optional[int] = None
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    partial_reset: bool = False
    ignore_terminations: bool = True
    record_metrics: bool = True
    # Recording
    record_dir: Optional[str] = None
    save_video: bool = False
    save_trajectory: bool = False
    max_steps_per_video: Optional[int] = None
    video_fps: int = 30
    trajectory_name: str = "trajectory"
    human_render_camera_configs: Optional[dict] = None


def _is_visual_obs_mode(mode: str) -> bool:
    return mode in ("rgb", "rgbd", "depth")


def make_maniskill_env(cfg: ManiSkillEnvConfig):
    """Build a ManiSkill vectorized env according to ``cfg``.

    Returns the wrapped ``ManiSkillVectorEnv`` instance.
    """
    # Lazy imports so the package doesn't hard-depend on mani_skill being installed.
    import mani_skill.envs  # noqa: F401  (registers envs)
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
        sensor_configs=dict(),
    )
    if cfg.control_mode is not None:
        env_kwargs["control_mode"] = cfg.control_mode
    if cfg.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = cfg.camera_width
    if cfg.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = cfg.camera_height
    if cfg.human_render_camera_configs is not None:
        env_kwargs["human_render_camera_configs"] = cfg.human_render_camera_configs

    env = gym.make(
        cfg.env_id,
        num_envs=cfg.num_envs,
        reconfiguration_freq=cfg.reconfiguration_freq,
        **env_kwargs,
    )

    # RGBD dict -> flat {rgb(/depth), state} dict
    if _is_visual_obs_mode(cfg.obs_mode):
        env = FlattenRGBDObservationWrapper(
            env,
            rgb=("rgb" in cfg.obs_mode),
            depth=("depth" in cfg.obs_mode),
            state=cfg.include_state,
        )

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

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
