"""ManiSkill environment factory.

Wraps a ManiSkill env with the same stack ManiSkill baselines use:

    gym.make -> [PerCameraRGBDWrapper] -> [FlattenActionSpaceWrapper]
              -> [RecordEpisode] -> ManiSkillVectorEnv

The resulting env exposes GPU torch tensors from ``reset`` / ``step`` and is
the only env shape the rest of ``rl_garden`` targets. Vision always comes out
as per-camera ``rgb_<cam>``/``depth_<cam>`` keys (``PerCameraRGBDWrapper``) --
``FlattenRGBDObservationWrapper``'s channel-stacked bare ``rgb``/``depth`` is
not used; fusing multiple cameras is the encoder's job (see
``CombinedExtractor``'s ``fusion_mode``), not the env's.

A non-default ``reward_scale``/``reward_bias`` is applied to the reward
``ManiSkillVectorEnv.step`` returns, so its ``episode`` metrics and
``RecordEpisode`` see the raw env reward.
"""
from __future__ import annotations

import functools
from typing import Any

import gymnasium as gym

from rl_garden.envs.maniskill.config import ManiSkillEnvConfig
from rl_garden.observations.schema import ObservationContractError


@functools.cache
def _scaled_reward_vector_env_class() -> type:
    """Define the reward-scaling ``ManiSkillVectorEnv`` subclass on first use,
    keeping ``mani_skill`` a lazy import."""
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    class _ScaledRewardManiSkillVectorEnv(ManiSkillVectorEnv):
        """``ManiSkillVectorEnv`` that returns ``reward_scale * reward + reward_bias``
        after its own ``step`` has recorded the raw reward in the episode metrics."""

        def __init__(self, env, num_envs: int, *, reward_scale: float, reward_bias: float, **kwargs) -> None:
            super().__init__(env, num_envs, **kwargs)
            self.reward_scale = float(reward_scale)
            self.reward_bias = float(reward_bias)

        def step(self, actions):
            obs, reward, terminated, truncated, infos = super().step(actions)
            return obs, self.reward_scale * reward + self.reward_bias, terminated, truncated, infos

    return _ScaledRewardManiSkillVectorEnv


def make_maniskill_env(cfg: ManiSkillEnvConfig):
    """Build a ManiSkill vectorized env according to ``cfg``.

    Returns the wrapped ``ManiSkillVectorEnv`` instance.
    """
    if cfg.frame_stack < 1:
        raise ObservationContractError("frame_stack must be at least 1")
    is_visual = bool(cfg.rgb_cameras or cfg.depth_cameras)

    # Lazy imports so the package doesn't hard-depend on mani_skill being installed.
    import mani_skill.envs  # noqa: F401  (registers envs)
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    # ManiSkill's own obs_mode selects which raw sensor buffers the sim
    # renders at all; PerCameraRGBDWrapper below then filters that down to
    # exactly the cameras/keys ObservationConfig asked for.
    if cfg.rgb_cameras and cfg.depth_cameras:
        ms_obs_mode = "rgbd"
    elif cfg.rgb_cameras:
        ms_obs_mode = "rgb"
    elif cfg.depth_cameras:
        ms_obs_mode = "depth"
    else:
        ms_obs_mode = "state"

    env_kwargs: dict[str, Any] = dict(
        obs_mode=ms_obs_mode,
        render_mode=cfg.render_mode,
        sim_backend=cfg.sim_backend,
        render_backend=cfg.render_backend,
        sensor_configs=dict(),
    )
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

    if is_visual:
        from rl_garden.envs.wrappers import PerCameraRGBDWrapper

        env = PerCameraRGBDWrapper(
            env,
            rgb_cameras=cfg.rgb_cameras,
            depth_cameras=cfg.depth_cameras,
            state=cfg.state,
        )

        if cfg.frame_stack > 1:
            from rl_garden.envs.wrappers import ImageFrameStackWrapper

            env = ImageFrameStackWrapper(env, frame_stack=cfg.frame_stack)
    else:
        if cfg.frame_stack > 1:
            raise ObservationContractError("frame_stack > 1 requires a visual observation mode")
        from rl_garden.envs.wrappers import DictStateObservationWrapper

        env = DictStateObservationWrapper(env)

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

    vector_kwargs: dict[str, Any] = {
        "ignore_terminations": cfg.ignore_terminations,
        "record_metrics": cfg.record_metrics,
    }
    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        return _scaled_reward_vector_env_class()(
            env,
            cfg.num_envs,
            reward_scale=cfg.reward_scale,
            reward_bias=cfg.reward_bias,
            **vector_kwargs,
        )
    return ManiSkillVectorEnv(env, cfg.num_envs, **vector_kwargs)
