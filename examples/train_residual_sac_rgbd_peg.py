"""Residual SAC on the vendored PegInsertionSidePegOnly-v1 ManiSkill RGBD env.

Usage:
    python examples/train_residual_sac_rgbd_peg.py --policy act --ckpt-path act-peg-only
    python examples/train_residual_sac_rgbd_peg.py --debug --encoder resnet10
"""
from __future__ import annotations

from dataclasses import dataclass

import tyro

from rl_garden.envs import ManiSkillEnvConfig

from train_residual_sac_rgbd import (
    ResidualRGBDArgs,
    run_residual_rgbd_training,
)


@dataclass
class Args(ResidualRGBDArgs):
    env_id: str = "PegInsertionSidePegOnly-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: str = "normalized_dense"
    robot_uids: str = "panda_wristcam_gripper_closed"
    fix_peg_pose: bool = False
    fix_box: bool = True


def main() -> None:
    args = tyro.cli(Args)
    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        fix_peg_pose=args.fix_peg_pose,
        fix_box=args.fix_box,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        fix_peg_pose=args.fix_peg_pose,
        fix_box=args.fix_box,
        reconfiguration_freq=1,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        render_mode=args.render_mode,
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )
    run_residual_rgbd_training(
        args,
        env_cfg,
        eval_cfg,
        run_label="residual_sac_rgbd_peg",
    )


if __name__ == "__main__":
    main()
