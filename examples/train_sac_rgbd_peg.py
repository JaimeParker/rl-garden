"""RGBD SAC on the vendored PegInsertionSidePegOnly-v1 ManiSkill env.

Usage:
    python examples/train_sac_rgbd_peg.py
    python examples/train_sac_rgbd_peg.py --encoder resnet10
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import tyro

from rl_garden.algorithms import RGBDSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionSACTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(VisionSACTrainingArgs):
    env_id: str = "PegInsertionSidePegOnly-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: str = "normalized_dense"
    robot_uids: str = "panda_wristcam_gripper_closed_wo_norm"
    fix_peg_pose: bool = False
    fix_box: bool = True


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__sac_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=vars(args),
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or args.env_id,
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

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
        record_dir=resolve_eval_record_dir(args, run_name),
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)

    agent = RGBDSAC(
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
