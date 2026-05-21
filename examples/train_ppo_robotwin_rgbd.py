"""PPO on RoboTwin RGB observations.

RoboTwin must be installed/importable, or pass ``--robotwin-root`` pointing to a
RoboTwin repository checkout. This entrypoint is intentionally PPO-first; the
environment itself still follows rl-garden's generic vector env contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import tyro

from rl_garden.algorithms import PPO
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionPPOTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.encoders import discover_image_keys
from rl_garden.envs import RoboTwinEnvConfig, make_robotwin_env


@dataclass
class Args(VisionPPOTrainingArgs):
    env_id: str = "place_shoe"
    robotwin_root: Optional[str] = None
    assets_path: Optional[str] = None
    seeds_path: Optional[str] = None
    device: str = "auto"
    step_lim: int = 400
    planner_backend: str = "mplib"
    embodiment: list[str] = field(default_factory=lambda: ["aloha-agilex"])
    reward_mode: str = "dense"
    control_mode: str = "delta_joint_pos"
    joint_delta_scale: float = 0.05
    ee_delta_pos_scale: float = 0.03
    ee_delta_rot_scale: float = 0.15
    gripper_delta_scale: float = 0.2
    collect_wrist_camera: bool = True


def _make_env(args: Args, num_envs: int, is_eval: bool = False):
    image_size = (
        int(args.camera_height or 64),
        int(args.camera_width or 64),
    )
    task_cfg = {
        "task_name": args.env_id,
        "step_lim": args.step_lim,
        "planner_backend": args.planner_backend,
        "embodiment": args.embodiment,
        "render_freq": 0,
        "episode_num": 100,
        "use_seed": False,
        "save_freq": 15,
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": args.collect_wrist_camera,
        },
        "data_type": {"rgb": True, "qpos": True},
        "save_path": "./data",
        "collect_data": False,
        "eval_video_log": bool(is_eval and args.capture_video),
    }
    if is_eval and args.capture_video and eval_record_dir is not None:
        task_cfg["eval_video_save_dir"] = eval_record_dir
    return make_robotwin_env(
        RoboTwinEnvConfig(
            task_name=args.env_id,
            num_envs=num_envs,
            seed=args.seed,
            robotwin_root=args.robotwin_root,
            assets_path=args.assets_path,
            seeds_path=args.seeds_path,
            step_lim=args.step_lim,
            max_episode_steps=args.step_lim,
            task_config=task_cfg,
            planner_backend=args.planner_backend,
            embodiment=args.embodiment,
            reward_mode=args.reward_mode,  # type: ignore[arg-type]
            control_mode=args.control_mode,  # type: ignore[arg-type]
            image_size=(image_height, image_width),
            joint_delta_scale=args.joint_delta_scale,
            ee_delta_pos_scale=args.ee_delta_pos_scale,
            ee_delta_rot_scale=args.ee_delta_rot_scale,
            gripper_delta_scale=args.gripper_delta_scale,
            image_size=image_size,
            include_wrist_cameras=args.collect_wrist_camera,
            auto_reset=True,
            ignore_terminations=False,
            device="auto",
        )
    )


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"robotwin_{args.env_id}__ppo_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    eval_record_dir = resolve_eval_record_dir(args, run_name)
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

    env = _make_env(args, args.num_envs)
    eval_env = _make_env(
        args,
        args.num_eval_envs,
        is_eval=True,
        eval_record_dir=eval_record_dir,
    )
    factory = image_encoder_factory_from_args(args)
    image_keys = discover_image_keys(env.single_observation_space)

    agent = PPO(
        env=env,
        eval_env=eval_env,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        finite_horizon_gae=args.finite_horizon_gae,
        detach_encoder_on_actor=args.detach_encoder_on_actor,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    agent.learn(total_timesteps=args.total_timesteps)
    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
