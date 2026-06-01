"""SAC on RoboTwin RGB observations.

RoboTwin must be installed/importable, or pass ``--robotwin-root`` pointing to a
RoboTwin repository checkout. This entrypoint keeps rl-garden's GPU-first SAC
path, but uses smaller images by default so visual replay fits in GPU memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import tyro

from rl_garden.algorithms import SAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionSACTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    resolve_checkpoint_dir,
    sac_family_policy_kwargs_from_args,
)
from rl_garden.encoders import discover_image_keys
from rl_garden.envs import RoboTwinEnvConfig, make_robotwin_env


@dataclass
class Args(VisionSACTrainingArgs):
    env_id: str = "place_shoe"
    robotwin_root: Optional[str] = None
    assets_path: Optional[str] = None
    seeds_path: Optional[str] = None
    step_lim: int = 400
    planner_backend: str = "mplib"
    embodiment: list[str] = field(default_factory=lambda: ["aloha-agilex"])
    reward_mode: str = "dense"
    control_mode: str = "delta_joint_pos"
    joint_delta_scale: float = 0.05
    gripper_delta_scale: float = 0.2
    include_wrist_cameras: bool = True
    render_every_control_step: bool = False
    control_step_cap: Optional[int] = None
    random_light: bool = False
    crazy_random_light_rate: float = 0.0
    head_camera_type: str = "D435"
    wrist_camera_type: str = "D435"
    device: str = "auto"
    buffer_size: int = 100_000


def _make_env(args: Args, num_envs: int, is_eval: bool = False):
    image_size = (args.camera_height or 64, args.camera_width or 64)
    task_cfg = {
        "task_name": args.env_id,
        "step_lim": args.step_lim,
        "planner_backend": args.planner_backend,
        "embodiment": args.embodiment,
        "render_freq": 0,
        "render_every_control_step": args.render_every_control_step,
        "control_step_cap": args.control_step_cap,
        "episode_num": 100,
        "use_seed": False,
        "save_freq": 15,
        "camera": {
            "head_camera_type": args.head_camera_type,
            "wrist_camera_type": args.wrist_camera_type,
            "collect_head_camera": True,
            "collect_wrist_camera": args.include_wrist_cameras,
        },
        "domain_randomization": {
            "random_background": True,
            "cluttered_table": True,
            "clean_background_rate": 0.02,
            "random_head_camera_dis": 0,
            "random_table_height": 0.03,
            "random_light": args.random_light,
            "crazy_random_light_rate": args.crazy_random_light_rate,
        },
        "data_type": {"rgb": True, "qpos": True},
        "save_path": "./data",
        "collect_data": False,
        "eval_video_log": bool(is_eval and args.capture_video),
    }
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
            joint_delta_scale=args.joint_delta_scale,
            gripper_delta_scale=args.gripper_delta_scale,
            render_every_control_step=args.render_every_control_step,
            control_step_cap=args.control_step_cap,
            random_light=args.random_light,
            crazy_random_light_rate=args.crazy_random_light_rate,
            head_camera_type=args.head_camera_type,
            wrist_camera_type=args.wrist_camera_type,
            include_wrist_cameras=args.include_wrist_cameras,
            image_size=image_size,
            auto_reset=True,
            ignore_terminations=False,
            device=args.device,
        )
    )


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"robotwin_{args.env_id}__sac_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
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

    env = _make_env(args, args.num_envs)
    eval_env = _make_env(args, args.num_eval_envs, is_eval=True) if args.num_eval_envs > 0 else None
    factory = image_encoder_factory_from_args(args)
    image_keys = discover_image_keys(env.single_observation_space)
    policy_kwargs = sac_family_policy_kwargs_from_args(args, image_keys)

    agent = SAC(
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
        policy_kwargs=policy_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
