"""Residual SAC on PegInsertionSidePegOnly-v1 state observations.

Usage:
    python examples/train_residual_sac_state_peg.py --policy act --ckpt-path act-peg-only
    python examples/train_residual_sac_state_peg.py --debug
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import tyro

from rl_garden.algorithms import ResidualSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    SACTrainingArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


class ZeroBaseActionProvider:
    """Base-action provider that always returns raw env-space zeros."""

    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self.action_shape = action_shape

    def __call__(self, obs):
        num_envs = obs.shape[0]
        return torch.zeros((num_envs,) + self.action_shape, device=obs.device)

    def reset(self, env_ids=None) -> None:
        del env_ids


@dataclass
class Args(SACTrainingArgs):
    env_id: str = "PegInsertionSidePegOnly-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: str = "normalized_dense"
    robot_uids: str = "panda_wristcam_gripper_closed"
    fix_peg_pose: bool = False
    fix_box: bool = True

    residual_action_scale: float = 0.1
    debug: bool = False
    policy: Literal["act", "zero"] = "act"
    ckpt_path: Optional[str] = "act-peg-only"
    act_temporal_agg: bool = True
    act_temporal_agg_k: float = 0.01


def make_base_action_provider(args: Args, env):
    action_shape = tuple(env.single_action_space.shape)
    if args.debug or args.policy == "zero":
        return ZeroBaseActionProvider(action_shape)
    if args.policy == "act":
        from rl_garden.models.act import ACTBaseActionProvider

        provider = ACTBaseActionProvider.from_checkpoint(
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            ckpt_path=args.ckpt_path,
            env=env,
            temporal_agg=args.act_temporal_agg,
            temporal_agg_k=args.act_temporal_agg_k,
        )
        print(
            "[residual] base_policy=act "
            f"ckpt={provider.checkpoint_path} "
            f"state_dim={provider.spec.state_dim} "
            f"action_dim={provider.spec.action_dim} "
            f"num_queries={provider.config.num_queries}",
            flush=True,
        )
        return provider
    raise ValueError(f"Unsupported residual base policy: {args.policy!r}.")


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = "debug_zero_base" if args.debug else f"{args.policy}_base"
    run_name = (
        args.exp_name
        or f"{args.env_id}__residual_sac_state_peg_{suffix}__{args.seed}__{int(time.time())}"
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
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        fix_peg_pose=args.fix_peg_pose,
        fix_box=args.fix_box,
    )
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        fix_peg_pose=args.fix_peg_pose,
        fix_box=args.fix_box,
        reconfiguration_freq=1,
        render_mode=args.render_mode,
        record_dir=resolve_eval_record_dir(args, run_name),
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )

    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    try:
        base_action_provider = make_base_action_provider(args, env)
        agent = ResidualSAC(
            env=env,
            eval_env=eval_env,
            base_action_provider=base_action_provider,
            residual_action_scale=args.residual_action_scale,
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
        )
        if args.load_checkpoint is not None:
            agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
        agent.learn(total_timesteps=args.total_timesteps)
    finally:
        logger.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
