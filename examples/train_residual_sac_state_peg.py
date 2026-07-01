"""Residual SAC on PegInsertionSidePegOnly-v1 state observations.

Usage:
    python examples/train_residual_sac_state_peg.py --base_policy act --base_ckpt_path act-peg-only
    python examples/train_residual_sac_state_peg.py --debug
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro

from rl_garden.algorithms import ResidualSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.training.online._args import SACTrainingArgs
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env
from rl_garden.policies.base_policies import make_base_policy


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
    fixed_peg_xy: tuple[float, float] = (-0.05, -0.15)
    fixed_peg_z_rot_deg: float = 67.5

    residual_action_scale: float = 0.1
    n_critics: int = 2
    critic_subsample_size: Optional[int] = None
    debug: bool = False
    base_policy: Literal["act", "sac", "zero"] = "act"
    base_ckpt_path: Optional[str] = "act-peg-only"
    base_act_temporal_agg: bool = True
    base_act_temporal_agg_k: float = 0.01
    base_sac_deterministic: bool = True

    # Deprecated aliases kept so older launch commands continue to work.
    policy: Optional[Literal["act", "zero"]] = None
    ckpt_path: Optional[str] = None


def _effective_base_policy(args: Args) -> Literal["act", "sac", "zero"]:
    if args.debug:
        return "zero"
    return args.policy if args.policy is not None else args.base_policy


def _effective_base_ckpt_path(args: Args) -> Optional[str]:
    return args.ckpt_path if args.ckpt_path is not None else args.base_ckpt_path


def make_base_action_provider(args: Args, env):
    base_policy = _effective_base_policy(args)
    provider = make_base_policy(
        base_policy=base_policy,
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env=env,
        base_ckpt_path=_effective_base_ckpt_path(args),
        base_act_temporal_agg=args.base_act_temporal_agg,
        base_act_temporal_agg_k=args.base_act_temporal_agg_k,
        base_sac_deterministic=args.base_sac_deterministic,
    )
    if base_policy == "act":
        print(
            "[residual] base_policy=act "
            f"ckpt={provider.checkpoint_path} "
            f"state_dim={provider.spec.state_dim} "
            f"action_dim={provider.spec.action_dim} "
            f"num_queries={provider.config.num_queries}",
            flush=True,
        )
    elif base_policy == "sac":
        print(
            "[residual] base_policy=sac "
            f"ckpt={_effective_base_ckpt_path(args)} "
            f"deterministic={args.base_sac_deterministic}",
            flush=True,
        )
    else:
        print("[residual] base_policy=zero", flush=True)
    return provider


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_policy = _effective_base_policy(args)
    suffix = "debug_zero_base" if args.debug else f"{base_policy}_base"
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
        fixed_peg_xy=args.fixed_peg_xy,
        fixed_peg_z_rot_deg=args.fixed_peg_z_rot_deg,
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
        fixed_peg_xy=args.fixed_peg_xy,
        fixed_peg_z_rot_deg=args.fixed_peg_z_rot_deg,
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
            n_critics=args.n_critics,
            critic_subsample_size=args.critic_subsample_size,
            critic_impl=args.critic_impl,
            alpha_tuning=args.alpha_tuning,
            q_mc_diagnostics=args.q_mc_diagnostics,
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
