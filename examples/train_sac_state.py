"""State-based SAC on ManiSkill — PickCube-v1 by default.

Usage (from repo root):
    python examples/train_sac_state.py --env_id PickCube-v1 --num_envs 16
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import tyro
from torch.utils.tensorboard import SummaryWriter

from rl_garden.algorithms import SAC
from rl_garden.common import Logger, seed_everything
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    seed: int = 1
    batch_size: int = 1024
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.5
    gamma: float = 0.8
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    control_mode: str = "pd_joint_delta_pos"
    log_freq: int = 1_000
    eval_freq: int = 25
    num_eval_steps: int = 50


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    run_name = args.exp_name or f"{args.env_id}__sac_state__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    logger = Logger(tensorboard=writer)

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id, num_envs=args.num_envs, obs_mode="state",
        control_mode=args.control_mode,
    )
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id, num_envs=args.num_eval_envs, obs_mode="state",
        control_mode=args.control_mode, reconfiguration_freq=1,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    agent = SAC(
        env=env, eval_env=eval_env,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, tau=args.tau,
        training_freq=args.training_freq, utd=args.utd,
        policy_lr=args.policy_lr, q_lr=args.q_lr,
        seed=args.seed, logger=logger,
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
    )
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
