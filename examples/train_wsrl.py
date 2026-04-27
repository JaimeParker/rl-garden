"""State-based WSRL on ManiSkill with offline→online training.

Usage:
    # Online-only (no offline pre-training)
    python examples/train_wsrl.py --env_id PickCube-v1 --num_offline_steps 0

    # Offline→online training
    python examples/train_wsrl.py --env_id PickCube-v1 --num_offline_steps 100000 --num_online_steps 50000

    # Disable REDQ (use 2 critics like standard SAC)
    python examples/train_wsrl.py --env_id PickCube-v1 --n_critics 2
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import tyro
from torch.utils.tensorboard import SummaryWriter

from rl_garden.algorithms import WSRL
from rl_garden.common import Logger, seed_everything
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    control_mode: str = "pd_joint_delta_pos"

    # Training phases
    num_offline_steps: int = 0  # Set to 0 for online-only
    num_online_steps: int = 1_000_000

    # Buffer and training
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    seed: int = 1
    batch_size: int = 256
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005

    # Optimizers
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    cql_alpha_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1

    # Q-ensemble (REDQ)
    n_critics: int = 10
    critic_subsample_size: int = 2

    # CQL parameters
    use_cql_loss: bool = True
    cql_n_actions: int = 10
    cql_alpha: float = 5.0
    cql_autotune_alpha: bool = False
    cql_importance_sample: bool = True
    cql_max_target_backup: bool = True

    # Cal-QL parameters
    use_calql: bool = True
    calql_bound_random_actions: bool = False

    # Phase control
    online_cql_alpha: Optional[float] = None
    online_use_cql_loss: Optional[bool] = None

    # Logging
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    log_freq: int = 1_000
    eval_freq: int = 25
    num_eval_steps: int = 50


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    run_name = args.exp_name or f"{args.env_id}__wsrl_state__{args.seed}__{int(time.time())}"
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

    agent = WSRL(
        env=env, eval_env=eval_env,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, tau=args.tau,
        training_freq=args.training_freq, utd=args.utd,
        policy_lr=args.policy_lr, q_lr=args.q_lr,
        alpha_lr=args.alpha_lr, cql_alpha_lr=args.cql_alpha_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        use_cql_loss=args.use_cql_loss,
        cql_n_actions=args.cql_n_actions,
        cql_alpha=args.cql_alpha,
        cql_autotune_alpha=args.cql_autotune_alpha,
        cql_importance_sample=args.cql_importance_sample,
        cql_max_target_backup=args.cql_max_target_backup,
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        seed=args.seed, logger=logger,
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
    )

    # Offline training phase
    if args.num_offline_steps > 0:
        logger.record("phase", "offline")
        agent.learn(total_timesteps=args.num_offline_steps)

        # Switch to online mode
        agent.switch_to_online_mode()
        logger.record("phase", "online")

    # Online training phase
    if args.num_online_steps > 0:
        agent.learn(total_timesteps=args.num_online_steps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
