from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro

from mani_skill.utils import gym_utils

from act.evaluate import evaluate
from act.make_env import make_eval_envs
from train import Agent, Args as TrainArgs


@dataclass
class EvalArgs:
    # basic config
    env_id: str = "PegInsertionSide-v1"
    """Environment ID to evaluate on."""

    ckpt_path: str = (
        "runs/act-PegInsertionSide-v1-state-100_motionplanning_demos-40/"
        "checkpoints/best_eval_success_at_end.pt"
    )
    """Path to the checkpoint .pt file."""

    seed: int = 1
    """Random seed."""

    cuda: bool = True
    """If true and available, use CUDA."""

    # evaluation settings
    num_eval_episodes: int = 100
    """Number of evaluation episodes."""

    num_eval_envs: int = 10
    """Number of parallel evaluation environments."""

    capture_video: bool = True
    """Whether to capture evaluation videos."""

    max_episode_steps: Optional[int] = 200
    """Max episode steps for evaluation envs (None = use env default)."""

    # ACT / env specific
    control_mode: str = "pd_joint_delta_pos"
    """Control mode used during training (must match dataset/training)."""

    sim_backend: str = "physx_cpu"
    """Simulation backend ('physx_cpu' or 'physx_cuda')."""

    temporal_agg: bool = True
    """Whether to use temporal aggregation when evaluating."""

    num_queries: int = 30
    """Number of action queries per decision (should match training)."""


def main(args: Optional[EvalArgs] = None):
    """Entry point for evaluating a trained ACT policy."""
    if args is None:
        args = tyro.cli(EvalArgs)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # derive run name from checkpoint path (used for video dir)
    # expected ckpt_path like runs/<run_name>/checkpoints/<tag>.pt
    ckpt_parts = args.ckpt_path.split("/")
    run_name = "eval_act"
    if "runs" in ckpt_parts:
        runs_idx = ckpt_parts.index("runs")
        if runs_idx + 1 < len(ckpt_parts):
            run_name = ckpt_parts[runs_idx + 1]

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    other_kwargs = None
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/eval_videos" if args.capture_video else None,
    )

    # Build a training-style Args object so Agent has all required hyperparameters.
    train_args = TrainArgs()
    # Overwrite fields that matter for evaluation so they match the checkpoint/env.
    train_args.env_id = args.env_id
    train_args.control_mode = args.control_mode
    train_args.sim_backend = args.sim_backend
    train_args.temporal_agg = args.temporal_agg
    train_args.num_queries = args.num_queries
    train_args.cuda = args.cuda

    # build agent and load checkpoint
    agent = Agent(eval_envs, train_args).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    # prefer EMA weights if available
    state_dict_key = "ema_agent" if "ema_agent" in ckpt else "agent"
    agent.load_state_dict(ckpt[state_dict_key])
    norm_stats = ckpt.get("norm_stats", None)

    # evaluation kwargs
    max_timesteps = (
        gym_utils.find_max_episode_steps_value(eval_envs)
        if args.max_episode_steps is None
        else args.max_episode_steps
    )

    eval_kwargs = dict(
        stats=norm_stats,
        num_queries=args.num_queries,
        temporal_agg=args.temporal_agg,
        max_timesteps=max_timesteps,
        device=device,
        sim_backend=args.sim_backend,
    )

    # run evaluation
    eval_metrics = evaluate(args.num_eval_episodes, agent, eval_envs, eval_kwargs)
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])

    print(
        f"Evaluated {args.num_eval_episodes} episodes "
        f"with {args.num_eval_envs} parallel envs on {args.env_id}"
    )
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}")

    eval_envs.close()


if __name__ == "__main__":
    main()

