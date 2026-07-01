"""Shared online training lifecycle (mirrors offline/_runner.py)."""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable

import torch

from rl_garden.common import Logger, seed_everything
from rl_garden.common.resolved_config import persist_resolved_config
from rl_garden.envs.backend_registry import EnvRequest, make_training_envs


def run_online(
    args: Any,
    *,
    obs_tag: str = "",
    make_env_request: Callable[[Any, str], EnvRequest],
    build_agent: Callable[[Any, Any, Any, Logger, str | None], Any],
    post_learn: Callable[[Any], None] | None = None,
) -> None:
    from rl_garden.training.online._registry import registry

    algorithm, _ = registry.entry_for_args(args)

    seed_everything(args.seed)

    if getattr(args, "buffer_device", None) == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available; falling back to CPU buffer.", stacklevel=2)
        args.buffer_device = "cpu"

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tag = f"_{obs_tag}" if obs_tag else ""
    run_name = args.exp_name or f"{args.env_id}__{algorithm}{tag}__{args.seed}__{int(time.time())}"

    _chkpt_override = getattr(args, "checkpoint_dir", None)
    if _chkpt_override is not None:
        checkpoint_dir: str | None = _chkpt_override
    elif not args.save_final_checkpoint and args.checkpoint_freq <= 0:
        checkpoint_dir = None
    else:
        checkpoint_dir = os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args,
        training_phase="online",
        algorithm=algorithm,
        run_name=run_name,
        log_dir=args.log_dir,
    )
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=resolved_config,
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

    req = make_env_request(args, run_name)
    env, eval_env = make_training_envs(args.env_backend, req)

    agent = build_agent(args, env, eval_env, logger, checkpoint_dir)
    agent.learn(total_timesteps=args.total_timesteps)
    if post_learn is not None:
        post_learn(agent)

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()
