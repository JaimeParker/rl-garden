"""Shared online training lifecycle (mirrors offline/_runner.py)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from rl_garden.common import Logger, seed_everything
from rl_garden.common.ddp import (
    broadcast_module_state,
    ddp_local_rank,
    ddp_rank,
    init_ddp,
    pin_backend_config_device,
    shutdown_ddp,
)
from rl_garden.common.effective_config import json_value, persist_effective_config
from rl_garden.envs.backend_registry import EnvRequest, make_training_envs
from rl_garden.training.inspection import (
    config_session,
    emit_materialized_config,
    has_config_session,
    is_dry_run,
    materialize_config,
    prepare_standalone,
    run_preflight,
)


def run_online(
    args: Any,
    *,
    obs_tag: str = "",
    make_env_request: Callable[[Any, str], EnvRequest],
    build_agent: Callable[[Any, Any, Any, Logger, str | None], Any],
    post_learn: Callable[[Any], None] | None = None,
) -> None:
    from rl_garden.training.online._registry import registry

    init_ddp()

    algorithm, _ = registry.entry_for_args(args)
    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args,
            registry=registry,
            training_phase="online",
            algorithm=algorithm,
        )
        with config_session(preflight, dry_run=False):
            return run_online(
                normalized_args,
                obs_tag=obs_tag,
                make_env_request=make_env_request,
                build_agent=build_agent,
                post_learn=post_learn,
            )

    is_rank0 = ddp_rank() == 0
    effective_seed = args.seed + ddp_rank()
    seed_everything(effective_seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tag = f"_{obs_tag}" if obs_tag else ""
    run_name = (
        args.exp_name
        or f"{args.env_id}__{algorithm}{tag}__{args.seed}__{int(time.time())}"
    )

    _chkpt_override = getattr(args, "checkpoint_dir", None)
    if _chkpt_override is not None:
        checkpoint_dir: str | None = _chkpt_override
    elif not args.save_final_checkpoint and args.checkpoint_freq <= 0:
        checkpoint_dir = None
    else:
        checkpoint_dir = os.path.join(args.log_dir, run_name, "checkpoints")
    if not is_rank0:
        # Only rank 0 writes checkpoints/logs/config -- run_name/checkpoint_dir
        # are not rank-salted, so every non-rank-0 write would race with
        # rank 0's on the same path.
        checkpoint_dir = None

    dry_run = is_dry_run()
    config_path: Path | None = None
    if dry_run or not is_rank0:
        logger = Logger(log_type="none")
    else:
        config_path = Path(args.log_dir) / run_name / "config.json"
        preflight_config = run_preflight(
            {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
        )
        persist_effective_config(preflight_config, config_path)
        resolved_config = json_value(preflight_config)
        logger = Logger.create(
            log_type=args.log_type,
            log_dir=args.log_dir,
            run_name=run_name,
            config=resolved_config,
            start_time=start_time,
            log_keywords=args.log_keywords,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            log_group=args.log_group or args.env_id,
        )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # A separate args copy carrying this rank's own seed, used only for env
    # construction and agent construction -- run_name/config persistence
    # above intentionally keep the original args.seed (the user's CLI value,
    # not an internal per-rank implementation detail).
    rank_args = replace(args, seed=effective_seed)
    req = make_env_request(rank_args, run_name)
    pin_backend_config_device(req.backend_config, ddp_local_rank())
    if dry_run:
        req = replace(req, capture_video=False, eval_record_dir=None)
    env = None
    eval_env = None
    try:
        env, eval_env = make_training_envs(args.env_backend, req)
        agent = build_agent(rank_args, env, eval_env, logger, checkpoint_dir)
        agent_policy = getattr(agent, "policy", None)
        if agent_policy is not None:
            broadcast_module_state(agent_policy, src=0)
        extra_broadcast = getattr(agent, "_ddp_extra_broadcast_modules", None)
        if extra_broadcast is not None:
            for extra_module in extra_broadcast():
                broadcast_module_state(extra_module, src=0)
        materialized_derived = {
            "run_name": run_name,
            "checkpoint_dir": checkpoint_dir,
            "steps_per_env": getattr(agent, "steps_per_env", None),
            "grad_steps_per_iteration": getattr(
                agent, "grad_steps_per_iteration", None
            ),
        }
        if dry_run:
            materialized_derived["dry_run_resource_overrides"] = {
                "capture_video": False,
                "eval_record_dir": None,
            }
        if dry_run:
            emit_materialized_config(
                env_request=req,
                env=env,
                eval_env=eval_env,
                agent=agent,
                derived=materialized_derived,
            )
            return
        if is_rank0:
            materialized = materialize_config(
                env_request=req,
                env=env,
                eval_env=eval_env,
                agent=agent,
                derived=materialized_derived,
            )
            persist_effective_config(materialized, config_path)
            logger.update_config(json_value(materialized))
        agent.learn(total_timesteps=args.total_timesteps)
        if post_learn is not None:
            post_learn(agent)
    finally:
        logger.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        shutdown_ddp()
