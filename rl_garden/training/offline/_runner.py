"""Shared lifecycle for offline pretraining."""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import torch
from gymnasium import spaces

from rl_garden.algorithms import (
    OfflineEnvSpec,
    infer_specs_from_h5,
    run_offline_pretraining,
)
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.cli_args import resolve_checkpoint_dir
from rl_garden.common.resolved_config import persist_resolved_config
from rl_garden.envs.backend_registry import EnvRequest, make_evaluation_env


def _save_filename(args: Any, algorithm: str) -> str:
    if args.save_filename is not None:
        return args.save_filename
    return f"{algorithm.replace('-', '_')}_offline_pretrained.pt"


def _eval_env_request(args: Any) -> EnvRequest:
    backend_config = args.resolve_backend_config()
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.spec_num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        include_state=args.include_state,
        per_camera_rgbd=args.per_camera_rgbd,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        num_eval_envs=args.num_eval_envs,
        capture_video=False,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


def run_offline(
    args: Any,
    *,
    build_agent: Callable[[Any, OfflineEnvSpec, Logger], Any],
) -> None:
    from rl_garden.training.offline._registry import registry

    algorithm, _ = registry.entry_for_args(args)

    seed_everything(args.seed)
    enable_fast_math()

    if not args.offline_dataset_path:
        raise SystemExit("--offline_dataset_path is required for offline pretraining.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")
    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available; falling back to CPU buffer.", stacklevel=2)
        args.buffer_device = "cpu"

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{algorithm}_offline_pretrain__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    resolved_config = persist_resolved_config(
        args,
        training_phase="offline",
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
        wandb_group=args.wandb_group or f"{algorithm}_offline_pretrain",
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
        + f"\n|resolved_algorithm|{algorithm}|",
    )

    obs_space, action_space = infer_specs_from_h5(
        args.offline_dataset_path,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=args.spec_num_envs)
    if args.std_log:
        obs_desc = obs_space.shape if isinstance(obs_space, spaces.Box) else obs_space
        print(
            f"[pretrain] algorithm={algorithm} obs={obs_desc} "
            f"action={action_space.shape}",
            flush=True,
        )

    agent = build_agent(args, env_spec, logger)
    loaded = load_maniskill_h5_to_replay_buffer(
        agent.replay_buffer,
        args.offline_dataset_path,
        num_traj=args.offline_num_traj,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        success_key=args.success_key,
    )
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[pretrain] loaded_transitions={loaded}", flush=True)

    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
        if args.std_log:
            print(f"[pretrain] resumed_from={args.load_checkpoint}", flush=True)

    # --- setup optional eval env ---
    eval_env = None
    if args.env_id is not None:
        eval_env = make_evaluation_env(args.env_backend, _eval_env_request(args))
        agent.eval_env = eval_env
        agent.eval_freq = args.eval_freq
        agent.num_eval_steps = args.num_eval_steps

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename=_save_filename(args, algorithm),
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=agent.eval_freq if eval_env is not None else 0,
        desc=f"{algorithm}-offline",
    )

    if eval_env is not None:
        eval_env.close()
    logger.close()
