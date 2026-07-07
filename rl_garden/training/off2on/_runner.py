"""Shared offline-to-online training orchestration.

Backs the ``wsrl``, ``calql``, and ``iql`` off2on entrypoints via
``run_off2on(args, build_agent=..., algorithm=...)``: env/dataset setup, the
offline gradient-step loop, the offline->online mode switch, and the online
``learn()`` call are algorithm-agnostic. Each entrypoint supplies its own
``build_<algo>(args, env, eval_env, logger, checkpoint_dir)`` callback that
constructs its agent class from ``args`` (mirroring the
``build_<algo>`` convention already used by ``rl_garden/training/online/*.py``),
including any ``--load_checkpoint`` handling.

Usage:
    # State observations (override visual defaults)
    python examples/train_off2on.py wsrl --env_id PickCube-v1 --obs_mode state \\
        --buffer_size 1000000 --batch_size 256 --utd 4.0

    # RGB observations with plain_conv encoder
    python examples/train_off2on.py wsrl --env_id PickCube-v1 --obs_mode rgb --encoder plain_conv

    # RGBD observations with ResNet encoder
    python examples/train_off2on.py wsrl --env_id PickCube-v1 --obs_mode rgbd --encoder resnet10

    # Online-only (no offline pre-training)
    python examples/train_off2on.py wsrl --env_id PickCube-v1 --num_offline_steps 0

    # Offline→online from a ManiSkill trajectory H5
    python examples/train_off2on.py wsrl --env_id PickCube-v1 \\
        --offline_dataset_path demos/pickcube.h5 --num_offline_steps 100000

    # Minari offline pretrain -> online continuation on the recovered live env
    python examples/train_off2on.py wsrl --dataset_source minari \\
        --offline_dataset_path "D4RL/antmaze/umaze-v1" --env_backend minari \\
        --num_offline_steps 100000 --num_online_steps 500000
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

from gymnasium import spaces
from tqdm import trange

from rl_garden.algorithms.offline import _log_eval_stdout
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.cli_args import resolve_checkpoint_dir, resolve_eval_record_dir
from rl_garden.common.resolved_config import persist_resolved_config
from rl_garden.envs.backend_registry import (
    EnvRequest,
    make_training_envs,
    should_create_eval_env,
)
from rl_garden.training._dataset import load_offline_dataset
from rl_garden.training.off2on._args import (
    Off2OnCommonArgs,
    warn_if_off2on_warmup_uses_uninitialized_policy,
)

BuildAgent = Callable[[Any, Any, Any, Logger, "str | None"], Any]


def _offline_update_loop(
    agent: Any,
    steps: int,
    logger: Logger,
    log_freq: int,
    std_log: bool,
    *,
    start_step: int = 0,
) -> None:
    gradient_steps = 1
    interval_update_time = 0.0
    interval_update_steps = 0
    for step in trange(steps, desc="offline"):
        should_log = log_freq > 0 and (
            (step + 1) % log_freq == 0 or (step + 1) == steps
        )
        update_t = time.perf_counter()
        losses = agent.train(gradient_steps, compute_info=should_log)
        interval_update_time += time.perf_counter() - update_t
        interval_update_steps += gradient_steps

        if (
            agent.eval_freq > 0
            and agent.eval_env is not None
            and (step + 1) % agent.eval_freq == 0
        ):
            eval_t = time.perf_counter()
            eval_metrics = agent._evaluate()
            agent._log_eval_metrics(eval_metrics, start_step + step + 1)
            if std_log:
                _log_eval_stdout(agent, eval_metrics, start_step + step + 1)
            logger.add_scalar(
                "time/eval_time", time.perf_counter() - eval_t, start_step + step + 1
            )

        if log_freq > 0 and (step + 1) % log_freq == 0:
            logger.log_metrics(losses, start_step + step + 1)
            offline_update_fps = (
                interval_update_steps / interval_update_time
                if interval_update_time > 0
                else float("nan")
            )
            logger.add_scalar(
                "time/offline_update_time",
                interval_update_time,
                start_step + step + 1,
            )
            logger.add_scalar(
                "time/offline_update_fps",
                offline_update_fps,
                start_step + step + 1,
            )
            if std_log:
                progress = 100.0 * (step + 1) / steps if steps > 0 else 100.0
                loss_summary, q_summary = logger.format_metrics(losses)
                q_part = f" q={q_summary}" if q_summary else ""
                print(
                    "[offline] "
                    f"step={step + 1}/{steps} ({progress:.2f}%) "
                    f"fps={offline_update_fps:.4f} "
                    f"{loss_summary}{q_part}",
                    flush=True,
                )
            interval_update_time = 0.0
            interval_update_steps = 0

        if agent.checkpoint_freq > 0 and (step + 1) % agent.checkpoint_freq == 0:
            agent._save_checkpoint(f"checkpoint_{step + 1}.pt")


def _evaluate_offline_end(agent: Any, logger: Logger, step: int, std_log: bool) -> None:
    metrics = agent._evaluate()
    if not metrics:
        return
    agent._log_eval_metrics(metrics, step)
    for key, value in agent.canonical_eval_metrics(metrics).items():
        logger.add_summary(f"off2on/offline_final_eval/{key}", value)
    if std_log:
        _log_eval_stdout(agent, metrics, step)


def _save_offline_checkpoint(
    agent: Any,
    checkpoint_dir: str | None,
    *,
    include_replay_buffer: bool,
    std_log: bool,
) -> None:
    if checkpoint_dir is None:
        return
    path = agent.save(
        Path(checkpoint_dir) / "offline_final.pt",
        include_replay_buffer=include_replay_buffer,
    )
    if std_log:
        print(f"[offline] saved_checkpoint={path}", flush=True)


def _set_offline_probe(agent: Any, logger: Logger, std_log: bool) -> None:
    probe_size = min(agent.batch_size, len(agent.replay_buffer))
    if probe_size <= 0:
        logger.add_summary("off2on/offline_probe_size", 0)
        return
    agent.set_offline_probe_batch(agent.replay_buffer.sample(probe_size))
    logger.add_summary("off2on/offline_probe_size", probe_size)
    if std_log:
        print(f"[offline] probe_size={probe_size}", flush=True)


def _resolve_env_id(args: Off2OnCommonArgs) -> str:
    """Default the online env_id to the Minari dataset id, unless overridden.

    Only applies when ``dataset_source == "minari"`` and ``env_id`` is left at
    its ``EnvRunArgs`` default; an explicit ``--env_id`` always wins.
    """
    if args.dataset_source == "minari" and args.env_id == "PickCube-v1":
        return args.offline_dataset_path
    return args.env_id


def _require_continuous_action_space(env, args: Off2OnCommonArgs) -> None:
    if not isinstance(env.single_action_space, spaces.Box):
        raise ValueError(
            f"env_backend={args.env_backend!r} env_id={args.env_id!r} has a "
            f"{type(env.single_action_space).__name__} action space; off2on "
            "training only supports continuous (Box) actions."
        )


def _switch_to_online_mode(agent: Any, args: Off2OnCommonArgs, logger: Logger) -> None:
    if args.num_offline_steps == 0:
        warn_if_off2on_warmup_uses_uninitialized_policy(args)
        if args.load_checkpoint is not None and args.offline_dataset_path is not None:
            loaded = load_offline_dataset(agent.replay_buffer, args)
            logger.add_summary("off2on/offline_loaded_transitions", loaded)
            _set_offline_probe(agent, logger, args.std_log)
    agent.switch_to_online_mode(
        online_replay_mode=args.online_replay_mode,
        offline_data_ratio=args.offline_data_ratio,
    )
    if args.std_log:
        print(f"[online] replay_mode={args.online_replay_mode}", flush=True)


def run_off2on(
    args: Off2OnCommonArgs, *, build_agent: BuildAgent, algorithm: str
) -> None:
    import torch

    seed_everything(args.seed)
    enable_fast_math()

    args.env_id = _resolve_env_id(args)

    if args.buffer_device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available; falling back to CPU buffer.", stacklevel=2)
        args.buffer_device = "cpu"

    is_visual = args.obs_mode != "state"
    obs_label = f"rgbd_{args.encoder}" if is_visual else "state"
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__{algorithm}_{obs_label}__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    resolved_config = persist_resolved_config(
        args,
        training_phase="off2on",
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

    if args.eval_freq > 0 and args.num_eval_envs <= 0:
        raise SystemExit(
            "--eval_freq > 0 requires --num_eval_envs > 0 to provide an eval environment."
        )

    backend_config = args.resolve_backend_config()
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    req = EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width if is_visual else None,
        camera_height=args.camera_height if is_visual else None,
        include_state=args.include_state if is_visual else True,
        per_camera_rgbd=args.per_camera_rgbd if is_visual else False,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
        create_eval_env=should_create_eval_env(args),
    )
    env, eval_env = make_training_envs(args.env_backend, req)
    _require_continuous_action_space(env, args)

    agent = build_agent(args, env, eval_env, logger, checkpoint_dir)

    # Offline training phase
    if args.num_offline_steps > 0:
        if args.offline_dataset_path is None:
            raise ValueError(
                "--offline_dataset_path is required when --num_offline_steps > 0."
            )
        loaded = load_offline_dataset(agent.replay_buffer, args)
        offline_start_step = agent._global_step
        logger.add_summary("off2on/offline_loaded_transitions", loaded)
        logger.add_summary("off2on/offline_start_step", offline_start_step)
        _offline_update_loop(
            agent,
            args.num_offline_steps,
            logger,
            args.log_freq,
            args.std_log,
            start_step=offline_start_step,
        )
        offline_end_step = offline_start_step + args.num_offline_steps
        agent._global_step = offline_end_step
        _evaluate_offline_end(agent, logger, offline_end_step, args.std_log)
        _save_offline_checkpoint(
            agent,
            checkpoint_dir,
            include_replay_buffer=args.save_replay_buffer,
            std_log=args.std_log,
        )
        _set_offline_probe(agent, logger, args.std_log)

    _switch_to_online_mode(agent, args, logger)

    # Online training phase
    if args.num_online_steps > 0:
        online_target_step = agent._global_step + args.num_online_steps
        agent.learn(total_timesteps=online_target_step)
    elif agent.checkpoint_dir is not None and agent.save_final_checkpoint:
        agent._save_checkpoint("final.pt")

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()
