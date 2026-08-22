"""Shared lifecycle for offline pretraining."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec, run_offline_pretraining
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.cli_args import (
    resolve_checkpoint_dir,
    resolve_num_eval_steps,
    warn_if_eval_budget_undersized,
)
from rl_garden.common.effective_config import json_value, persist_effective_config
from rl_garden.envs.backend_registry import (
    EnvRequest,
    make_evaluation_env,
    should_create_eval_env,
)
from rl_garden.training._dataset import (
    infer_offline_dataset_specs,
    load_offline_dataset,
)
from rl_garden.training.inspection import (
    config_session,
    emit_materialized_config,
    has_config_session,
    is_dry_run,
    materialize_config,
    prepare_standalone,
    run_preflight,
)


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
        num_eval_steps=resolve_num_eval_steps(
            num_eval_steps=args.num_eval_steps,
            num_eval_episodes=args.num_eval_episodes,
            eval_episode_horizon=args.eval_episode_horizon,
            default=50,
        ),
        backend_config=backend_config,
    )


def run_offline(
    args: Any,
    *,
    build_agent: Callable[[Any, OfflineEnvSpec, Logger, Any | None], Any],
) -> None:
    resources: list[Any] = []
    try:
        _run_offline(args, build_agent=build_agent, resources=resources)
    finally:
        for resource in reversed(resources):
            resource.close()


def _run_offline(
    args: Any,
    *,
    build_agent: Callable[[Any, OfflineEnvSpec, Logger, Any | None], Any],
    resources: list[Any],
) -> None:
    from rl_garden.training.offline._registry import registry

    algorithm, _ = registry.entry_for_args(args)
    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args,
            registry=registry,
            training_phase="offline",
            algorithm=algorithm,
        )
        with config_session(preflight, dry_run=False):
            return _run_offline(
                normalized_args, build_agent=build_agent, resources=resources
            )

    seed_everything(args.seed)
    enable_fast_math()

    if not args.offline_dataset:
        raise SystemExit("--offline_dataset is required for offline pretraining.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")
    warn_if_eval_budget_undersized(
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        eval_episode_horizon=args.eval_episode_horizon,
    )

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{algorithm}_offline_pretrain__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    dry_run = is_dry_run()
    if dry_run:
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
            wandb_group=args.wandb_group or f"{algorithm}_offline_pretrain",
        )
    resources.append(logger)
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
        + f"\n|resolved_algorithm|{algorithm}|",
    )

    obs_space, action_space = infer_offline_dataset_specs(args)
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=args.spec_num_envs)
    if args.std_log:
        obs_desc = obs_space.shape if isinstance(obs_space, spaces.Box) else obs_space
        print(
            f"[pretrain] algorithm={algorithm} obs={obs_desc} "
            f"action={action_space.shape}",
            flush=True,
        )

    # --- optional eval env, built before the agent so it can be injected
    # through the constructor instead of assigned onto the agent afterward.
    # None whenever there's no live simulator to evaluate in (e.g. offline
    # data collected from a real robot) or periodic eval wasn't requested. ---
    eval_env = None
    if args.env_id is not None and should_create_eval_env(args):
        eval_env = make_evaluation_env(args.env_backend, _eval_env_request(args))
        resources.append(eval_env)

    agent = build_agent(args, env_spec, logger, eval_env)
    agent.num_eval_episodes = int(args.num_eval_episodes)
    if dry_run and args.load_checkpoint is not None:
        # Dry-run never loads the dataset, so there is no fresh buffer state
        # for a resumed replay buffer to collide with here; only the model
        # weights are needed for an accurate materialized-config summary.
        agent.load(args.load_checkpoint, load_replay_buffer=False)
        if args.std_log:
            print(f"[pretrain] resumed_from={args.load_checkpoint}", flush=True)
    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
    if dry_run:
        emit_materialized_config(
            env_request=_eval_env_request(args) if args.env_id is not None else {},
            env=env_spec,
            eval_env=eval_env,
            agent=agent,
            derived=materialized_derived,
        )
        return
    materialized = materialize_config(
        env_request=_eval_env_request(args) if args.env_id is not None else {},
        env=env_spec,
        eval_env=eval_env,
        agent=agent,
        derived=materialized_derived,
    )
    persist_effective_config(materialized, config_path)
    logger.update_config(json_value(materialized))
    loaded = load_offline_dataset(agent.replay_buffer, args)
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[pretrain] loaded_transitions={loaded}", flush=True)
    if hasattr(agent, "fit_obs_normalizer"):
        agent.fit_obs_normalizer()
    if hasattr(agent, "pretrain_vae"):
        agent.pretrain_vae()
    if args.load_checkpoint is not None:
        # Loaded last so a resumed checkpoint's buffer/normalizer state is
        # authoritative over the freshly re-populated offline dataset above.
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
        if args.std_log:
            print(f"[pretrain] resumed_from={args.load_checkpoint}", flush=True)

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
        eval_freq=args.eval_freq if eval_env is not None else 0,
        desc=f"{algorithm}-offline",
    )
