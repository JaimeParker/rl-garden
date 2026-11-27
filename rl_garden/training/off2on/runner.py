"""Shared offline-to-online training orchestration.

Backs the ``wsrl``, ``calql``, and ``iql`` off2on entrypoints via
``run_off2on(args, build_agent=..., algorithm=...)``: env/dataset setup, the
offline gradient-step loop, the offline->online mode switch, and the online
``learn()`` call are algorithm-agnostic. Each entrypoint supplies its own
``build_<algo>(args, env, eval_env, logger, checkpoint_dir)`` callback that
constructs its agent class from ``args`` (mirroring the
``build_<algo>`` convention already used by ``rl_garden/training/online/entrypoints/*.py``),
including any ``--load_checkpoint`` handling.

Usage:
    # State observations (the default)
    python examples/train_off2on.py wsrl --env_id PickCube-v1 \\
        --buffer_size 1000000 --batch_size 256 --utd 4.0

    # RGB observations with plain_conv encoder
    python examples/train_off2on.py wsrl --env_id PickCube-v1 \\
        --obs.rgb base_camera --encoder.backbone plain_conv

    # RGBD observations with ResNet encoder
    python examples/train_off2on.py wsrl --env_id PickCube-v1 \\
        --obs.rgb base_camera --obs.depth base_camera --encoder.backbone resnet10

    # Online-only (no offline pre-training)
    python examples/train_off2on.py wsrl --env_id PickCube-v1 --num_offline_steps 0

    # Offline→online from a ManiSkill trajectory H5
    python examples/train_off2on.py wsrl --env_id PickCube-v1 \\
        --offline_dataset demos/pickcube.h5 --num_offline_steps 100000

    # Minari offline pretrain -> online continuation on the recovered live env
    python examples/train_off2on.py wsrl --dataset_backend minari \\
        --offline_dataset "D4RL/antmaze/umaze-v1" --env_backend minari \\
        --num_offline_steps 100000 --num_online_steps 500000
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from gymnasium import spaces

from rl_garden.algorithms.offline import _log_eval_stdout, run_offline_pretraining
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.cli_args import resolve_checkpoint_dir, warn_if_eval_budget_undersized
from rl_garden.common.effective_config import json_value, persist_effective_config
from rl_garden.common.env_args import make_env_request
from rl_garden.envs.backend_registry import make_training_envs, should_create_eval_env
from rl_garden.training.inspection import (
    checkpoint_to_restore,
    config_session,
    current_preflight,
    emit_materialized_config,
    has_config_session,
    is_dry_run,
    materialize_config,
    prepare_standalone,
    run_preflight,
)
from rl_garden.training.off2on.args import (
    Off2OnCommonArgs,
    warn_if_off2on_warmup_uses_uninitialized_policy,
)
from rl_garden.training.offline_dataset import load_offline_dataset

_logger = logging.getLogger(__name__)

BuildAgent = Callable[[Any, Any, Any, Logger, "str | None"], Any]


def _offline_update_loop(
    agent: Any,
    steps: int,
    logger: Logger,
    log_freq: int,
    std_log: bool,
    *,
    eval_freq: int | None = None,
) -> None:
    # `eval_freq` here is in *gradient-update* units (this loop takes exactly
    # one gradient step per `step`), independent of the online phase's
    # env-step-unit `eval_freq` (see `OffPolicyAlgorithm.learn()`). `None`
    # preserves the pre-existing behavior of reading `agent.eval_freq`.
    offline_eval_freq = agent.eval_freq if eval_freq is None else eval_freq
    run_offline_pretraining(
        agent,
        num_steps=steps,
        gradient_steps=1,
        checkpoint_dir=getattr(agent, "checkpoint_dir", None),
        checkpoint_freq=getattr(agent, "checkpoint_freq", 0),
        # off2on saves its own offline_final.pt separately via
        # `_save_offline_checkpoint` right after this loop returns.
        save_final_checkpoint=False,
        save_replay_buffer=getattr(agent, "save_replay_buffer", False),
        log_freq=log_freq,
        std_log=std_log,
        eval_freq=offline_eval_freq,
        desc="offline",
    )


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


def _set_offline_probe(
    agent: Any, logger: Logger, std_log: bool, offline_buffer: Any = None
) -> None:
    buffer = agent.replay_buffer if offline_buffer is None else offline_buffer
    probe_size = min(agent.batch_size, buffer.sampleable_size)
    if probe_size <= 0:
        logger.add_summary("off2on/offline_probe_size", 0)
        return
    agent.set_offline_probe_batch(buffer.sample(probe_size))
    logger.add_summary("off2on/offline_probe_size", probe_size)
    if std_log:
        print(f"[offline] probe_size={probe_size}", flush=True)


def _require_continuous_action_space(env, args: Off2OnCommonArgs) -> None:
    if not isinstance(env.single_action_space, spaces.Box):
        raise ValueError(  # noqa: TRY004 - invalid environment configuration value
            f"env_backend={args.env_backend!r} env_id={args.env_id!r} has a "
            f"{type(env.single_action_space).__name__} action space; off2on "
            "training only supports continuous (Box) actions."
        )


def _resolve_env_id(args: Off2OnCommonArgs) -> str:
    """Return the live env id implied by a Minari dataset when not overridden."""
    if args.dataset_backend == "minari" and args.env_id == "PickCube-v1":
        return args.offline_dataset
    return args.env_id


def _load_offline_stage_dataset(agent: Any, args: Off2OnCommonArgs, logger: Logger) -> None:
    """Fill the replay buffer of the offline stage from the dataset, unless a
    checkpoint's replay file already did."""
    if args.offline_dataset is None:
        return
    if args.load_checkpoint is not None and agent.replay_buffer_restored:
        if len(agent.replay_buffer) > 0:
            _logger.warning(
                "--offline-dataset is not loaded: the replay buffer restored from the checkpoint "
                "holds %d transitions; resume without --load-replay-buffer to use the dataset "
                "instead",
                len(agent.replay_buffer),
            )
            return
    loaded = load_offline_dataset(agent.replay_buffer, args)
    logger.add_summary("off2on/offline_loaded_transitions", loaded)


def _holds_restored_offline_stage_data(agent: Any, args: Off2OnCommonArgs) -> bool:
    return (
        args.load_checkpoint is not None
        and agent.replay_buffer_restored
        and len(agent.replay_buffer) > 0
    )


def _prepare_dataset(agent: Any, args: Off2OnCommonArgs) -> None:
    """The setup a run does on its dataset before any policy update."""
    # A loaded policy was trained with the statistics its checkpoint holds.
    if args.load_checkpoint is None and hasattr(agent, "fit_obs_normalizer"):
        agent.fit_obs_normalizer()
    if hasattr(agent, "pretrain_vae"):
        agent.pretrain_vae()


def _switch_to_online_mode(agent: Any, args: Off2OnCommonArgs, logger: Logger) -> None:
    if args.num_offline_steps == 0:
        warn_if_off2on_warmup_uses_uninitialized_policy(args)
        if (
            args.offline_dataset is not None
            and args.load_checkpoint is None
            and args.online_replay_mode != "empty"
        ):
            # The references keep the dataset for the online phase however many
            # offline steps run.
            _load_offline_stage_dataset(agent, args, logger)
            _prepare_dataset(agent, args)
            if hasattr(agent, "pretrain_bc_flow"):
                agent.pretrain_bc_flow()
            _set_offline_probe(agent, logger, args.std_log)
        elif args.load_checkpoint is not None and args.offline_dataset is not None:
            _load_offline_stage_dataset(agent, args, logger)
            if hasattr(agent, "pretrain_vae"):
                agent.pretrain_vae()
            _set_offline_probe(agent, logger, args.std_log)
    agent.switch_to_online_mode(
        online_replay_mode=args.online_replay_mode,
        offline_data_ratio=args.offline_data_ratio,
    )
    if args.std_log:
        print(f"[online] replay_mode={args.online_replay_mode}", flush=True)


def _is_explicit(args: Off2OnCommonArgs, name: str) -> bool:
    """Whether the CLI, a preset or the caller set this field."""
    return name in current_preflight().sources or getattr(args, name) != getattr(type(args), name)


#: Args fields that fix the range of returns a critic represents, which a
#: resume takes from the command line like every other hyperparameter.
_VALUE_RANGE_FIELDS = ("min_reward", "max_reward", "r_min", "r_max")


def _report_changed_value_range(args: Off2OnCommonArgs, checkpoint: Mapping[str, Any]) -> None:
    recorded = checkpoint["metadata"].get("hyperparameters", {})
    changed = {
        name: recorded[name]
        for name in _VALUE_RANGE_FIELDS
        if hasattr(args, name) and name in recorded and recorded[name] != getattr(args, name)
    }
    if changed:
        _logger.warning(
            "%s was trained with another return range: %s; pass %s to keep it",
            args.load_checkpoint,
            ", ".join(
                f"{name}={value} (now {getattr(args, name)})" for name, value in changed.items()
            ),
            " ".join(f"--{name.replace('_', '-')} {value}" for name, value in changed.items()),
        )


def _resumes_online_stage(agent: Any, args: Off2OnCommonArgs) -> bool:
    return args.load_checkpoint is not None and agent._online_start_step is not None


def _replay_mode_of_saving_run(
    checkpoint: Mapping[str, Any],
) -> tuple[str, float | str] | None:
    """The online replay mode and offline data ratio of the run that wrote
    ``checkpoint``, from its run configuration. ``None`` when it has none, or
    when that run loaded a checkpoint itself: it may have continued in another
    mode than the one it was given."""
    run_config = checkpoint["metadata"].get("run_config") or {}
    inputs = run_config.get("inputs", {})
    if inputs.get("online_replay_mode") is None or inputs.get("load_checkpoint") is not None:
        return None
    mode = inputs["online_replay_mode"]
    return mode, (inputs["offline_data_ratio"] if mode == "mixed" else 0.0)


def _reconcile_with_saved_stage(
    agent: Any, args: Off2OnCommonArgs, checkpoint: Mapping[str, Any]
) -> None:
    """Make an agent loaded in its online stage usable with ``args``, whose
    stage settings the checkpoint overrides: one the user set to something
    else is an error, a default is reported. A checkpoint that does not record
    its stage continues in the mode of the run that wrote it, else in the one
    ``args`` states explicitly."""
    if args.num_offline_steps > 0:
        raise SystemExit(
            f"{args.load_checkpoint} is a checkpoint of the online stage and cannot train "
            "offline any more; pass --num-offline-steps 0."
        )
    if agent.online_replay_mode is None:
        saving_run_mode = _replay_mode_of_saving_run(checkpoint)
        if saving_run_mode is not None:
            agent.declare_online_replay_mode(*saving_run_mode)
        else:
            needed = ["online_replay_mode"]
            if args.online_replay_mode == "mixed":
                needed.append("offline_data_ratio")
            if not all(_is_explicit(args, name) for name in needed):
                raise SystemExit(
                    f"{args.load_checkpoint} is an online-stage checkpoint that does not record "
                    "its replay mode; pass --online-replay-mode (and, for mixed, "
                    "--offline-data-ratio) as the run that wrote it used."
                )
            agent.declare_online_replay_mode(args.online_replay_mode, args.offline_data_ratio)

    saved = {"online_replay_mode": agent.online_replay_mode}
    if agent.online_replay_mode == "mixed":
        saved["offline_data_ratio"] = agent.offline_data_ratio
    if hasattr(agent, "online_use_cql_loss"):
        saved["online_use_cql_loss"] = agent.use_cql_loss
        saved["online_cql_alpha"] = agent.cql_alpha
    if hasattr(agent, "online_discount"):
        saved["online_discount"] = agent.gamma
    if agent._replaced_initial_training_phase is not None or (
        agent.initial_training_phase is not None
    ):
        phase = agent.initial_training_phase
        saved["warmup_steps"] = 0 if phase is None else phase.duration_steps
    differing = {
        name: value for name, value in saved.items() if getattr(args, name) != value
    }
    rejected = [name for name in differing if _is_explicit(args, name)]
    if rejected:
        raise SystemExit(
            f"{args.load_checkpoint} continues the online stage it was saved in, which "
            "contradicts: "
            + ", ".join(
                f"--{name.replace('_', '-')} {getattr(args, name)} (saved: {differing[name]})"
                for name in rejected
            )
        )
    if differing:
        _logger.warning(
            "continuing the saved online stage with %s",
            ", ".join(f"{name}={value}" for name, value in differing.items()),
        )


def _why_offline_dataset_is_not_loaded(agent: Any) -> str | None:
    if agent.online_replay_mode == "empty":
        return "the 'empty' online stage discards offline data"
    if agent.online_replay_mode == "mixed" and agent.offline_replay_buffer is not None:
        return "the offline buffer is as the checkpoint was saved with it"
    if agent.online_replay_mode == "append" and agent.replay_buffer_restored:
        return "the restored replay buffer already holds the offline data"
    return None


def _restore_offline_data(agent: Any, args: Off2OnCommonArgs, logger: Logger) -> None:
    """Give a resumed online stage the offline data its replay mode keeps,
    unless the checkpoint's buffers already hold it."""
    not_loaded_because = _why_offline_dataset_is_not_loaded(agent)
    if not_loaded_because is not None:
        if args.offline_dataset is not None:
            _logger.warning("--offline-dataset is not loaded: %s", not_loaded_because)
        if agent.offline_replay_buffer is not None:
            _set_offline_probe(agent, logger, args.std_log, agent.offline_replay_buffer)
        return
    if args.offline_dataset is None:
        if agent.lacks_offline_data:
            raise SystemExit(
                f"{args.load_checkpoint} continues the 'mixed' online stage, which needs its "
                "offline data: pass "
                "--load-replay-buffer if the run saved offline_buffer_<suffix>.pt, or "
                "--offline-dataset."
            )
        if agent.online_replay_mode == "append":
            _logger.warning(
                "the 'append' online stage continues without its replay buffer: any offline data "
                "it held is gone (pass --load-replay-buffer or --offline-dataset)"
            )
        return
    buffer = (
        agent.restore_offline_replay_buffer()
        if agent.online_replay_mode == "mixed"
        else agent.replay_buffer
    )
    loaded = load_offline_dataset(buffer, args)
    logger.add_summary("off2on/offline_loaded_transitions", loaded)
    _set_offline_probe(agent, logger, args.std_log, buffer)
    if args.std_log:
        print(f"[online] offline data={loaded} transitions from the dataset", flush=True)


def _resume_online_stage(agent: Any, args: Off2OnCommonArgs, logger: Logger) -> None:
    _restore_offline_data(agent, args, logger)
    if args.offline_dataset is not None and hasattr(agent, "pretrain_vae"):
        offline_data = (
            agent.offline_replay_buffer
            if agent.online_replay_mode == "mixed"
            else agent.replay_buffer
        )
        if agent.online_replay_mode != "empty" and len(offline_data) > 0:
            agent.pretrain_vae()
    if getattr(args, "online_eval_freq", None) is None and _is_explicit(args, "eval_freq"):
        agent.eval_freq = args.eval_freq
    if args.std_log:
        print(
            f"[online] resumed replay_mode={agent.online_replay_mode} "
            f"offline_data_ratio={agent.offline_data_ratio} "
            f"online_buffer={len(agent.replay_buffer)}",
            flush=True,
        )


def _apply_online_eval_freq(agent: Any, args: Off2OnCommonArgs) -> None:
    """Switch ``agent.eval_freq`` to the online (env-step-unit) cadence.

    ``OffPolicyAlgorithm.learn()`` (the online rollout loop) reads
    ``agent.eval_freq`` directly, in env-step units -- distinct from the
    offline phase's gradient-update-unit cadence (see
    ``_offline_update_loop``'s ``eval_freq`` param). A missing/``None``
    ``online_eval_freq`` (e.g. non-Cal-QL off2on algorithms, which don't
    expose this field) leaves ``agent.eval_freq`` as already configured.
    """
    online_eval_freq = getattr(args, "online_eval_freq", None)
    if online_eval_freq is not None:
        agent.eval_freq = online_eval_freq


def run_off2on(
    args: Off2OnCommonArgs, *, build_agent: BuildAgent, algorithm: str
) -> None:
    resources: list[Any] = []
    try:
        _run_off2on(
            args,
            build_agent=build_agent,
            algorithm=algorithm,
            resources=resources,
        )
    finally:
        for resource in reversed(resources):
            resource.close()


def _run_off2on(
    args: Off2OnCommonArgs,
    *,
    build_agent: BuildAgent,
    algorithm: str,
    resources: list[Any],
) -> None:
    if not has_config_session():
        from rl_garden.training.off2on.registry import registry

        registry.entry_for_args(args)
        normalized_args, preflight = prepare_standalone(
            args,
            registry=registry,
            training_phase="off2on",
            algorithm=algorithm,
        )
        with config_session(preflight, dry_run=False):
            return _run_off2on(
                normalized_args,
                build_agent=build_agent,
                algorithm=algorithm,
                resources=resources,
            )

    seed_everything(args.seed)
    enable_fast_math()

    num_eval_episodes = getattr(args, "num_eval_episodes", None)
    warn_if_eval_budget_undersized(
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=num_eval_episodes,
        eval_episode_horizon=args.eval_episode_horizon,
    )

    # getattr: a handful of off2on algorithms (AWAC, SPOT) are Box-only
    # and never mix in ObservationArgs at all, so args has no .obs/.encoder.
    obs = getattr(args, "obs", None)
    is_visual = obs.is_visual if obs is not None else False
    obs_label = f"rgbd_{args.encoder.backbone}" if is_visual else "state"
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__{algorithm}_{obs_label}__{args.seed}__{int(time.time())}"
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
            log_group=args.log_group or args.env_id,
        )
    resources.append(logger)
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    if should_create_eval_env(args) and args.num_eval_envs <= 0:
        raise SystemExit(
            "--eval_freq, --offline_eval_freq, or --online_eval_freq > 0 requires "
            "--num_eval_envs > 0 to provide an eval environment."
        )

    req = make_env_request(args, run_name)
    if dry_run:
        req = replace(req, capture_video=False, eval_record_dir=None)
    env, eval_env = make_training_envs(args.env_backend, req)
    resources.append(env)
    if eval_env is not None:
        resources.append(eval_env)
    _require_continuous_action_space(env, args)

    agent = build_agent(args, env, eval_env, logger, checkpoint_dir)
    resumes_online_stage = _resumes_online_stage(agent, args)
    if args.load_checkpoint is not None:
        checkpoint = checkpoint_to_restore(args)
        _report_changed_value_range(args, checkpoint)
        if resumes_online_stage:
            _reconcile_with_saved_stage(agent, args, checkpoint)
    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
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
    materialized = materialize_config(
        env_request=req,
        env=env,
        eval_env=eval_env,
        agent=agent,
        derived=materialized_derived,
    )
    persist_effective_config(materialized, config_path)
    agent.run_config = json_value(materialized)
    logger.update_config(agent.run_config)

    # Offline training phase
    if args.num_offline_steps > 0:
        if args.offline_dataset is None and not _holds_restored_offline_stage_data(agent, args):
            raise SystemExit("--offline_dataset is required when --num_offline_steps > 0.")
        _load_offline_stage_dataset(agent, args, logger)
        offline_start_step = agent._global_step
        logger.add_summary("off2on/offline_start_step", offline_start_step)
        _prepare_dataset(agent, args)
        _offline_update_loop(
            agent,
            args.num_offline_steps,
            logger,
            args.log_freq,
            args.std_log,
            eval_freq=getattr(args, "offline_eval_freq", None),
        )
        offline_end_step = offline_start_step + args.num_offline_steps
        _evaluate_offline_end(agent, logger, offline_end_step, args.std_log)
        _save_offline_checkpoint(
            agent,
            checkpoint_dir,
            include_replay_buffer=args.save_replay_buffer,
            std_log=args.std_log,
        )
        _set_offline_probe(agent, logger, args.std_log)

    if resumes_online_stage:
        _resume_online_stage(agent, args, logger)
    else:
        _switch_to_online_mode(agent, args, logger)
    _apply_online_eval_freq(agent, args)

    # Online training phase
    if args.num_online_steps > 0:
        online_target_step = agent._global_step + args.num_online_steps
        agent.learn(total_timesteps=online_target_step)
    elif agent.checkpoint_dir is not None and agent.save_final_checkpoint:
        agent._save_checkpoint("final.pt")
