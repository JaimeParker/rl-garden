"""PWM offline policy-learning run function (bespoke runner, same shape as
``rl_garden.training.offline.tdmpc2_multitask``).

Reads the converted TD-MPC2 multitask dataset's manifest, loads only the
``--task`` episodes into the mmap buffer, loads the frozen world model from
``--world_model_checkpoint`` (a ``TDMPC2Multitask`` checkpoint or
``tools/conversion/convert_tdmpc2_checkpoint.py`` output) and checks it
against the dataset's task table. Optional evaluation builds
``--eval_env_backend`` with ``env_id=--task`` wrapped in
``MultitaskPadWrapper``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from rl_garden.common import Logger, seed_everything
from rl_garden.common.effective_config import json_value, persist_effective_config
from rl_garden.training.inspection import (
    config_session,
    emit_materialized_config,
    has_config_session,
    is_dry_run,
    materialize_config,
    prepare_standalone,
    run_preflight,
)
from rl_garden.training.offline._args import PWMTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class PWMArgs(PWMTrainingArgs):
    """PWM offline policy learning. Requires ``--dataset_dir`` (converted
    TD-MPC2 multitask dataset), ``--mmap_dir`` (fresh directory for the
    training-time mmap buffer), ``--world_model_checkpoint`` and ``--task``
    (a task name from the dataset manifest). State-only."""


def run_pwm(args: PWMArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_pwm(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _task_buffer_size(dataset_dir: str, task: str, manifest_index: int) -> int:
    import json

    import torch

    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    if "num_episodes" in manifest:
        return int(manifest["num_episodes"][manifest_index]) * int(manifest["episode_lengths"][manifest_index])
    data = torch.load(Path(dataset_dir) / task / "episodes.pt", weights_only=True)
    return int(data["obs"].shape[0] * data["obs"].shape[1])


def load_task_episodes(buffer, dataset_dir: str, task: str, task_idx: int, obs_dim: int, action_dim: int) -> int:
    """Zero-pads one task's episodes to the buffer's dims and bulk-loads them
    (``load_multitask_dataset`` restricted to a single task). Returns the
    number of transitions loaded."""
    import torch

    data = torch.load(Path(dataset_dir) / task / "episodes.pt", weights_only=True)
    obs, action, reward = data["obs"], data["action"], data["reward"]
    if obs.shape[-1] != obs_dim or action.shape[-1] != action_dim:
        raise ValueError(
            f"Task {task!r} tensor shapes {tuple(obs.shape)}/{tuple(action.shape)} "
            f"don't match manifest obs_dim={obs_dim}/action_dim={action_dim}."
        )
    num_episodes, episode_length = obs.shape[0], obs.shape[1]
    obs_padded = torch.zeros(num_episodes, episode_length, buffer.obs_dim)
    obs_padded[..., :obs_dim] = obs
    action_padded = torch.zeros(num_episodes, episode_length, buffer.action_dim)
    action_padded[..., :action_dim] = action
    for ep in range(num_episodes):
        buffer.load_episode(obs_padded[ep], action_padded[ep], reward[ep], task_idx)
    return num_episodes * episode_length


def _make_eval_env(args: PWMArgs, *, obs_dim: int, action_dim: int, task_idx: int,
                   task_obs_dim: int, task_action_dim: int, num_eval_steps: int):
    from rl_garden.envs.backend_registry import EnvRequest, make_evaluation_env
    from rl_garden.envs.wrappers.multitask_pad import MultitaskPadWrapper
    from rl_garden.observations.config import ObservationConfig

    request = EnvRequest(
        env_id=args.task,
        num_envs=1,
        control_mode="",
        render_mode="rgb_array",
        seed=args.seed,
        observation=ObservationConfig(state=True),
        num_eval_envs=args.eval_runs,
        capture_video=False,
        num_eval_steps=num_eval_steps,
    )
    env = make_evaluation_env(args.eval_env_backend, request)
    return MultitaskPadWrapper(
        env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        task_idx=task_idx,
        task_obs_dim=task_obs_dim,
        task_action_dim=task_action_dim,
    )


def _run_pwm(args: PWMArgs, cleanup: list[Callable[[], None]]) -> None:
    from gymnasium import spaces

    from rl_garden.algorithms.offline import OfflineEnvSpec, run_offline_pretraining
    from rl_garden.algorithms.pwm import PWM
    from rl_garden.buffers.tdmpc2_multitask_dataset import infer_multitask_dataset_specs
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="pwm"
        )
        with config_session(preflight, dry_run=False):
            return _run_pwm(normalized_args, cleanup)

    for flag in ("dataset_dir", "mmap_dir", "world_model_checkpoint", "task"):
        if not getattr(args, flag):
            raise SystemExit(f"--{flag} is required for pwm.")

    seed_everything(args.seed)
    tasks, obs_dims, action_dims, episode_lengths = infer_multitask_dataset_specs(args.dataset_dir)
    if args.task not in tasks:
        raise SystemExit(f"--task {args.task!r} is not in the dataset's tasks {tasks}.")
    task_idx = tasks.index(args.task)
    buffer_size = _task_buffer_size(args.dataset_dir, args.task, task_idx)
    # Generous cap: eval_runs episodes run in parallel, one episode each.
    num_eval_steps = 2 * episode_lengths[task_idx]

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"pwm__{args.task}__{args.seed}__{int(time.time())}"
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.save_final_checkpoint or args.checkpoint_freq > 0:
        checkpoint_dir = str(Path(args.log_dir) / run_name / "checkpoints")

    dry_run = is_dry_run()
    if dry_run:
        logger = Logger(log_type="none")
        temporary_mmap = TemporaryDirectory(prefix="rlg-dry-run-")
        cleanup.append(temporary_mmap.cleanup)
        mmap_dir = temporary_mmap.name
    else:
        mmap_dir = args.mmap_dir
        config_path = Path(args.log_dir) / run_name / "config.json"
        preflight_config = run_preflight({"run_name": run_name, "checkpoint_dir": checkpoint_dir})
        persist_effective_config(preflight_config, config_path)
        logger = Logger.create(
            log_type=args.log_type,
            log_dir=args.log_dir,
            run_name=run_name,
            config=json_value(preflight_config),
            start_time=start_time,
            log_keywords=args.log_keywords,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            log_group=args.log_group or "pwm",
        )
    cleanup.append(logger.close)

    obs_dim, action_dim = max(obs_dims), max(action_dims)
    env = OfflineEnvSpec(
        observation_space=spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype="float32"),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype="float32"),
        num_envs=1,
    )
    eval_env = None
    if not dry_run and args.eval_freq > 0:
        eval_env = _make_eval_env(
            args,
            obs_dim=obs_dim,
            action_dim=action_dim,
            task_idx=task_idx,
            task_obs_dim=obs_dims[task_idx],
            task_action_dim=action_dims[task_idx],
            num_eval_steps=num_eval_steps,
        )
        cleanup.append(eval_env.close)

    agent = construct_agent(
        PWM,
        env=env,
        world_model_checkpoint=args.world_model_checkpoint,
        task_idx=task_idx,
        mmap_dir=mmap_dir,
        mmap_mode="create",
        buffer_size=buffer_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        gamma=args.gamma,
        lam=args.lam,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        lr_final=args.lr_final,
        lr_decay_steps=args.num_offline_steps,
        actor_grad_norm=args.actor_grad_norm,
        critic_grad_norm=args.critic_grad_norm,
        critic_iterations=args.critic_iterations,
        critic_batches=args.critic_batches,
        num_critics=args.num_critics,
        actor_units=args.actor_units,
        critic_units=args.critic_units,
        init_logstd=args.init_logstd,
        min_logstd=args.min_logstd,
        ret_rms=args.ret_rms,
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_episodes=args.eval_runs,
        num_eval_steps=num_eval_steps,
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    from rl_garden.algorithms.pwm import validate_world_model_for_dataset

    validate_world_model_for_dataset(
        agent.bound_world_model.world_model,
        agent.world_model_source,
        tasks=tasks,
        obs_dims=obs_dims,
        action_dims=action_dims,
    )

    if args.load_checkpoint is not None:
        # The mmap buffer is repopulated from --dataset_dir, never checkpointed.
        agent.load(args.load_checkpoint, load_replay_buffer=False)

    env_request = {"dataset_dir": args.dataset_dir, "task": args.task, "eval_env_backend": args.eval_env_backend}
    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir, "task_idx": task_idx}
    if dry_run:
        emit_materialized_config(
            env_request=env_request, env=env, eval_env=None, agent=agent, derived=materialized_derived
        )
        return
    materialized = materialize_config(
        env_request=env_request, env=env, eval_env=eval_env, agent=agent, derived=materialized_derived
    )
    persist_effective_config(materialized, config_path)
    logger.update_config(json_value(materialized))

    loaded = load_task_episodes(
        agent.replay_buffer, args.dataset_dir, args.task, task_idx, obs_dims[task_idx], action_dims[task_idx]
    )
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[pwm] task={args.task} task_idx={task_idx} loaded_transitions={loaded}", flush=True)

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="pwm_offline.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=args.eval_freq if eval_env is not None else 0,
        desc="pwm-offline",
    )


def _pwm_algorithm_cls() -> type:
    from rl_garden.algorithms.pwm import PWM

    return PWM


registry.register("pwm", PWMArgs, run_pwm, algorithm_cls=_pwm_algorithm_cls)
