"""Diffusion BC pretraining run function (DPPO phase 1).

Does not reuse ``rl_garden.training.offline._runner.run_offline``: that
runner populates ``agent.replay_buffer`` via ``load_offline_dataset``, but
``DiffusionBC`` has no replay buffer -- its dataset is a fixed set of
``(obs_history, action_chunk)`` windows loaded directly in the constructor
(see ``rl_garden.buffers.chunked_dataset``). Mirrors
``tdmpc2_multitask.py``'s bespoke-runner shape (same reasoning: a
non-replay-buffer dataset doesn't fit the shared runner), but simpler since
there is no separate dataset-loading step to run after agent construction.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rl_garden.buffers.h5_dataset import infer_box_specs_from_h5
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
from rl_garden.training.offline._args import DiffusionBCTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class DiffusionBCArgs(DiffusionBCTrainingArgs):
    """Diffusion BC pretraining (DPPO phase 1). Requires ``--dataset_path``
    (H5 trajectory file, state-only). Produces the EMA checkpoint that
    ``dppo``'s ``--bc_checkpoint`` loads into ``actor``/``actor_ft``."""


def run_diffusion_bc(args: DiffusionBCArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_diffusion_bc(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_diffusion_bc(args: DiffusionBCArgs, cleanup: list[Callable[[], None]]) -> None:
    from rl_garden.algorithms import DiffusionBC, OfflineEnvSpec
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="diffusion_bc"
        )
        with config_session(preflight, dry_run=False):
            return _run_diffusion_bc(normalized_args, cleanup)

    if not args.dataset_path:
        raise SystemExit("--dataset_path is required for diffusion_bc.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")

    seed_everything(args.seed)

    obs_space, action_space = infer_box_specs_from_h5(args.dataset_path)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"diffusion_bc__{args.seed}__{int(time.time())}"
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.save_final_checkpoint or args.checkpoint_freq > 0:
        checkpoint_dir = str(Path(args.log_dir) / run_name / "checkpoints")

    dry_run = is_dry_run()
    config_path = Path(args.log_dir) / run_name / "config.json"
    if dry_run:
        logger = Logger(log_type="none")
    else:
        preflight_config = run_preflight(
            {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
        )
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
            wandb_group=args.wandb_group or "diffusion_bc",
        )
    cleanup.append(logger.close)

    env = OfflineEnvSpec(observation_space=obs_space, action_space=action_space, num_envs=1)
    agent = construct_agent(
        DiffusionBC,
        env=env,
        dataset_path=args.dataset_path,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
        denoising_steps=args.denoising_steps,
        activation_fn=args.activation_fn,
        residual_style=args.residual_style,
        time_dim=args.time_dim,
        kernel_init=args.kernel_init,
        denoised_clip_value=args.denoised_clip_value,
        randn_clip_value=args.randn_clip_value,
        final_action_clip_value=args.final_action_clip_value,
        min_sampling_denoising_std=args.min_sampling_denoising_std,
        actor_lr=args.actor_lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        ema_start_step=args.ema_start_step,
        num_traj=args.offline_num_traj,
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
    )

    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
        if args.std_log:
            print(f"[diffusion_bc] resumed_from={args.load_checkpoint}", flush=True)

    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
    if dry_run:
        emit_materialized_config(
            env_request={"dataset_path": args.dataset_path},
            env=env,
            eval_env=None,
            agent=agent,
            derived=materialized_derived,
        )
        return
    materialized = materialize_config(
        env_request={"dataset_path": args.dataset_path},
        env=env,
        eval_env=None,
        agent=agent,
        derived=materialized_derived,
    )
    persist_effective_config(materialized, config_path)
    logger.update_config(json_value(materialized))
    if args.std_log:
        print(
            f"[diffusion_bc] dataset_size={agent._dataset_size} "
            f"obs={obs_space.shape} action={action_space.shape}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="diffusion_bc_offline_pretrained.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="diffusion-bc-offline",
    )


registry.register("diffusion_bc", DiffusionBCArgs, run_diffusion_bc)
