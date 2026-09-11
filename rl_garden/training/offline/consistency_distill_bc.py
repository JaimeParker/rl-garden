"""ConsistencyDistillBC run function: fully offline LCM-style consistency
distillation of a frozen ``DiffusionBC`` teacher into a one/few-step student.

Does not reuse ``rl_garden.training.offline._runner.run_offline`` for the
same reason ``training/offline/diffusion_bc.py`` doesn't: ``ConsistencyDistillBC``
has no replay buffer -- its dataset is a fixed set of ``(obs_history,
action_chunk)`` windows loaded directly in the constructor. Mirrors that
module's bespoke-runner shape exactly.
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
from rl_garden.training.offline._args import ConsistencyDistillBCTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class ConsistencyDistillBCArgs(ConsistencyDistillBCTrainingArgs):
    """Offline consistency distillation of a frozen ``DiffusionBC`` teacher.
    Requires ``--dataset_path`` (H5 trajectory file, state-only, same
    dataset the teacher was trained on) and ``--bc_checkpoint`` (a
    ``diffusion_bc`` checkpoint -- ``horizon_steps``/``cond_steps``/
    ``net_backbone``/``unet_*`` must match what that checkpoint was trained
    with)."""


def run_consistency_distill_bc(args: ConsistencyDistillBCArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_consistency_distill_bc(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_consistency_distill_bc(
    args: ConsistencyDistillBCArgs, cleanup: list[Callable[[], None]]
) -> None:
    from rl_garden.algorithms import ConsistencyDistillBC, OfflineEnvSpec
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="consistency_distill_bc"
        )
        with config_session(preflight, dry_run=False):
            return _run_consistency_distill_bc(normalized_args, cleanup)

    if not args.dataset_path:
        raise SystemExit("--dataset_path is required for consistency_distill_bc.")
    if not args.bc_checkpoint:
        raise SystemExit("--bc_checkpoint is required for consistency_distill_bc.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")

    seed_everything(args.seed)

    obs_space, action_space = infer_box_specs_from_h5(args.dataset_path)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"consistency_distill_bc__{args.seed}__{int(time.time())}"
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
            log_group=args.log_group or "consistency_distill_bc",
        )
    cleanup.append(logger.close)

    from rl_garden.networks import DiffusionMLP

    net_cls = DiffusionMLP
    net_kwargs = None
    if args.net_backbone == "unet":
        from rl_garden.networks import DiffusionUNet1D

        net_cls = DiffusionUNet1D
        net_kwargs = dict(
            down_dims=args.unet_down_dims,
            kernel_size=args.unet_kernel_size,
            n_groups=args.unet_n_groups,
            cond_predict_scale=args.unet_cond_predict_scale,
        )

    env = OfflineEnvSpec(observation_space=obs_space, action_space=action_space, num_envs=1)
    agent = construct_agent(
        ConsistencyDistillBC,
        env=env,
        dataset_path=args.dataset_path,
        bc_checkpoint=args.bc_checkpoint,
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
        net_cls=net_cls,
        net_kwargs=net_kwargs,
        cm_lr=args.cm_lr,
        weight_decay=args.weight_decay,
        cm_ema_decay=args.cm_ema_decay,
        cm_grad_clip_norm=args.cm_grad_clip_norm,
        cm_sigma_data=args.cm_sigma_data,
        cm_timestep_scaling=args.cm_timestep_scaling,
        batch_size=args.batch_size,
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
            print(f"[consistency_distill_bc] resumed_from={args.load_checkpoint}", flush=True)

    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
    if dry_run:
        emit_materialized_config(
            env_request={"dataset_path": args.dataset_path, "bc_checkpoint": args.bc_checkpoint},
            env=env,
            eval_env=None,
            agent=agent,
            derived=materialized_derived,
        )
        return
    materialized = materialize_config(
        env_request={"dataset_path": args.dataset_path, "bc_checkpoint": args.bc_checkpoint},
        env=env,
        eval_env=None,
        agent=agent,
        derived=materialized_derived,
    )
    persist_effective_config(materialized, config_path)
    logger.update_config(json_value(materialized))
    if args.std_log:
        print(
            f"[consistency_distill_bc] dataset_size={agent._dataset_size} "
            f"obs={obs_space.shape} action={action_space.shape}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="consistency_distill_bc_offline.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="consistency-distill-bc-offline",
    )


registry.register("consistency_distill_bc", ConsistencyDistillBCArgs, run_consistency_distill_bc)
