"""Vision-conditioned Diffusion BC pretraining run function.

Standalone sibling of ``training/offline/diffusion_bc.py`` (not built on it,
mirrors its shape) -- same reasoning as ``VisionDiffusionBC`` itself being a
standalone sibling of ``DiffusionBC``. Also does not reuse
``training/offline/_runner.py::run_offline``, for the same reason
``diffusion_bc.py`` doesn't: no replay buffer, dataset is loaded directly in
the constructor.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gymnasium import spaces

from rl_garden.buffers.h5_dataset import infer_specs_from_h5
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
from rl_garden.training.offline._args import VisionDiffusionBCTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class VisionDiffusionBCArgs(VisionDiffusionBCTrainingArgs):
    """Vision-conditioned Diffusion BC pretraining. Requires ``--dataset_path``
    (H5 trajectory file with nested ``obs/<key>`` groups, e.g. ``obs/rgb``/
    ``obs/state``). Produces the same EMA checkpoint shape ``DiffusionBC``
    does, but sized for a vision encoder's conditioning -- not compatible
    with ``DPPO``'s ``--bc_checkpoint`` (that path is ``DiffusionBC``'s
    state-only contract)."""


def run_vision_diffusion_bc(args: VisionDiffusionBCArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_vision_diffusion_bc(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_vision_diffusion_bc(args: VisionDiffusionBCArgs, cleanup: list[Callable[[], None]]) -> None:
    from rl_garden.algorithms import OfflineEnvSpec, VisionDiffusionBC
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.encoders import discover_image_keys
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="vision_diffusion_bc"
        )
        with config_session(preflight, dry_run=False):
            return _run_vision_diffusion_bc(normalized_args, cleanup)

    if not args.dataset_path:
        raise SystemExit("--dataset_path is required for vision_diffusion_bc.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")

    seed_everything(args.seed)

    obs_space, action_space = infer_specs_from_h5(args.dataset_path)
    if not isinstance(obs_space, spaces.Dict):
        raise SystemExit(
            "vision_diffusion_bc requires a Dict-shaped H5 dataset (nested "
            "obs/<key> groups); got a flat Box-shaped obs array. Use "
            "diffusion_bc for Box datasets."
        )

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"vision_diffusion_bc__{args.seed}__{int(time.time())}"
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
            log_group=args.log_group or "vision_diffusion_bc",
        )
    cleanup.append(logger.close)

    env = OfflineEnvSpec(observation_space=obs_space, action_space=action_space, num_envs=1)
    image_keys = discover_image_keys(obs_space)
    agent = construct_agent(
        VisionDiffusionBC,
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
        image_encoder_factory=image_encoder_factory_from_args(args),
        image_keys=image_keys,
        state_key="state",
        use_proprio=args.include_state,
        image_fusion_mode=args.image_fusion_mode,
        enable_stacking=False,
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
            print(f"[vision_diffusion_bc] resumed_from={args.load_checkpoint}", flush=True)

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
            f"[vision_diffusion_bc] dataset_size={agent._dataset_size} "
            f"image_keys={image_keys} action={action_space.shape}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="vision_diffusion_bc_offline_pretrained.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="vision-diffusion-bc-offline",
    )


registry.register("vision_diffusion_bc", VisionDiffusionBCArgs, run_vision_diffusion_bc)
