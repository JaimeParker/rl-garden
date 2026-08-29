"""A2A flow-matching BC pretraining run function.

Standalone sibling of ``training/offline/vision_diffusion_bc.py`` (not built
on it) -- mirrors its shape for the same reason: no replay buffer, dataset is
loaded directly in the constructor, so ``training/offline/_runner.py::run_offline``
doesn't apply.
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
from rl_garden.training.offline._args import A2ABCTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class A2ABCArgs(A2ABCTrainingArgs):
    """A2A flow-matching BC pretraining. Requires ``--dataset_path`` (H5
    trajectory file with nested ``obs/<key>`` groups, e.g. ``obs/rgb``/
    ``obs/state``). Vision conditioning is mandatory -- unlike
    ``vision_diffusion_bc``, ``--include_state=False`` is rejected: the
    state-history window is the flow's source, not optional."""


def run_a2a_bc(args: A2ABCArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_a2a_bc(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_a2a_bc(args: A2ABCArgs, cleanup: list[Callable[[], None]]) -> None:
    from rl_garden.algorithms import A2ABC, OfflineEnvSpec
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.encoders import discover_image_keys
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="a2a_bc"
        )
        with config_session(preflight, dry_run=False):
            return _run_a2a_bc(normalized_args, cleanup)

    if not args.dataset_path:
        raise SystemExit("--dataset_path is required for a2a_bc.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")
    if args.include_state is False:
        raise SystemExit(
            "a2a_bc requires include_state=True -- the state-history window "
            "is the flow's source, not optional."
        )

    seed_everything(args.seed)

    obs_space, action_space = infer_specs_from_h5(args.dataset_path)
    if not isinstance(obs_space, spaces.Dict):
        raise SystemExit(
            "a2a_bc requires a Dict-shaped H5 dataset (nested obs/<key> "
            "groups); got a flat Box-shaped obs array."
        )

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"a2a_bc__{args.seed}__{int(time.time())}"
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
            log_group=args.log_group or "a2a_bc",
        )
    cleanup.append(logger.close)

    env = OfflineEnvSpec(observation_space=obs_space, action_space=action_space, num_envs=1)
    image_keys = discover_image_keys(obs_space)
    agent = construct_agent(
        A2ABC,
        env=env,
        dataset_path=args.dataset_path,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
        latent_dim=args.latent_dim,
        cnn_num_layers=args.cnn_num_layers,
        cnn_hidden_channels=args.cnn_hidden_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_activation_fn=args.activation_fn,
        decoder_net_arch=args.decoder_net_arch,
        flow_hidden_dims=args.flow_hidden_dims,
        num_sampling_steps=args.num_sampling_steps,
        consistency_weight=args.consistency_weight,
        enc_recon_weight=args.enc_recon_weight,
        flow_recon_weight=args.flow_recon_weight,
        enc_contrastive_weight=args.enc_contrastive_weight,
        flow_contrastive_weight=args.flow_contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        image_encoder_factory=image_encoder_factory_from_args(args),
        image_keys=image_keys,
        state_key="state",
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
            print(f"[a2a_bc] resumed_from={args.load_checkpoint}", flush=True)

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
            f"[a2a_bc] dataset_size={agent._dataset_size} "
            f"image_keys={image_keys} action={action_space.shape}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="a2a_bc_offline_pretrained.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="a2a-bc-offline",
    )


registry.register("a2a_bc", A2ABCArgs, run_a2a_bc)
