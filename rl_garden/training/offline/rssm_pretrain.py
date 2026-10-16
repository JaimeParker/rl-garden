"""Reward-free RSSM pretraining run function (NCRL stage 1).

Bespoke runner like ``tdmpc2_multitask``: the generic ``run_offline`` assumes
one homogeneous ``offline_dataset``, while this reads a converted image
sequence dataset (``tools/conversion/convert_ncrl_dataset.py``) whose
observation/action spaces come from its manifests. It reuses the generic
``run_offline_pretraining`` step loop.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

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
from rl_garden.training.offline._args import RSSMPretrainTrainingArgs
from rl_garden.training.offline._registry import registry


def run_rssm_pretrain(args: RSSMPretrainTrainingArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_rssm_pretrain(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_rssm_pretrain(args: RSSMPretrainTrainingArgs, cleanup: list[Callable[[], None]]) -> None:
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.algorithms.rssm_pretrain import RSSMPretrain, sequence_dataset_env_spec
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="rssm_pretrain"
        )
        with config_session(preflight, dry_run=False):
            return _run_rssm_pretrain(normalized_args, cleanup)

    if not args.dataset_dir:
        raise SystemExit("--dataset_dir is required for rssm_pretrain.")

    seed_everything(args.seed)
    env = sequence_dataset_env_spec(args.dataset_dir, args.padded_action_dim)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"rssm_pretrain__{args.seed}__{int(time.time())}"
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and (args.save_final_checkpoint or args.checkpoint_freq > 0):
        checkpoint_dir = os.path.join(args.log_dir, run_name, "checkpoints")

    dry_run = is_dry_run()
    if dry_run:
        logger = Logger(log_type="none")
    else:
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
            log_group=args.log_group or "rssm_pretrain",
        )
    cleanup.append(logger.close)

    agent = construct_agent(
        RSSMPretrain,
        env=env,
        dataset_dir=args.dataset_dir,
        dataset_backend=args.dataset_backend,
        backbone=args.backbone,
        size=args.size,
        deter=args.deter,
        hidden=args.hidden,
        classes=args.classes,
        cnn_depth=args.cnn_depth,
        stoch=args.stoch,
        blocks=args.blocks,
        kl_free=args.kl_free,
        kl_free_mode=args.kl_free_mode,
        dyn_scale=args.dyn_scale,
        rep_scale=args.rep_scale,
        recon_scale=args.recon_scale,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        lr=args.lr,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        compute_dtype=args.compute_dtype,
        buffer_device=args.buffer_device,
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
        # The dataset is re-read from --dataset_dir, never checkpointed.
        agent.load(args.load_checkpoint, load_replay_buffer=False)

    materialized_derived = {"run_name": run_name, "checkpoint_dir": checkpoint_dir}
    env_request = {"dataset_dir": args.dataset_dir}
    if dry_run:
        emit_materialized_config(
            env_request=env_request, env=env, eval_env=None, agent=agent, derived=materialized_derived
        )
        return
    materialized = materialize_config(
        env_request=env_request, env=env, eval_env=None, agent=agent, derived=materialized_derived
    )
    persist_effective_config(materialized, config_path)
    logger.update_config(json_value(materialized))

    loaded = agent.load_dataset()
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(
            f"[rssm_pretrain] backend={args.dataset_backend} steps={loaded} "
            f"padded_action_dim={agent.padded_action_dim}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="rssm_pretrain_offline_pretrained.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="rssm-pretrain-offline",
    )


registry.register("rssm_pretrain", RSSMPretrainTrainingArgs, run_rssm_pretrain)
