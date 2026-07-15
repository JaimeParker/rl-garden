"""TD-MPC2 multitask offline pretraining run function.

Does not reuse ``rl_garden.training.offline._runner.run_offline``: that
runner assumes one homogeneous dataset via ``infer_offline_dataset_specs``/
``load_offline_dataset``, which doesn't fit "N tasks, each with its own
obs/action dimensionality, zero-padded to a shared max" (see
``rl_garden.algorithms.tdmpc2.multitask.dataset``). It does reuse the
generic ``run_offline_pretraining`` step loop (checkpointing/logging/
progress bar), since ``TDMPC2Multitask.train(gradient_steps)`` already
matches that loop's expected agent interface.
"""
from __future__ import annotations

import time

from rl_garden.common import Logger, seed_everything
from rl_garden.common.resolved_config import persist_resolved_config


def run_tdmpc2_multitask(args: "TDMPC2MultitaskArgs") -> None:
    from gymnasium import spaces

    from rl_garden.algorithms.offline import OfflineEnvSpec, run_offline_pretraining
    from rl_garden.algorithms.tdmpc2.multitask import TDMPC2Multitask
    from rl_garden.algorithms.tdmpc2.multitask.dataset import (
        infer_multitask_dataset_specs,
        load_multitask_dataset,
    )

    if not args.dataset_dir:
        raise SystemExit("--dataset_dir is required for tdmpc2_multitask.")
    if not args.mmap_dir:
        raise SystemExit("--mmap_dir is required for tdmpc2_multitask.")

    seed_everything(args.seed)

    tasks, obs_dims, action_dims, episode_lengths = infer_multitask_dataset_specs(args.dataset_dir)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"tdmpc2_multitask__{args.seed}__{int(time.time())}"
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.save_final_checkpoint or args.checkpoint_freq > 0:
        import os

        checkpoint_dir = os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args,
        training_phase="offline",
        algorithm="tdmpc2_multitask",
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
        wandb_group=args.wandb_group or "tdmpc2_multitask",
    )

    env = OfflineEnvSpec(
        observation_space=spaces.Box(low=-1e6, high=1e6, shape=(max(obs_dims),), dtype="float32"),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(max(action_dims),), dtype="float32"),
        num_envs=1,
    )
    agent = TDMPC2Multitask(
        env=env,
        tasks=tasks,
        obs_dims=obs_dims,
        action_dims=action_dims,
        episode_lengths=episode_lengths,
        mmap_dir=args.mmap_dir,
        mmap_mode="create",
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        task_dim=args.task_dim,
        latent_dim=args.latent_dim,
        enc_dim=args.enc_dim,
        num_enc_layers=args.num_enc_layers,
        mlp_dim=args.mlp_dim,
        simnorm_dim=args.simnorm_dim,
        num_q=args.num_q,
        num_bins=args.num_bins,
        vmin=args.vmin,
        vmax=args.vmax,
        dropout=args.dropout,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        enc_lr_scale=args.enc_lr_scale,
        grad_clip_norm=args.grad_clip_norm,
        tau=args.tau,
        rho=args.rho,
        consistency_coef=args.consistency_coef,
        reward_coef=args.reward_coef,
        value_coef=args.value_coef,
        discount_denom=args.discount_denom,
        discount_min=args.discount_min,
        discount_max=args.discount_max,
        seed=args.seed,
        device=args.device,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )

    loaded = load_multitask_dataset(agent.replay_buffer, args.dataset_dir)
    logger.add_summary("offline/loaded_transitions", loaded)
    if args.std_log:
        print(f"[tdmpc2_multitask] tasks={tasks} loaded_transitions={loaded}", flush=True)

    if args.load_checkpoint is not None:
        # The mmap buffer is always repopulated from --dataset_dir above, not
        # from a checkpoint snapshot -- MmapMultitaskEpisodeBuffer doesn't
        # implement the generic replay-buffer-file protocol.
        agent.load(args.load_checkpoint, load_replay_buffer=False)

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="tdmpc2_multitask_offline_pretrained.pt",
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="tdmpc2-multitask-offline",
    )

    logger.close()


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.training.offline._args import TDMPC2MultitaskTrainingArgs  # noqa: E402
from rl_garden.training.offline._registry import registry  # noqa: E402


@dataclass
class TDMPC2MultitaskArgs(TDMPC2MultitaskTrainingArgs):
    """TD-MPC2 multitask offline pretraining. Requires ``--dataset_dir``
    (output of ``tools/conversion/convert_tdmpc2_multitask_dataset.py``) and
    ``--mmap_dir`` (a fresh directory for the training-time mmap buffer)."""


registry.register("tdmpc2_multitask", TDMPC2MultitaskArgs, run_tdmpc2_multitask)
