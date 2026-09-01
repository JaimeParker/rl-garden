"""HILP pretraining run function.

Does not reuse ``rl_garden.training.offline._runner.run_offline``: that
runner populates ``agent.replay_buffer`` via ``load_offline_dataset``, but
``HILP`` has no replay buffer -- its dataset is a ``HindsightGoalDataset``
loaded directly in the constructor. Mirrors ``diffusion_bc.py``'s bespoke
non-replay-buffer shape.
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
from rl_garden.training.offline._args import HILPTrainingArgs
from rl_garden.training.offline._registry import registry


@dataclass
class HILPArgs(HILPTrainingArgs):
    """HILP pretraining. Requires ``--dataset_path`` (H5 trajectory file,
    state-only). Produces a phi-representation + skill-value/critic/actor
    checkpoint."""


def run_hilp(args: HILPArgs) -> None:
    cleanup: list[Callable[[], None]] = []
    try:
        _run_hilp(args, cleanup)
    finally:
        for callback in reversed(cleanup):
            callback()


def _run_hilp(args: HILPArgs, cleanup: list[Callable[[], None]]) -> None:
    from rl_garden.algorithms import HILP, OfflineEnvSpec
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.training.inspection import construct_agent

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="hilp"
        )
        with config_session(preflight, dry_run=False):
            return _run_hilp(normalized_args, cleanup)

    if not args.dataset_path:
        raise SystemExit("--dataset_path is required for hilp.")
    if args.num_offline_steps <= 0:
        raise SystemExit("--num_offline_steps must be positive.")

    seed_everything(args.seed)

    obs_space, action_space = infer_box_specs_from_h5(args.dataset_path)

    run_name = args.exp_name or f"hilp__{args.seed}__{int(time.time())}"
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
        start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
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
            log_group=args.log_group or "hilp",
        )
    cleanup.append(logger.close)

    env = OfflineEnvSpec(observation_space=obs_space, action_space=action_space, num_envs=1)
    agent = construct_agent(
        HILP,
        env=env,
        dataset_path=args.dataset_path,
        skill_dim=args.skill_dim,
        value_hidden_dims=args.value_hidden_dims,
        actor_hidden_dims=args.actor_hidden_dims,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        skill_expectile=args.skill_expectile,
        skill_temperature=args.skill_temperature,
        skill_discount=args.skill_discount,
        p_currgoal=args.p_currgoal,
        p_trajgoal=args.p_trajgoal,
        p_randomgoal=args.p_randomgoal,
        lr=args.lr,
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
            print(f"[hilp] resumed_from={args.load_checkpoint}", flush=True)

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
            f"[hilp] dataset_size={agent._dataset.size} "
            f"obs={obs_space.shape} action={action_space.shape}",
            flush=True,
        )

    run_offline_pretraining(
        agent,
        num_steps=args.num_offline_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_filename="hilp_offline_pretrained.pt",
        save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint,
        log_freq=args.log_freq,
        std_log=args.std_log,
        eval_freq=0,
        desc="hilp-offline",
    )


registry.register("hilp", HILPArgs, run_hilp)
