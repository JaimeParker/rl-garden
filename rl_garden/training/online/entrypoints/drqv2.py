"""DrQ-v2 run function."""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.ddpg import DDPGParams

import warnings
from typing import Literal


def build_drqv2(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.algorithms.ddpg import DDPG
    from rl_garden.training.inspection import construct_agent

    if args.encoder.backbone != "drqv2_conv":
        warnings.warn(
            f"DrQv2's validated default encoder is 'drqv2_conv'; overriding "
            f"with --encoder.backbone {args.encoder.backbone!r} deviates from "
            "the DrQ-v2 paper architecture.",
            stacklevel=2,
        )
    # DDPG requires image observations (see DDPG.__init__'s validated
    # default and _setup_model's "at least one image key" check) -- that
    # restriction is independent of critic_encoder_config/encoder_sharing,
    # which DDPG does support (a distinct/differently-backboned critic
    # encoder over the same image+state keys).
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        DDPG,
        **params_kwargs(args, DDPG.params_class),
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        replay_lazy_next_obs=args.replay_lazy_next_obs,
        replay_pin_sampled_batch=args.replay_pin_sampled_batch,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        bootstrap_at_done=args.bootstrap_at_done,
        training_freq=args.training_freq,
        utd=args.utd,
        image_augmentation_seed=args.seed + 1_000_003,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        **image_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_drqv2(args: DrQv2Args) -> None:
    from rl_garden.training.online.runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    from rl_garden.common.env_args import make_env_request

    run_online(
        args,
        make_env_request=make_env_request,
        build_agent=build_drqv2,
        post_learn=lambda agent: agent.replay_buffer.flush(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import CheckpointArgs, ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs, EnvRunArgs
from rl_garden.training.online.registry import registry


@dataclass
class DrQv2Args(DDPGParams, EnvRunArgs, CheckpointArgs, ObservationArgs, EnvBackendArgs):
    """DrQ-v2 with multi-env backend support. Requires image observations
    (``DDPG``'s "at least one image observation key" check) -- pass
    ``--obs.rgb <camera>`` (optionally ``--obs.depth <camera>``).

    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``,
    ``--maniskill.reward-mode``.
    """

    # --- Env (overrides EnvRunArgs' defaults) ---

    control_mode: str = "pd_ee_delta_pose"

    # --- Training ---

    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    mmap_dir: str | None = None
    mmap_mode: Literal["create", "open"] = "create"
    learning_starts: int = 4_000
    batch_size: int = 256

    # --- DDPG ---

    gamma: float = 0.99
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    training_freq: int = 32
    utd: float = 0.5

    # --- Checkpoint (overrides CheckpointArgs' default) ---

    load_replay_buffer: bool = False

    # --- Replay buffer (not part of CheckpointArgs) ---

    replay_lazy_next_obs: bool = False
    replay_pin_sampled_batch: bool = False


def _drqv2_algorithm_cls() -> type:
    from rl_garden.algorithms.ddpg import DDPG

    return DDPG


registry.register(
    "drqv2", DrQv2Args, run_drqv2, algorithm_cls=_drqv2_algorithm_cls, build_agent=build_drqv2
)
