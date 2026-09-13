"""SAC run function."""

from __future__ import annotations


def _sac_common_kwargs(
    args, env, eval_env, logger, checkpoint_dir, image_kwargs
) -> dict:
    """Kwargs shared by every SAC-family algorithm built from ``SACTrainingArgs``
    (plain SAC and RecurrentSAC alike) -- everything except the recurrent-only
    ``rnn_*``/``burn_in_len``/etc. fields."""
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    return dict(
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        nstep=args.nstep,
        tau=args.tau,
        bootstrap_at_done=args.bootstrap_at_done,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_tuning=args.alpha_tuning,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        alpha_lr=args.alpha_lr,
        q_landscape_diagnostics=args.q_landscape_diagnostics,
        q_landscape_num_actions=args.q_landscape_num_actions,
        q_landscape_batch_size=args.q_landscape_batch_size,
        q_mc_diagnostics=args.q_mc_diagnostics,
        initial_training_phase=sac_initial_training_phase_from_args(args),
        critic_impl=args.critic_impl,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_mode=args.actor_log_std_mode,
        net_arch=net_arch,
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


def _sac_observation_kwargs(args) -> dict:
    """Observation-related kwargs shared by every SAC-family entrypoint that
    supports a distinct critic encoder (sac/rlpd/rlpd_hybrid). The algorithm
    resolves image keys/encoders itself from ``encoder_config``/``obs_groups``
    (``ObservationEncoderMixin``); this entrypoint only forwards ``args``.
    """
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config

    kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
        "critic_backbone_type": args.critic_backbone_type,
    }
    if args.encoder_sharing is not None:
        kwargs["encoder_sharing"] = args.encoder_sharing
    return kwargs


def build_sac(args, env, eval_env, logger, checkpoint_dir):
    import os

    from rl_garden.algorithms import SAC
    from rl_garden.algorithms.sac_ddp import SACDDP
    from rl_garden.common.ddp import ddp_rank, is_ddp_active
    from rl_garden.training.inspection import construct_agent

    image_kwargs = _sac_observation_kwargs(args)

    algo_cls = SACDDP if is_ddp_active() else SAC
    mmap_dir = args.mmap_dir
    if is_ddp_active() and mmap_dir is not None:
        if args.mmap_mode == "open":
            raise SystemExit(
                "--mmap-mode open (resuming a shared disk-backed buffer) is "
                "not supported under multi-GPU DDP: each rank would need its "
                "own resumed buffer, and this feature only supports "
                "creating a fresh, rank-local one. Use --mmap-mode create."
            )
        mmap_dir = os.path.join(mmap_dir, f"rank{ddp_rank()}")

    agent = construct_agent(
        algo_cls,
        mmap_dir=mmap_dir,
        mmap_mode=args.mmap_mode,
        **_sac_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.load_actor_checkpoint is not None:
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    return agent


def run_sac(args: SACArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_sac,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import VisionSACTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class SACArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """SAC. State-only observations by default; pass ``--obs.rgb <camera>``
    (optionally ``--obs.depth <camera>``) for Dict/RGBD observations.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``.
    """


def _sac_algorithm_cls() -> type:
    from rl_garden.algorithms import SAC

    return SAC


registry.register("sac", SACArgs, run_sac, algorithm_cls=_sac_algorithm_cls)
