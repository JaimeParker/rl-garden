"""RLPDHybrid run function."""

from __future__ import annotations

from typing import Literal


def build_rlpd_hybrid(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import RLPDHybrid
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
        "critic_backbone_type": args.critic_backbone_type,
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        RLPDHybrid,
        env=env,
        eval_env=eval_env,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        discrete_hidden_dim=args.discrete_hidden_dim,
        discrete_lr=args.discrete_lr,
        # HilSerlArgs-only field (real-robot gripper-flip reward shaping);
        # getattr so plain RLPDHybridArgs (sim training) callers are
        # unaffected without needing this field.
        use_grasp_penalty=getattr(args, "use_grasp_penalty", False),
        # HilSerlArgs-only field (real-robot replay-buffer image dedup). The
        # buffer's image keys are derived by RLPDHybrid itself from
        # self.observation_encoders.schema.image_keys -- unused when
        # memory_efficient_buffer is off (the default).
        memory_efficient_buffer=getattr(args, "memory_efficient_buffer", False),
        memory_efficient_frame_stack=args.obs.frame_stack,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        backup_entropy=args.backup_entropy,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        use_pnorm=args.use_pnorm,
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
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        exclude_bias_from_decay=args.exclude_bias_from_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
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
        actor_use_layer_norm=args.actor_use_layer_norm,
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
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.offline_dataset is not None:
        if args.offline_buffer_size is None:
            raise ValueError(
                "--offline_buffer_size is required when --offline_dataset is set."
            )
        agent.load_offline_replay_buffer(
            args.offline_dataset,
            backend=args.dataset_backend,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            success_key=args.success_key,
        )
    return agent


def run_rlpd_hybrid(args: RLPDHybridArgs) -> None:
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
        build_agent=build_rlpd_hybrid,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.networks import BackboneType, KernelInit
from rl_garden.training.online._args import VisionSACTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class RLPDHybridArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """RLPDHybrid -- RLPD with an independent discrete Double-DQN head for a
    hybrid continuous-arm + discrete-gripper action space (HIL-SERL's
    ``sac_hybrid_single``). The last action dim is treated as the discrete
    gripper choice; the continuous actor/critic only see the remaining dims.
    """

    n_critics: int = 10
    critic_subsample_size: int | None = 2
    critic_use_layer_norm: bool = True
    backup_entropy: bool = True
    utd: float = 4.0

    actor_dropout_rate: float | None = None
    critic_dropout_rate: float | None = None
    kernel_init: KernelInit | None = None
    backbone_type: BackboneType = "mlp"
    use_pnorm: bool = False

    weight_decay: float = 0.0
    use_adamw: bool = False
    exclude_bias_from_decay: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: float | None = None

    # Registry-backed (rl_garden.buffers.dataset_backend_registry), not a
    # Literal: a new backend registers itself, no CLI arg change needed here.
    dataset_backend: str = "h5"
    offline_dataset: str | None = None
    offline_num_traj: int | None = None
    offline_buffer_size: int | None = None
    offline_data_ratio: float = 0.5
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: str | None = None

    discrete_hidden_dim: int = 256
    discrete_lr: float = 3e-4




def _rlpd_hybrid_algorithm_cls() -> type:
    from rl_garden.algorithms import RLPDHybrid

    return RLPDHybrid

registry.register(
    "rlpd_hybrid",
    RLPDHybridArgs,
    run_rlpd_hybrid,
    algorithm_cls=_rlpd_hybrid_algorithm_cls)
