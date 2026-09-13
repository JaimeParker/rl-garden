"""ACRLPD (Q-chunking's action-chunked RLPD) run function.

``ACRLPDCore._build_replay_buffer``/``_state_only_obs_space`` raise
``TypeError`` if the resolved observation space carries any image key --
ACRLPD is state-only for v1 -- so ``--obs.rgb``/``--obs.depth`` are exposed
(for CLI/config uniformity with every other algorithm) but any non-default
value is rejected by the algorithm itself, not silently ignored.
"""

from __future__ import annotations


def build_acrlpd(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ACRLPD
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
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        ACRLPD,
        env=env,
        eval_env=eval_env,
        horizon_length=args.horizon_length,
        target_entropy_multiplier=args.target_entropy_multiplier,
        q_agg=args.q_agg,
        bc_alpha=args.bc_alpha,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        critic_use_layer_norm=args.critic_use_layer_norm,
        backup_entropy=args.backup_entropy,
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
                "--offline_buffer_size is required when --offline_dataset is set "
                "(ACRLPD's prior-data buffer has no cheap way to count "
                "h5 transitions ahead of time)."
            )
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset,
            backend="h5",
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            success_key=args.success_key,
        )
        if args.std_log:
            print(
                "[acrlpd] "
                f"offline_dataset={args.offline_dataset} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


def run_acrlpd(args: "ACRLPDArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_acrlpd,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional  # noqa: E402

from rl_garden.common.cli_args import ObservationArgs  # noqa: E402
from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import BackboneType, KernelInit  # noqa: E402
from rl_garden.training.online._args import SACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class ACRLPDArgs(SACTrainingArgs, ObservationArgs, EnvBackendArgs):
    """ACRLPD -- Q-chunking's action-chunked RLPD (Li, Zhou, Levine 2025,
    ``3rd_party/qc/agents/acrlpd.py``). State-only in practice: the
    algorithm itself rejects any image observation (see module docstring).

    Defaults matching the reference (n_critics=10, no REDQ subsampling,
    mean- not min-ensemble aggregation, no entropy backup) rather than plain
    RLPD's own defaults.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend custom``.
    """

    horizon_length: int = 5
    target_entropy_multiplier: float = 0.5
    q_agg: Literal["mean", "min"] = "mean"
    bc_alpha: float = 0.0

    n_critics: int = 10
    critic_subsample_size: Optional[int] = None
    critic_use_layer_norm: bool = True
    backup_entropy: bool = False
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

    offline_dataset: str | None = None
    offline_num_traj: int | None = None
    offline_buffer_size: int | None = None
    offline_data_ratio: float = 0.5
    success_key: str | None = None




def _acrlpd_algorithm_cls() -> type:
    from rl_garden.algorithms import ACRLPD

    return ACRLPD

registry.register("acrlpd", ACRLPDArgs, run_acrlpd, algorithm_cls=_acrlpd_algorithm_cls)
