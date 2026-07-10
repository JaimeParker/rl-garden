"""RLPDHybrid run function."""
from __future__ import annotations

from typing import Literal, Optional


def _rlpd_hybrid_env_request(args, run_name):
    from rl_garden.training.online.rlpd import _rlpd_env_request

    return _rlpd_env_request(args, run_name)


def build_rlpd_hybrid(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import RLPDHybrid
    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        image_keys_from_env,
        vit_sac_kwargs_from_args,
    )
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    is_visual = args.obs_mode != "state"
    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    image_kwargs: dict = {}
    if is_visual:
        factory = image_encoder_factory_from_args(args)
        image_keys = image_keys_from_env(env, args)
        image_kwargs = dict(
            image_keys=image_keys,
            image_encoder_factory=factory,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
            image_augmentation_seed=args.seed + 1_000_003,
            **vit_sac_kwargs_from_args(args, image_keys),
        )

    agent = RLPDHybrid(
        env=env,
        eval_env=eval_env,
        discrete_hidden_dim=args.discrete_hidden_dim,
        discrete_lr=args.discrete_lr,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
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
    if args.offline_dataset_path is not None:
        if args.offline_buffer_size is None:
            raise ValueError(
                "--offline_buffer_size is required when --offline_dataset_path is set."
            )
        agent.load_offline_replay_buffer(
            args.offline_dataset_path,
            source=args.dataset_source,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            success_key=args.success_key,
        )
    return agent


def run_rlpd_hybrid(args: "RLPDHybridArgs") -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_rlpd_hybrid_env_request,
        build_agent=build_rlpd_hybrid,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.networks import BackboneType, KernelInit  # noqa: E402
from rl_garden.training.online._args import VisionSACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class RLPDHybridArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """RLPDHybrid -- RLPD with an independent discrete Double-DQN head for a
    hybrid continuous-arm + discrete-gripper action space (HIL-SERL's
    ``sac_hybrid_single``). The last action dim is treated as the discrete
    gripper choice; the continuous actor/critic only see the remaining dims.
    """

    n_critics: int = 10
    critic_subsample_size: Optional[int] = 2
    critic_use_layer_norm: bool = True
    utd: float = 4.0

    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: BackboneType = "mlp"
    use_pnorm: bool = False

    weight_decay: float = 0.0
    use_adamw: bool = False
    exclude_bias_from_decay: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None

    dataset_source: Literal["maniskill_h5", "minari"] = "maniskill_h5"
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None
    offline_buffer_size: Optional[int] = None
    offline_data_ratio: float = 0.5
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: Optional[str] = None

    discrete_hidden_dim: int = 256
    discrete_lr: float = 3e-4


registry.register("rlpd_hybrid", RLPDHybridArgs, run_rlpd_hybrid)
