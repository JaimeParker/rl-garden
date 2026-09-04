"""RLPD run function."""

from __future__ import annotations

from typing import Literal


def _rlpd_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

    is_visual = args.obs_mode != "state"
    backend_config = args.resolve_backend_config()
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width if is_visual else None,
        camera_height=args.camera_height if is_visual else None,
        include_state=args.include_state if is_visual else True,
        per_camera_rgbd=args.per_camera_rgbd if is_visual else False,
        frame_stack=args.frame_stack,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
        num_eval_envs=args.num_eval_envs,
        create_eval_env=should_create_eval_env(args),
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


def build_rlpd(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import RLPD
    from rl_garden.common.cli_args import (
        critic_features_extractor_kwargs_from_args,
        image_encoder_factory_from_args,
        image_keys_from_env,
        vit_sac_kwargs_from_args,
    )
    from rl_garden.training.inspection import construct_agent
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
        sac_kwargs = vit_sac_kwargs_from_args(args, image_keys)
        policy_kwargs = dict(sac_kwargs.pop("policy_kwargs", {}) or {})
        if args.critic_encoder:
            critic_image_keys = image_keys_from_env(
                env, args, image_key_filter=args.critic_image_keys
            )
            policy_kwargs.update(
                critic_features_extractor_kwargs_from_args(args, critic_image_keys)
            )
        image_kwargs = dict(
            image_keys=image_keys,
            image_encoder_factory=factory,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
            image_augmentation_seed=args.seed + 1_000_003,
            critic_backbone_type=args.critic_backbone_type,
            **sac_kwargs,
        )
        if policy_kwargs:
            image_kwargs["policy_kwargs"] = policy_kwargs

    agent = construct_agent(
        RLPD,
        env=env,
        eval_env=eval_env,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
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
        std_parameterization=args.std_parameterization,
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
                "(RLPD's prior-data buffer has no cheap way to count "
                "h5/minari transitions ahead of time)."
            )
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset,
            backend=args.dataset_backend,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            success_key=args.success_key,
        )
        if args.std_log:
            print(
                "[rlpd] "
                f"offline_dataset={args.offline_dataset} "
                f"backend={args.dataset_backend} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


def run_rlpd(args: RLPDArgs) -> None:
    from rl_garden.training.online._runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_rlpd_env_request,
        build_agent=build_rlpd,
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
class RLPDArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """RLPD -- high UTD + REDQ-style critic ensemble subsampling + LayerNorm
    + offline/online prior-data mixing from step 0, on top of SAC.

    Env backend: ``--env_backend maniskill`` (default), ``robotwin``, or
    ``d4rl_legacy``.
    """

    n_critics: int = 10
    critic_subsample_size: int | None = 2
    critic_use_layer_norm: bool = True
    utd: float = 4.0

    actor_dropout_rate: float | None = None
    critic_dropout_rate: float | None = None
    kernel_init: KernelInit | None = None
    backbone_type: BackboneType = "mlp"
    use_pnorm: bool = False
    std_parameterization: Literal["exp", "uniform"] = "exp"

    weight_decay: float = 0.0
    use_adamw: bool = False
    exclude_bias_from_decay: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: float | None = None

    dataset_backend: Literal["h5", "minari", "d4rl_legacy", "robomimic", "ogbench", "rlbench"] = "h5"
    offline_dataset: str | None = None
    offline_num_traj: int | None = None
    offline_buffer_size: int | None = None
    offline_data_ratio: float = 0.5
    # Applied to both the live env (via EnvRequest) and the offline dataset
    # loader, so online and offline rewards share one scale in the Bellman
    # target RLPD mixes them into. Matches off2on's Off2OnCommonArgs fields.
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: str | None = None


registry.register("rlpd", RLPDArgs, run_rlpd)
