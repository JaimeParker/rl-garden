"""SACFlow run function. State observations by default; pass ``--obs_mode
rgb`` (with a Dict/RGBD-producing env backend) for CNN-based vision -- see
``rl_garden.algorithms.sac_flow.SACFlow``'s own docstring for exactly which
vision paths are supported (CombinedExtractor encoders; not ViT, not a
separate critic encoder)."""

from __future__ import annotations


def _sac_flow_env_request(args, run_name):
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
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        create_eval_env=should_create_eval_env(args),
        backend_config=backend_config,
    )


def build_sac_flow(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import SACFlow
    from rl_garden.common.cli_args import image_encoder_factory_from_args, image_keys_from_env
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }

    is_visual = args.obs_mode != "state"
    image_kwargs: dict = {}
    if is_visual:
        if args.encoder == "vit":
            raise SystemExit(
                "sac_flow does not support --encoder vit: FlowMatchingActor "
                "has not been verified against a structured (ViT token) "
                "features extractor. Use a CNN encoder (e.g. plain_conv, "
                "resnet10/18, drqv2_conv, cnn3d)."
            )
        if args.critic_encoder is not None:
            raise SystemExit(
                "sac_flow does not support --critic-encoder (a separate "
                "critic-only image encoder): SACFlowPolicy always shares "
                "one encoder between actor and critic, unlike SACPolicy."
            )
        image_kwargs = dict(
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_keys=image_keys_from_env(env, args),
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
            image_augmentation_seed=args.seed + 1_000_003,
        )

    agent = construct_agent(
        SACFlow,
        env=env,
        eval_env=eval_env,
        denoising_steps=args.denoising_steps,
        noise_std=args.noise_std,
        flow_hidden_dims=[args.flow_hidden_dim] * args.flow_hidden_layers,
        flow_use_layer_norm=args.flow_use_layer_norm,
        **image_kwargs,
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
        critic_use_layer_norm=args.critic_use_layer_norm,
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
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.load_actor_checkpoint is not None:
        # Inherited from SAC; expects a BC checkpoint with a matching actor
        # architecture. A flow actor's state_dict keys don't overlap with a
        # Gaussian BC actor's, so this will raise a clear "missing keys"
        # ValueError rather than silently loading nothing -- there is no
        # flow-compatible BC checkpoint format in this version.
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    return agent


def run_sac_flow(args: "SACFlowArgs") -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_sac_flow_env_request,
        build_agent=build_sac_flow,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionSACFlowTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class SACFlowArgs(VisionSACFlowTrainingArgs, EnvBackendArgs):
    """SACFlow -- SAC with a flow-matching actor. State observations by
    default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD observations
    (not ``--encoder vit``, not ``--critic-encoder`` -- see ``SACFlow``'s
    own docstring).

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend custom``.
    """


registry.register("sac_flow", SACFlowArgs, run_sac_flow)
