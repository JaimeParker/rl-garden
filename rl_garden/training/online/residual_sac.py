"""ResidualSAC run function."""
from __future__ import annotations

from typing import Literal, Optional


def _effective_base_policy(args) -> Literal["act", "sac", "zero"]:
    if args.debug:
        return "zero"
    return args.base_policy


def _residual_sac_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest

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
        backend_config=backend_config,
    )


def _make_base_action_provider(args, env):
    from rl_garden.policies.base_policies import make_base_policy

    base_policy = _effective_base_policy(args)
    provider = make_base_policy(
        base_policy=base_policy,
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env=env,
        base_ckpt_path=args.base_ckpt_path,
        base_act_temporal_agg=args.base_act_temporal_agg,
        base_act_temporal_agg_k=args.base_act_temporal_agg_k,
        base_sac_encoder=args.base_sac_encoder,
        base_sac_encoder_features_dim=args.base_sac_encoder_features_dim,
        base_sac_image_fusion_mode=args.base_sac_image_fusion_mode,
        base_sac_deterministic=args.base_sac_deterministic,
    )
    if base_policy == "act":
        print(
            "[residual] base_policy=act "
            f"ckpt={provider.checkpoint_path} "
            f"state_dim={provider.spec.state_dim} "
            f"action_dim={provider.spec.action_dim} "
            f"num_queries={provider.config.num_queries}",
            flush=True,
        )
    elif base_policy == "sac":
        print(
            "[residual] base_policy=sac "
            f"ckpt={args.base_ckpt_path} "
            f"deterministic={args.base_sac_deterministic}",
            flush=True,
        )
    else:
        print("[residual] base_policy=zero", flush=True)
    return provider


def build_residual_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ResidualSAC
    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        image_keys_from_env,
        vit_sac_kwargs_from_args,
    )
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    if args.load_actor_checkpoint is not None:
        raise ValueError(
            "ResidualSAC does not support --load_actor_checkpoint; "
            "use --load_checkpoint for ResidualSAC checkpoints or "
            "--base_ckpt_path for the frozen base policy."
        )

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

    base_action_provider = _make_base_action_provider(args, env)
    agent = ResidualSAC(
        env=env,
        eval_env=eval_env,
        base_action_provider=base_action_provider,
        residual_action_scale=args.residual_action_scale,
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
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.offline_dataset_path is not None:
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset_path,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
        )
        if args.std_log:
            print(
                "[residual] "
                f"offline_dataset={args.offline_dataset_path} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


def run_residual_sac(args: ResidualSACArgs) -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    base_policy = _effective_base_policy(args)
    base_tag = "debug_zero_base" if args.debug else f"{base_policy}_base"
    obs_tag = f"{base_tag}_rgbd_{args.encoder}" if is_visual else f"{base_tag}_state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_residual_sac_env_request,
        build_agent=build_residual_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionSACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class ResidualSACArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """ResidualSAC — residual actions on top of ACT, SAC, or zero base policies.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    ManiSkill-specific peg options are available under ``--maniskill.*``.
    """

    residual_action_scale: float = 0.1
    debug: bool = False
    base_policy: Literal["act", "sac", "zero"] = "act"
    base_ckpt_path: Optional[str] = "act-peg-only"
    base_act_temporal_agg: bool = True
    base_act_temporal_agg_k: float = 0.01
    base_sac_encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    base_sac_encoder_features_dim: int = 256
    base_sac_image_fusion_mode: Optional[Literal["stack_channels", "per_key"]] = None
    base_sac_deterministic: bool = True
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None
    offline_buffer_size: Optional[int] = None
    offline_data_ratio: float = 0.5


registry.register("residual_sac", ResidualSACArgs, run_residual_sac)
