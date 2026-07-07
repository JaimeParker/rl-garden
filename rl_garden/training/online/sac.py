"""SAC run function."""
from __future__ import annotations


def _sac_env_request(args, run_name):
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


def _sac_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs) -> dict:
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


def build_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import SAC
    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        image_keys_from_env,
        vit_sac_kwargs_from_args,
    )

    is_visual = args.obs_mode != "state"
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

    agent = SAC(**_sac_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs))
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.load_actor_checkpoint is not None:
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    return agent


def run_sac(args: SACArgs) -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_sac_env_request,
        build_agent=build_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionSACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class SACArgs(VisionSACTrainingArgs, EnvBackendArgs):
    """SAC — visual defaults; pass ``--obs_mode state`` for state obs.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``.
    """


registry.register("sac", SACArgs, run_sac)
