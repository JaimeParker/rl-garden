"""PPO run function."""
from __future__ import annotations


def _ppo_env_request(args, run_name):
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
        frame_stack=1,
        num_eval_envs=args.num_eval_envs,
        create_eval_env=should_create_eval_env(args),
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


def _ppo_image_kwargs(args, env) -> dict:
    from rl_garden.common.cli_args import (
        image_encoder_factory_from_args,
        image_keys_from_env,
    )

    if args.obs_mode == "state":
        return {}
    return dict(
        image_keys=image_keys_from_env(env, args),
        image_encoder_factory=image_encoder_factory_from_args(args),
        image_fusion_mode=args.image_fusion_mode,
    )


def _ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs: dict) -> dict:
    """Kwargs shared by ``PPO`` and ``RecurrentPPO`` construction -- everything
    except the rnn-specific params that only ``RecurrentPPO`` accepts."""
    return dict(
        env=env,
        eval_env=eval_env,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        finite_horizon_gae=args.finite_horizon_gae,
        detach_encoder_on_actor=args.detach_encoder_on_actor,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        actor_use_layer_norm=args.actor_use_layer_norm,
        value_use_layer_norm=args.value_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        value_use_group_norm=args.value_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        value_dropout_rate=args.value_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        log_std_init=args.log_std_init,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
        **image_kwargs,
    )


def build_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import PPO

    image_kwargs = _ppo_image_kwargs(args, env)
    agent = PPO(**_ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs))
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_ppo(args: PPOArgs) -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_ppo_env_request,
        build_agent=build_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import VisionPPOTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class PPOArgs(VisionPPOTrainingArgs, EnvBackendArgs):
    """PPO — visual defaults; pass ``--obs_mode state`` for state obs.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``.
    """


registry.register("ppo", PPOArgs, run_ppo)
