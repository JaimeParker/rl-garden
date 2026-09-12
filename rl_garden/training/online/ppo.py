"""PPO run function."""

from __future__ import annotations


def _ppo_observation_kwargs(args) -> dict:
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.encoders.config import EncoderConfig

    # detach_encoder_on_actor's old semantics: None/True -> the encoder is
    # trained only by the value loss, actor path detached
    # ("shared_critic_grad"); False -> both losses train it ("shared").
    # args.encoder_sharing (when set) overrides this generic default, same
    # as every other algorithm.
    detach = args.detach_encoder_on_actor
    encoder_sharing = "shared_critic_grad" if (detach is None or detach) else "shared"
    if args.encoder_sharing is not None:
        encoder_sharing = args.encoder_sharing

    # --encoder.normalize-obs is the only normalize_obs knob (the old
    # PPOTrainingArgs.normalize_obs flag was deleted); build_observation_encoder
    # (rl_garden.encoders.factory) allows it alone on a state-only schema.
    if args.obs.is_visual:
        encoder_config = args.encoder
    else:
        encoder_config = EncoderConfig(normalize_obs=True) if args.encoder.normalize_obs else None

    return {
        "encoder_config": encoder_config,
        "obs_groups": resolve_obs_groups_config(args),
        "encoder_sharing": encoder_sharing,
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }


def _ppo_common_kwargs(
    args, env, eval_env, logger, checkpoint_dir, image_kwargs: dict
) -> dict:
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
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        desired_kl=args.desired_kl,
        adaptive_lr_min=args.adaptive_lr_min,
        adaptive_lr_max=args.adaptive_lr_max,
        actor_use_layer_norm=args.actor_use_layer_norm,
        value_use_layer_norm=args.value_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        value_use_group_norm=args.value_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        value_dropout_rate=args.value_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        critic_backbone_type=args.critic_backbone_type,
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
    from rl_garden.training.inspection import construct_agent

    image_kwargs = _ppo_observation_kwargs(args)
    agent = construct_agent(
        PPO,
        **_ppo_common_kwargs(args, env, eval_env, logger, checkpoint_dir, image_kwargs),
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_ppo(args: PPOArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import VisionPPOTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class PPOArgs(VisionPPOTrainingArgs, EnvBackendArgs):
    """PPO. State-only observations by default; pass ``--obs.rgb <camera>``
    (optionally ``--obs.depth <camera>``) for Dict/RGBD observations.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``.
    """


registry.register("ppo", PPOArgs, run_ppo)
