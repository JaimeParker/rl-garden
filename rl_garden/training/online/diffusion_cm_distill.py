"""DiffusionCMDistillOnline run function: DPPO's online PPO fine-tuning
fused with a per-iteration LCM one-step distillation step.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations (inherited from DPPO's own vision support
-- ``DiffusionCMDistillOnline`` subclasses ``DPPO`` directly, see
``rl_garden/algorithms/dppo.py``'s module docstring and
``rl_garden/algorithms/diffusion_cm_distill.py``, whose ``_distill_step``
reuses ``DPPOPolicy._cond`` unmodified). Same action-chunking convention as
``rl_garden/training/online/dppo.py`` -- see that module's docstring.
"""

from __future__ import annotations


def _diffusion_cm_distill_observation_kwargs(args) -> dict:
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config

    kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
    }
    if args.encoder_sharing is not None:
        kwargs["encoder_sharing"] = args.encoder_sharing
    return kwargs


def build_diffusion_cm_distill(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import DiffusionCMDistillOnline
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.act_steps)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.act_steps)

    image_kwargs = _diffusion_cm_distill_observation_kwargs(args)

    agent = construct_agent(
        DiffusionCMDistillOnline,
        env=env,
        eval_env=eval_env,
        **image_kwargs,
        bc_checkpoint=args.bc_checkpoint or None,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        horizon_steps=args.horizon_steps,
        act_steps=args.act_steps,
        denoising_steps=args.denoising_steps,
        ft_denoising_steps=args.ft_denoising_steps,
        actor_activation_fn=args.actor_activation_fn,
        actor_residual_style=args.actor_residual_style,
        critic_activation_fn=args.critic_activation_fn,
        critic_residual_style=args.critic_residual_style,
        time_dim=args.time_dim,
        kernel_init=args.kernel_init,
        denoised_clip_value=args.denoised_clip_value,
        randn_clip_value=args.randn_clip_value,
        final_action_clip_value=args.final_action_clip_value,
        min_sampling_denoising_std=args.min_sampling_denoising_std,
        min_logprob_denoising_std=args.min_logprob_denoising_std,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        critic_warmup_updates=args.critic_warmup_updates,
        update_epochs=args.update_epochs,
        update_batch_size=args.update_batch_size,
        norm_adv=args.norm_adv,
        gamma_denoising=args.gamma_denoising,
        clip_ploss_coef=args.clip_ploss_coef,
        clip_ploss_coef_base=args.clip_ploss_coef_base,
        clip_ploss_coef_rate=args.clip_ploss_coef_rate,
        clip_vloss_coef=args.clip_vloss_coef,
        clip_advantage_lower_quantile=args.clip_advantage_lower_quantile,
        clip_advantage_upper_quantile=args.clip_advantage_upper_quantile,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        reward_horizon=args.reward_horizon,
        finite_horizon_gae=args.finite_horizon_gae,
        cm_mlp_dims=args.cm_mlp_dims,
        cm_lr=args.cm_lr,
        cm_ema_decay=args.cm_ema_decay,
        cm_grad_clip_norm=args.cm_grad_clip_norm,
        cm_sigma_data=args.cm_sigma_data,
        cm_timestep_scaling=args.cm_timestep_scaling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_diffusion_cm_distill(args: "DiffusionCMDistillOnlineArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_diffusion_cm_distill,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import DiffusionCMDistillOnlineTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class DiffusionCMDistillOnlineArgs(
    DiffusionCMDistillOnlineTrainingArgs, ObservationArgs, EnvBackendArgs
):
    """DiffusionCMDistillOnline: DPPO's online PPO fine-tuning fused with a
    per-iteration diffusion-to-consistency-model distillation step. Requires
    ``--bc_checkpoint`` (a ``DiffusionBC`` checkpoint -- state-only obs, so a
    checkpoint trained against Box obs will not load into a Dict-obs run).
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    CNN-based Dict/RGBD observations."""


def _diffusion_cm_distill_algorithm_cls() -> type:
    from rl_garden.algorithms import DiffusionCMDistillOnline

    return DiffusionCMDistillOnline


registry.register(
    "diffusion_cm_distill_online",
    DiffusionCMDistillOnlineArgs,
    run_diffusion_cm_distill,
    algorithm_cls=_diffusion_cm_distill_algorithm_cls,
)
