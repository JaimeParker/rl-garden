"""DPPO (Diffusion PPO) fine-tuning run function.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Action chunking is applied here, at env
construction time, via ``ActionChunkWrapper`` -- ``DPPO`` itself only ever
sees an already-chunked ``env.single_action_space`` (see
``rl_garden/algorithms/dppo.py``'s module docstring).
"""

from __future__ import annotations


def _dppo_observation_kwargs(args) -> dict:
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


def build_dppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import DPPO
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.act_steps)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.act_steps)

    image_kwargs = _dppo_observation_kwargs(args)

    agent = construct_agent(
        DPPO,
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


def run_dppo(args: "DPPOArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_dppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import DPPOTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class DPPOArgs(DPPOTrainingArgs, ObservationArgs, EnvBackendArgs):
    """DPPO (Diffusion PPO) fine-tuning. Requires ``--bc_checkpoint`` (a
    ``DiffusionBC`` checkpoint -- state-only obs, so a checkpoint trained
    against Box obs will not load into a Dict-obs DPPO run). State-only
    observations by default; pass ``--obs.rgb <camera>`` for CNN-based
    Dict/RGBD observations."""


registry.register("dppo", DPPOArgs, run_dppo)
