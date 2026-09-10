"""DPPO (Diffusion PPO) fine-tuning run function.

Box observations by default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD
observations. Action chunking is applied here, at env construction time, via
``ActionChunkWrapper`` -- ``DPPO`` itself only ever sees an already-chunked
``env.single_action_space`` (see ``rl_garden/algorithms/dppo.py``'s module
docstring).
"""

from __future__ import annotations


def _dppo_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

    is_visual = args.obs_mode != "state"
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
        create_eval_env=should_create_eval_env(args),
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=args.resolve_backend_config(),
    )


def build_dppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import DPPO
    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.act_steps)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.act_steps)

    is_visual = args.obs_mode != "state"
    image_kwargs: dict = {}
    if is_visual:
        image_kwargs = dict(
            image_encoder_factory=image_encoder_factory_from_args(args),
        )

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
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_dppo_env_request,
        build_agent=build_dppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import VisionArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import DPPOTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class DPPOArgs(DPPOTrainingArgs, VisionArgs, EnvBackendArgs):
    """DPPO (Diffusion PPO) fine-tuning. Requires ``--bc_checkpoint`` (a
    ``DiffusionBC`` checkpoint -- state-only obs, so a checkpoint trained
    against Box obs will not load into a Dict-obs DPPO run). Box
    observations by default; pass ``--obs_mode rgb`` for CNN-based Dict/RGBD
    observations (defaults to "state", not VisionArgs' own "rgb" default, to
    preserve dppo's existing CLI behavior for every caller that doesn't pass
    --obs_mode)."""

    obs_mode: str = "state"


registry.register("dppo", DPPOArgs, run_dppo)
