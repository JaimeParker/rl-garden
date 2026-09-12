"""TD-MPC2 run function."""

from __future__ import annotations


def _tdmpc2_env_request(args, run_name):
    from rl_garden.common.env_args import make_env_request

    if args.num_envs != 1 or args.num_eval_envs != 1:
        raise ValueError(
            "TDMPC2 requires --num_envs 1 --num_eval_envs 1 (vectorized rollout "
            f"is not supported in this port); got num_envs={args.num_envs}, "
            f"num_eval_envs={args.num_eval_envs}."
        )
    return make_env_request(args, run_name)


def build_tdmpc2(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.common.cli_args import resolve_obs_groups_config
    from rl_garden.algorithms.tdmpc2 import TDMPC2
    from rl_garden.training.inspection import construct_agent

    # TDMPC2 only accepts encoder_config/obs_groups/image_augmentation_seed
    # (no distinct critic encoder; encoder_sharing is a fixed "shared" class
    # attribute, not a constructor kwarg).
    image_kwargs = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
    }

    agent = construct_agent(
        TDMPC2,
        env=env,
        eval_env=eval_env,
        episode_length=args.episode_length,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        batch_size=args.batch_size,
        seed_steps=args.seed_steps,
        use_planner=args.use_planner,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_pi_trajs=args.num_pi_trajs,
        iterations=args.iterations,
        min_std=args.min_std,
        max_std=args.max_std,
        temperature=args.temperature,
        latent_dim=args.latent_dim,
        mlp_dim=args.mlp_dim,
        simnorm_dim=args.simnorm_dim,
        num_q=args.num_q,
        num_bins=args.num_bins,
        vmin=args.vmin,
        vmax=args.vmax,
        dropout=args.dropout,
        episodic=args.episodic,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        enc_lr_scale=args.enc_lr_scale,
        grad_clip_norm=args.grad_clip_norm,
        tau=args.tau,
        rho=args.rho,
        consistency_coef=args.consistency_coef,
        reward_coef=args.reward_coef,
        value_coef=args.value_coef,
        termination_coef=args.termination_coef,
        discount_denom=args.discount_denom,
        discount_min=args.discount_min,
        discount_max=args.discount_max,
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
    return agent


def run_tdmpc2(args: TDMPC2Args) -> None:
    from rl_garden.training.online._runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_tdmpc2_env_request,
        build_agent=build_tdmpc2,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import VisionTDMPC2TrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class TDMPC2Args(VisionTDMPC2TrainingArgs, EnvBackendArgs):
    """TD-MPC2 — implicit world model + CEM/MPPI planner (single-task,
    single-env only; see ``rl_garden.algorithms.tdmpc2.agent`` docstring).

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    Dict/RGBD observations.
    """


registry.register("tdmpc2", TDMPC2Args, run_tdmpc2)
