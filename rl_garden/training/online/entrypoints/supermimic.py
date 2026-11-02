"""SuperMimic run function (``rl_garden.algorithms.supermimic``; guide:
``docs/guides/supermimic.md``)."""

from __future__ import annotations

from typing import Any


def build_supermimic(
    args,
    env,
    eval_env,
    logger,
    checkpoint_dir,
    *,
    algorithm_cls: type | None = None,
    **extra_kwargs: Any,
):
    """Build ``algorithm_cls`` (``SuperMimic`` when ``None``) from ``args``.
    ``extra_kwargs`` go to its constructor next to the common kwargs. The
    policy trains no observation encoder, so ``--encoder``/``--obs_groups``/
    ``--critic_encoder``/``--encoder_sharing`` are not read."""
    from rl_garden.training.inspection import construct_agent

    if algorithm_cls is None:
        from rl_garden.algorithms.supermimic import SuperMimic

        algorithm_cls = SuperMimic

    agent = construct_agent(
        algorithm_cls,
        env=env,
        eval_env=eval_env,
        heads=args.heads,
        sim_steps_per_iteration=args.sim_steps_per_iteration,
        updates_per_iteration=args.updates_per_iteration,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        wm_hidden_dims=tuple(args.wm_hidden_dims),
        head_hidden_dims=tuple(args.head_hidden_dims),
        latent_dim=args.latent_dim,
        goal_encoder_dims=tuple(args.goal_encoder_dims),
        decoder_dims=tuple(args.decoder_dims),
        critic_dims=tuple(args.critic_dims),
        horizon=args.horizon,
        wm_window=args.wm_window,
        wm_batches=args.wm_batches,
        wm_batch_size=args.wm_batch_size,
        imagination_batch=args.imagination_batch,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        lr_final_ratio=args.lr_final_ratio,
        lr_decay_updates=args.lr_decay_updates,
        actor_betas=tuple(args.actor_betas),
        wm_grad_norm=args.wm_grad_norm,
        actor_grad_norm=args.actor_grad_norm,
        actor_grad_value=args.actor_grad_value,
        critic_grad_norm=args.critic_grad_norm,
        critic_epochs=args.critic_epochs,
        critic_minibatches=args.critic_minibatches,
        action_noise_std=args.action_noise_std,
        obs_history_len=args.obs_history_len,
        warmup_updates=args.warmup_updates,
        amp_dtype=args.amp_dtype,
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
        **extra_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_supermimic(args: SuperMimicArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_supermimic,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import SuperMimicTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class SuperMimicArgs(SuperMimicTrainingArgs, EnvBackendArgs):
    """SuperMimic: online model-based RL with a differentiable explicit-state
    world model and a deterministic policy trained by backpropagation through
    imagined rollouts. State observations; privileged keys via
    ``--obs.extra_state sim ctx goal time``.

    Env backend: any backend whose env emits the declared observation keys;
    ``--heads analytic`` also needs ``env.unwrapped.analytic_mdp``.
    """


def _supermimic_algorithm_cls() -> type:
    from rl_garden.algorithms.supermimic import SuperMimic

    return SuperMimic


registry.register(
    "supermimic",
    SuperMimicArgs,
    run_supermimic,
    algorithm_cls=_supermimic_algorithm_cls,
    build_agent=build_supermimic,
)
