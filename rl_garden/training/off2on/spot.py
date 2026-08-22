"""SPOT off2on training registration.

Builds ``Off2OnSPOT`` and reuses the same ``_runner.run_off2on`` orchestration
as ``iql``/``awac``/``calql``. Overrides ``Off2OnCommonArgs``' CQL/Cal-QL-tuned
defaults that don't apply to SPOT's plain twin-critic TD3 backbone (no
LayerNorm, 2 critics, no separate warmup phase -- CORL's own online loop
starts noisy rollout immediately at the switch).
"""

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on._args import SPOTOff2OnTrainingArgs
from rl_garden.training.off2on._registry import registry


@dataclass
class SPOTOff2OnArgs(SPOTOff2OnTrainingArgs, EnvBackendArgs):
    """SPOT off2on args: plain twin-critic TD3 backbone, no warmup.

    Box observations only; pass ``--obs_mode state``.
    """

    n_critics: int = 2
    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = False
    warmup_steps: int = 0


def build_spot(args: SPOTOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnSPOT
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.off2on._args import initial_training_phase_from_args

    agent = construct_agent(
        Off2OnSPOT,
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
        offline_sampling=args.offline_sampling,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        n_critics=args.n_critics,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        vae_lr=args.vae_lr,
        vae_hidden_dim=args.vae_hidden_dim,
        vae_latent_dim=args.vae_latent_dim,
        vae_iterations=args.vae_iterations,
        beta=args.beta,
        lambd=args.lambd,
        num_samples=args.num_samples,
        iwae=args.iwae,
        lambd_cool=args.lambd_cool,
        lambd_end=args.lambd_end,
        expl_noise=args.expl_noise,
        online_discount=args.online_discount,
        # CORL's max_online_steps is a gradient-step count (config.online_iterations,
        # incremented once per SPOT.train() call), not an env-step count --
        # derive it from num_online_steps * utd, matching
        # OffPolicyAlgorithm.grad_steps_per_iteration's own convention. Only
        # read by _current_lambd() when lambd_cool=True.
        max_online_updates=max(1, round(args.num_online_steps * args.utd)),
        initial_training_phase=initial_training_phase_from_args(args),
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
    return agent


def run_spot(args: SPOTOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_spot, algorithm="spot")


registry.register("spot", SPOTOff2OnArgs, run_spot)
