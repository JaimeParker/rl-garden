"""SUPE run function.

Env wrapping happens here, at env-construction time, via
``SkillActionWrapper`` -- exactly like ``DPPO``'s ``ActionChunkWrapper``
wiring (``rl_garden/training/online/dppo.py:39-46``): ``SUPE`` itself only
ever sees an already-skill-wrapped ``env.single_action_space``.
"""

from __future__ import annotations

import numpy as np

from rl_garden.training.online.rlpd import _rlpd_env_request


def build_supe(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import SUPE
    from rl_garden.algorithms.supe import load_opal_vae
    from rl_garden.envs.wrappers import SkillActionWrapper
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online._args import sac_initial_training_phase_from_args

    obs_dim = int(np.prod(env.single_observation_space.shape))
    opal_vae = load_opal_vae(
        args.opal_checkpoint, obs_dim, env.single_action_space, device=args.device,
    )
    skill_dim = opal_vae.skill_dim

    env = SkillActionWrapper(
        env, opal_vae.decoder, horizon=args.horizon, skill_dim=skill_dim, deterministic=False,
    )
    if eval_env is not None:
        eval_env = SkillActionWrapper(
            eval_env, opal_vae.decoder, horizon=args.horizon, skill_dim=skill_dim,
            deterministic=True,
        )

    net_arch = {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }

    agent = construct_agent(
        SUPE,
        env=env,
        eval_env=eval_env,
        opal_checkpoint=args.opal_checkpoint,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        use_pnorm=args.use_pnorm,
        offline_relabel_type=args.offline_relabel_type,
        use_rnd_offline=args.use_rnd_offline,
        use_rnd_online=args.use_rnd_online,
        rnd_coeff=args.rnd_coeff,
        relabeler_hidden_dims=args.relabeler_hidden_dims,
        relabeler_lr=args.relabeler_lr,
        rnd_hidden_dims=args.rnd_hidden_dims,
        rnd_feature_dim=args.rnd_feature_dim,
        rnd_lr=args.rnd_lr,
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
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        exclude_bias_from_decay=args.exclude_bias_from_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
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
        actor_use_layer_norm=args.actor_use_layer_norm,
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
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.offline_dataset is not None:
        if args.offline_buffer_size is None:
            raise ValueError(
                "--offline_buffer_size is required when --offline_dataset is set "
                "(RLPD's prior-data buffer has no cheap way to count "
                "h5/minari transitions ahead of time)."
            )
        loaded = agent.load_skill_relabeled_offline_buffer(
            args.offline_dataset,
            buffer_size=args.offline_buffer_size,
            chunk_size=args.horizon,
            discount=args.gamma,
            offline_data_ratio=args.offline_data_ratio,
            num_traj=args.offline_num_traj,
        )
        if args.std_log:
            print(
                "[supe] "
                f"offline_dataset={args.offline_dataset} "
                f"loaded_windows={loaded} "
                f"offline_data_ratio={args.offline_data_ratio} "
                f"offline_relabel_type={args.offline_relabel_type} "
                f"horizon={args.horizon} skill_dim={skill_dim}",
                flush=True,
            )
    return agent


def run_supe(args: "SUPEArgs") -> None:
    from rl_garden.training.online._runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    run_online(
        args,
        obs_tag="state",
        make_env_request=_rlpd_env_request,
        build_agent=build_supe,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.training.online._registry import registry  # noqa: E402
from rl_garden.training.online.explore import ExPLOREArgs  # noqa: E402


@dataclass
class SUPEArgs(ExPLOREArgs):
    """SUPE -- ExPLORe operating over OPAL's learned skill space instead of
    raw actions. State-based (Box observation space) only. Requires a
    pretrained OPAL checkpoint (``opal`` offline entrypoint).

    ``total_timesteps`` counts macro-steps (each is ``horizon`` raw env
    steps), matching ``DPPO``'s ``act_steps`` convention -- the raw env-step
    budget is ``total_timesteps * horizon``.
    """

    opal_checkpoint: str = ""
    horizon: int = 4


registry.register("supe", SUPEArgs, run_supe)
