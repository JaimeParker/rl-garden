"""NCRL stage-2 (online fine-tuning) run function. Stage 1 is the offline
``rssm_pretrain`` entrypoint; see ``docs/guides/ncrl.md``."""

from __future__ import annotations

_DM_CONTROL_EPISODE_FRAMES = 1000


def resolve_episode_timing(args) -> tuple[int, int]:
    """``(episode_frames, action_repeat)``: explicit flags win; otherwise the
    dm_control backend's values (1000 frames per episode,
    ``--dm_control.action_repeat``)."""
    if args.env_backend == "dm_control":
        default_frames, default_repeat = _DM_CONTROL_EPISODE_FRAMES, args.dm_control.action_repeat
    else:
        default_frames, default_repeat = None, None
    episode_frames = args.episode_frames if args.episode_frames is not None else default_frames
    action_repeat = args.action_repeat if args.action_repeat is not None else default_repeat
    if episode_frames is None or action_repeat is None:
        raise SystemExit(f"ncrl on --env_backend {args.env_backend} needs --episode_frames and --action_repeat.")
    return episode_frames, action_repeat


def build_ncrl(args, env, eval_env, logger, checkpoint_dir):
    import dataclasses

    from rl_garden.algorithms.ncrl import NCRL
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent

    # Informational only: NCRL rebuilds the pretrained RSSM encoder (per --backbone).
    encoder_config = dataclasses.replace(args.encoder, backbone="dreamer_conv")
    episode_frames, action_repeat = resolve_episode_timing(args)

    agent = construct_agent(
        NCRL,
        env=env,
        eval_env=eval_env,
        pretrained_world_model=args.pretrained_world_model,
        backbone=args.backbone,
        retrieval_db=args.retrieval_db,
        prior_dataset_dir=args.prior_dataset_dir,
        online_ratio=args.online_ratio,
        num_retrieve_episodes=args.num_retrieve_episodes,
        max_retrieve_iterations=args.max_retrieve_iterations,
        use_guidance=args.use_guidance,
        bc_train_steps=args.bc_train_steps,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        guidance_schedule=tuple(args.guidance_schedule),
        episode_frames=episode_frames,
        action_repeat=action_repeat,
        warmup_frames=args.warmup_frames,
        learning_starts=args.learning_starts,
        imag_horizon=args.imag_horizon,
        discount=args.discount,
        lam=args.lam,
        actor_ent=args.actor_ent,
        model_lr=args.model_lr,
        reward_lr=args.reward_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        slow_target_fraction=args.slow_target_fraction,
        ac_units=args.ac_units,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        padded_action_dim=args.padded_action_dim,
        reward_bins=args.reward_bins,
        kl_free=args.kl_free,
        kl_free_mode=args.kl_free_mode,
        dyn_scale=args.dyn_scale,
        rep_scale=args.rep_scale,
        recon_scale=args.recon_scale,
        batch_size=args.batch_size,
        batch_length=args.batch_length,
        train_ratio=args.train_ratio,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        compute_dtype=args.compute_dtype,
        encoder_config=encoder_config,
        obs_groups=resolve_obs_groups_config(args),
        critic_encoder_config=resolve_critic_encoder_config(args),
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
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_ncrl(args: "NCRLArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online

    run_online(args, obs_tag="ncrl", make_env_request=make_env_request, build_agent=build_ncrl)


# ---------------------------------------------------------------------------
# Args + registration (bottom of file, no # noqa: E402 -- ruff has it disabled)
# ---------------------------------------------------------------------------
from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import NCRLTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class NCRLArgs(NCRLTrainingArgs, EnvBackendArgs):
    """NCRL online fine-tuning: pretrained RSSM + experience rehearsal +
    execution guidance + DreamerV2-style actor-critic
    (``rl_garden.algorithms.ncrl``). Defaults to the dm_control backend with
    one env, image-only ``rgb_default`` 64x64 observations and 5 eval
    episodes every 10k frames."""

    env_backend: str = "dm_control"
    env_id: str = "walker-walk"
    num_envs: int = 1
    num_eval_envs: int = 5
    eval_freq: int = 5_000
    num_eval_steps: int = 500


def _ncrl_algorithm_cls() -> type:
    from rl_garden.algorithms.ncrl import NCRL

    return NCRL


registry.register("ncrl", NCRLArgs, run_ncrl, algorithm_cls=_ncrl_algorithm_cls)
