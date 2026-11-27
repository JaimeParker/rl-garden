"""RLPD run function."""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs

from rl_garden.hyperparameters.sac import RLPDParams

from typing import Any


def build_rlpd(
    args,
    env,
    eval_env,
    logger,
    checkpoint_dir,
    *,
    algorithm_cls: type | None = None,
    **extra_kwargs: Any,
):
    """Build ``algorithm_cls`` (``RLPD`` when ``None``) from ``args``.
    ``extra_kwargs`` are merged into its constructor call after the keywords
    derived from ``args`` and win over them."""
    from rl_garden.algorithms import RLPD
    from rl_garden.common.cli_args import (
        policy_module_kwargs,
        resolve_critic_encoder_config,
        resolve_obs_groups_config,
    )
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.args import sac_initial_training_phase_from_args

    net_arch = args.net_arch if args.net_arch is not None else {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
        "critic_backbone_type": args.critic_backbone_type,
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    if algorithm_cls is None:
        algorithm_cls = RLPD
    agent_kwargs: dict[str, Any] = dict(
        **{
            **params_kwargs(args, RLPDParams),
            "net_arch": net_arch,
        },
        env=env,
        eval_env=eval_env,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        bootstrap_at_done=args.bootstrap_at_done,
        training_freq=args.training_freq,
        utd=args.utd,
        q_landscape_diagnostics=args.q_landscape_diagnostics,
        q_landscape_num_actions=args.q_landscape_num_actions,
        q_landscape_batch_size=args.q_landscape_batch_size,
        q_mc_diagnostics=args.q_mc_diagnostics,
        initial_training_phase=sac_initial_training_phase_from_args(args),
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
        **policy_module_kwargs(args),
    )
    agent_kwargs.update(extra_kwargs)
    agent = construct_agent(algorithm_cls, **agent_kwargs)
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    if args.load_actor_checkpoint is not None:
        agent.load_actor_checkpoint(args.load_actor_checkpoint)
    if args.offline_dataset is not None:
        if args.offline_buffer_size is None:
            raise ValueError(
                "--offline_buffer_size is required when --offline_dataset is set "
                "(RLPD's prior-data buffer has no cheap way to count "
                "h5/minari transitions ahead of time)."
            )
        loaded = agent.load_offline_replay_buffer(
            args.offline_dataset,
            backend=args.dataset_backend,
            num_traj=args.offline_num_traj,
            buffer_size=args.offline_buffer_size,
            offline_data_ratio=args.offline_data_ratio,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            success_key=args.success_key,
            backend_config=getattr(args, args.dataset_backend, None),
        )
        if args.std_log:
            print(
                "[rlpd] "
                f"offline_dataset={args.offline_dataset} "
                f"backend={args.dataset_backend} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio}",
                flush=True,
            )
    return agent


def run_rlpd(args: RLPDArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_rlpd,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import VisionSACTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class RLPDArgs(RLPDParams, VisionSACTrainingArgs, PolicyModuleArgs, EnvBackendArgs):
    """RLPD -- high UTD + REDQ-style critic ensemble subsampling + LayerNorm
    + offline/online prior-data mixing from step 0, on top of SAC.

    Env backend: ``--env_backend maniskill`` (default) or any other
    registered backend, e.g. ``d4rl_legacy``.
    """

    utd: float = 4.0



    # Registry-backed (rl_garden.buffers.dataset_backend_registry), not a
    # Literal: a new backend registers itself, no CLI arg change needed here.
    dataset_backend: str = "h5"

    offline_dataset: str | None = None
    offline_num_traj: int | None = None
    offline_buffer_size: int | None = None
    offline_data_ratio: float = 0.5
    # Applied to both the live env (via EnvRequest) and the offline dataset
    # loader, so online and offline rewards share one scale in the Bellman
    # target RLPD mixes them into. Matches off2on's Off2OnCommonArgs fields.
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    success_key: str | None = None


def _rlpd_algorithm_cls() -> type:
    from rl_garden.algorithms import RLPD

    return RLPD


registry.register(
    "rlpd",
    RLPDArgs,
    run_rlpd,
    algorithm_cls=_rlpd_algorithm_cls,
    build_agent=build_rlpd,
    training_only_inputs=("offline_dataset", "load_actor_checkpoint"),
)
