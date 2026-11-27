"""ExPLORe run function."""

from __future__ import annotations



def build_explore(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ExPLORe
    from rl_garden.common.cli_args import policy_module_kwargs
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.entrypoints.sac import sac_common_kwargs, sac_observation_kwargs

    agent = construct_agent(
        ExPLORe,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
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
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        exclude_bias_from_decay=args.exclude_bias_from_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        std_parameterization=args.std_parameterization,
        **sac_common_kwargs(
            args, env, eval_env, logger, checkpoint_dir, sac_observation_kwargs(args),
        ),
        **policy_module_kwargs(args),
    )
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
                "[explore] "
                f"offline_dataset={args.offline_dataset} "
                f"backend={args.dataset_backend} "
                f"loaded_transitions={loaded} "
                f"offline_data_ratio={args.offline_data_ratio} "
                f"offline_relabel_type={args.offline_relabel_type}",
                flush=True,
            )
    return agent


def run_explore(args: ExPLOREArgs) -> None:
    from rl_garden.training.online.runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    from rl_garden.common.env_args import make_env_request

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_explore,
        post_learn=lambda agent: getattr(agent.replay_buffer, "flush", lambda: None)(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Literal

from rl_garden.training.online.entrypoints.rlpd import RLPDArgs
from rl_garden.training.online.registry import registry


@dataclass
class ExPLOREArgs(RLPDArgs):
    """ExPLORe -- RLPD plus optimistic reward relabeling of the offline prior
    data, plus an optional RND novelty bonus. State-based (Box observation
    space) only.
    """

    offline_relabel_type: Literal["gt", "pred", "min"] = "pred"
    use_rnd_offline: bool = False
    use_rnd_online: bool = False
    rnd_coeff: float = 1.0
    relabeler_hidden_dims: tuple[int, ...] = (256, 256)
    relabeler_lr: float = 3e-4
    rnd_hidden_dims: tuple[int, ...] = (256, 256)
    rnd_feature_dim: int = 256
    rnd_lr: float = 3e-4


def _explore_algorithm_cls() -> type:
    from rl_garden.algorithms import ExPLORe

    return ExPLORe


registry.register(
    "explore",
    ExPLOREArgs,
    run_explore,
    algorithm_cls=_explore_algorithm_cls,
    build_agent=build_explore,
    training_only_inputs=("offline_dataset", "load_actor_checkpoint"),
)
