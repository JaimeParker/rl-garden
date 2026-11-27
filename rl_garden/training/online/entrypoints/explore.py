"""ExPLORe run function."""

from __future__ import annotations

from rl_garden.hyperparameters.sac import ExPLOReParams



def build_explore(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import ExPLORe
    from rl_garden.common.cli_args import policy_module_kwargs
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online.entrypoints.sac import sac_common_kwargs, sac_observation_kwargs

    agent = construct_agent(
        ExPLORe,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        **sac_common_kwargs(
            args, env, eval_env, logger, checkpoint_dir, sac_observation_kwargs(args),
            params_class=ExPLOReParams,
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

from rl_garden.training.online.entrypoints.rlpd import RLPDArgs
from rl_garden.training.online.registry import registry


@dataclass
class ExPLOREArgs(ExPLOReParams, RLPDArgs):
    """ExPLORe -- RLPD plus optimistic reward relabeling of the offline prior
    data, plus an optional RND novelty bonus. State-based (Box observation
    space) only.
    """



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
