from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineMeanFlowBCArgs,
)
from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.mean_flow_bc import MeanFlowBCParams
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class MeanFlowBCArgs(OfflineMeanFlowBCArgs, OfflineCommonArgs, OfflineDeviceArgs):
    """MeanFlow behavior cloning offline pretraining."""

    net_arch: Optional[tuple[int, ...]] = (512, 512, 512, 512)


def _mean_flow_bc_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.common.cli_args import resolve_obs_groups_config
    kwargs = {
        **params_kwargs(args, MeanFlowBCParams),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "offline_sampling": args.offline_sampling,
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "eval_env": eval_env,
        "eval_freq": args.eval_freq if eval_env is not None else 0,
        "num_eval_steps": args.num_eval_steps,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
        # No critic here, so encoder_sharing is irrelevant (see algorithm
        # module docstring); no critic_encoder_config either.
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "image_augmentation_seed": args.seed,
    }
    return kwargs


def build_mean_flow_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import MeanFlowBC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(
        MeanFlowBC, **_mean_flow_bc_kwargs(args, env_spec, logger, eval_env)
    )


def run_mean_flow_bc(args: MeanFlowBCArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_mean_flow_bc)




def _mean_flow_bc_algorithm_cls() -> type:
    from rl_garden.algorithms import MeanFlowBC

    return MeanFlowBC

registry.register(
    "mean_flow_bc",
    MeanFlowBCArgs,
    run_mean_flow_bc,
    algorithm_cls=_mean_flow_bc_algorithm_cls,
    build_agent=build_mean_flow_bc,
)
