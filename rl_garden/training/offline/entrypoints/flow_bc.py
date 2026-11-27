from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineFlowBCArgs,
)
from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.flow_bc import FlowBCParams
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class FlowBCArgs(OfflineFlowBCArgs, OfflineCommonArgs, OfflineDeviceArgs):
    """Flow-matching behavior cloning offline pretraining."""

    net_arch: Optional[tuple[int, ...]] = (512, 512, 512, 512)


def _flow_bc_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.common.cli_args import resolve_obs_groups_config
    kwargs = {
        **params_kwargs(args, FlowBCParams),
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
        # Passed unconditionally rather than gated on --obs.is_visual: the
        # dataset's own stored keys decide state-vs-Dict shape, not --obs
        # (see infer_specs_from_h5), so --obs.is_visual can't be trusted
        # here. A state-only dataset combined with a non-default
        # image-related --encoder.* field now raises ObservationContractError
        # (build_observation_encoder) instead of silently ignoring it.
        "encoder_config": args.encoder,
        "obs_groups": resolve_obs_groups_config(args),
    }
    return kwargs


def build_flow_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FlowBC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FlowBC, **_flow_bc_kwargs(args, env_spec, logger, eval_env))


def run_flow_bc(args: FlowBCArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_flow_bc)




def _flow_bc_algorithm_cls() -> type:
    from rl_garden.algorithms import FlowBC

    return FlowBC

registry.register(
    "flow_bc",
    FlowBCArgs,
    run_flow_bc,
    algorithm_cls=_flow_bc_algorithm_cls,
    build_agent=build_flow_bc,
)
