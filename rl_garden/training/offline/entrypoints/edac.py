"""EDAC offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineEDACArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class EDACArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineEDACArgs,
    PolicyModuleArgs,
):
    """EDAC offline pretraining. Box or Dict (vision) observations -- EDAC
    is an ``OfflineSAC`` subclass, which supports vision through
    ``SACCore`` (``OfflineCommonArgs`` already composes ``ObservationArgs``,
    so no extra base is needed)."""

    gamma: float = 0.99
    alpha_lr: Optional[float] = 3e-4
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = (256, 256, 256)


def _edac_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.common.cli_args import (
        policy_module_kwargs,
        resolve_critic_encoder_config,
        resolve_obs_groups_config,
    )

    from rl_garden.hyperparameters import params_kwargs
    from rl_garden.hyperparameters.offline_sac import EDACParams

    kwargs = {
        **params_kwargs(args, EDACParams),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "net_arch": (list(args.net_arch) if args.net_arch is not None
                     and not isinstance(args.net_arch, dict) else args.net_arch),
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        kwargs["encoder_sharing"] = args.encoder_sharing
    kwargs.update(policy_module_kwargs(args))
    return kwargs


def build_edac(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import EDAC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(EDAC, **_edac_kwargs(args, env_spec, logger, eval_env))


def run_edac(args: EDACArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_edac)


def _edac_algorithm_cls() -> type:
    from rl_garden.algorithms import EDAC

    return EDAC


registry.register(
    "edac", EDACArgs, run_edac, algorithm_cls=_edac_algorithm_cls, build_agent=build_edac
)
