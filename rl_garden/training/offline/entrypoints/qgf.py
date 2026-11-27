"""QGF offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.hyperparameters import params_kwargs
from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineQGFArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class QGFArgs(
    OfflineQGFArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
):
    """QGF (Q-Guided Flow) offline pretraining. Box or Dict (vision) observations."""


def _qgf_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.algorithms import QGF

    kwargs = {
        **params_kwargs(args, QGF.params_class),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "horizon_length": args.horizon_length,
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
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        kwargs["encoder_sharing"] = args.encoder_sharing
    return kwargs


def build_qgf(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import QGF
    from rl_garden.training.inspection import construct_agent

    return construct_agent(QGF, **_qgf_kwargs(args, env_spec, logger, eval_env))


def run_qgf(args: QGFArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_qgf)




def _qgf_algorithm_cls() -> type:
    from rl_garden.algorithms import QGF

    return QGF

registry.register(
    "qgf", QGFArgs, run_qgf, algorithm_cls=_qgf_algorithm_cls, build_agent=build_qgf
)
