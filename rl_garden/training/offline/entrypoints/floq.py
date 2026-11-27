"""FloQ offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.hyperparameters import params_kwargs
from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineFloQArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class FloQArgs(
    OfflineFloQArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
):
    """FloQ offline pretraining. Box or Dict (vision) observations."""

    net_arch: Optional[tuple[int, ...]] = (512, 512, 512, 512)


def _floq_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.algorithms import FloQ
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config

    kwargs = {
        **params_kwargs(args, FloQ.params_class),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "net_arch": list(args.net_arch) if args.net_arch is not None else None,
        "actor_use_group_norm": args.actor_use_group_norm,
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


def build_floq(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FloQ
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FloQ, **_floq_kwargs(args, env_spec, logger, eval_env))


def run_floq(args: FloQArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_floq)




def _flo_q_algorithm_cls() -> type:
    from rl_garden.algorithms import FloQ

    return FloQ

registry.register(
    "floq", FloQArgs, run_floq, algorithm_cls=_flo_q_algorithm_cls, build_agent=build_floq
)
