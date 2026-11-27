"""FAC offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from rl_garden.hyperparameters import params_kwargs
from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineFACArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class FACArgs(
    OfflineFACArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
):
    """FAC (Flow Actor-Critic) offline pretraining. Box or Dict (vision)
    observations."""

    normalize_q_loss: bool = True
    net_arch: Optional[tuple[int, ...]] = (512, 512, 512, 512)
    q_agg: Literal['mean', 'min'] = 'min'
    gamma: float = 0.995


def _fac_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.algorithms import FAC
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config

    kwargs = {
        **params_kwargs(args, FAC.params_class),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "alpha": args.alpha,
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


def build_fac(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FAC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FAC, **_fac_kwargs(args, env_spec, logger, eval_env))


def run_fac(args: FACArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    # No --num_offline_steps vs. --bc_pretrain_steps relationship to validate:
    # FACCore.train() runs the BC-flow prelude (pretrain + logp cache) to
    # completion on the first train() call regardless of --num_offline_steps,
    # and --num_offline_steps counts only actor-critic updates afterward,
    # matching the reference's offline_steps (rl_garden/algorithms/fac.py
    # module docstring).
    run_offline(args, build_agent=build_fac)


def _fac_algorithm_cls() -> type:
    from rl_garden.algorithms import FAC

    return FAC


registry.register(
    "fac", FACArgs, run_fac, algorithm_cls=_fac_algorithm_cls, build_agent=build_fac
)
