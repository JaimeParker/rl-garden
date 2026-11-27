from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.hyperparameters.cql import CQLParams
from rl_garden.training.offline.args import (
    OfflineCommonArgs,
    OfflineCompileArgs,
    OfflineCQLArgs,
    OfflineSACNetworkArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class CQLArgs(
    OfflineCommonArgs,
    OfflineSACNetworkArgs,
    OfflineCompileArgs,
    OfflineCQLArgs,
    PolicyModuleArgs,
):
    """Conservative Q-learning offline pretraining."""

    gamma: float = 0.99
    utd: float = 1.0


def _cql_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None,
    *, params_class: type = CQLParams,
) -> dict:
    from rl_garden.common.cli_args import (
        policy_module_kwargs,
        resolve_critic_encoder_config,
        resolve_obs_groups_config,
    )

    from rl_garden.hyperparameters import params_kwargs

    net_arch = args.net_arch if args.net_arch is not None else {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }
    kwargs = {
        **params_kwargs(args, params_class),
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "utd": args.utd,
        "use_compile": args.use_compile,
        "compile_mode": args.compile_mode,
        "net_arch": net_arch,
        "seed": args.seed,
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
    kwargs.update(policy_module_kwargs(args))
    return kwargs


def build_cql(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import CQL
    from rl_garden.training.inspection import construct_agent

    return construct_agent(CQL, **_cql_kwargs(args, env_spec, logger, eval_env, params_class=CQLParams))


def run_cql(args: CQLArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_cql)




def _cql_algorithm_cls() -> type:
    from rl_garden.algorithms import CQL

    return CQL

registry.register(
    "cql", CQLArgs, run_cql, algorithm_cls=_cql_algorithm_cls, build_agent=build_cql
)
