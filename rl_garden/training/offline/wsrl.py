from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import OfflineWSRLArgs
from rl_garden.training.offline._registry import registry
from rl_garden.training.offline.calql import CalQLArgs
from rl_garden.training.offline.cql import _cql_kwargs

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class WSRLOfflineArgs(CalQLArgs, OfflineWSRLArgs):
    """WSRL-compatible offline checkpoint pretraining."""


def _wsrl_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    kwargs = _cql_kwargs(args, env_spec, logger, eval_env)
    kwargs.update(
        learning_starts=0,
        training_freq=args.training_freq,
    )
    return kwargs


def build_wsrl(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import WSRL

    return WSRL(
        **_wsrl_kwargs(args, env_spec, logger, eval_env),
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
    )


def run_wsrl(args: WSRLOfflineArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_wsrl)


registry.register("wsrl", WSRLOfflineArgs, run_wsrl)
