from dataclasses import dataclass

from rl_garden.training.offline._args import OfflineCalQLArgs
from rl_garden.training.offline._registry import registry
from rl_garden.training.offline.cql import CQLArgs, _cql_kwargs


@dataclass
class CalQLArgs(CQLArgs, OfflineCalQLArgs):
    """Calibrated Q-learning offline pretraining."""


def build_calql(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import CalQL

    return CalQL(
        **_cql_kwargs(args, env_spec, logger, eval_env),
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
    )


def run_calql(args: CalQLArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_calql)


registry.register("calql", CalQLArgs, run_calql)
