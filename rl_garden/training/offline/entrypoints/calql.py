from dataclasses import dataclass

from rl_garden.hyperparameters.cql import CalQLParams

from rl_garden.training.offline.args import OfflineCalQLArgs
from rl_garden.training.offline.entrypoints.cql import CQLArgs, _cql_kwargs
from rl_garden.training.offline.registry import registry


@dataclass
class CalQLArgs(CQLArgs, OfflineCalQLArgs):
    """Calibrated Q-learning offline pretraining."""


def build_calql(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import CalQL
    from rl_garden.training.inspection import construct_agent

    return construct_agent(
        CalQL,
        **_cql_kwargs(args, env_spec, logger, eval_env, params_class=CalQLParams),
    )


def run_calql(args: CalQLArgs) -> None:
    from rl_garden.training.offline.runner import run_offline

    run_offline(args, build_agent=build_calql)


def _calql_algorithm_cls() -> type:
    from rl_garden.algorithms import CalQL

    return CalQL


registry.register(
    "calql", CalQLArgs, run_calql, algorithm_cls=_calql_algorithm_cls, build_agent=build_calql
)
