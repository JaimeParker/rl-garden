"""Registers ``aux_split_rppo`` on rl-garden's online registry when imported.

The package's ``rlgarden.algorithms`` entry point names this module, so the
stock ``rl-garden-train-online`` and ``rl-garden-eval`` commands find the
algorithm once the package is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_garden.training.online.entrypoints.recurrent_ppo import (
    RecurrentPPOArgs,
    build_recurrent_ppo,
    run_recurrent_ppo,
)
from rl_garden.training.online.registry import registry

ALGORITHM_NAME = "aux_split_rppo"


@dataclass
class AuxSplitRecurrentPPOArgs(RecurrentPPOArgs):
    """RecurrentPPO with a two-head action mean and an auxiliary BCE head on
    ``infos["aux_target"]``."""

    aux_weight: float = 0.1
    aux_hidden: int = 16


def build_aux_split_rppo(args, env, eval_env, logger, checkpoint_dir):
    from rlg_downstream_example.algorithm import AuxSplitRecurrentPPO

    return build_recurrent_ppo(
        args,
        env,
        eval_env,
        logger,
        checkpoint_dir,
        algorithm_cls=AuxSplitRecurrentPPO,
        aux_weight=args.aux_weight,
        aux_hidden=args.aux_hidden,
    )


def run_aux_split_rppo(args: AuxSplitRecurrentPPOArgs) -> None:
    run_recurrent_ppo(args, build_agent=build_aux_split_rppo)


def _algorithm_cls() -> type:
    from rlg_downstream_example.algorithm import AuxSplitRecurrentPPO

    return AuxSplitRecurrentPPO


registry.register(
    ALGORITHM_NAME,
    AuxSplitRecurrentPPOArgs,
    run_aux_split_rppo,
    algorithm_cls=_algorithm_cls,
    build_agent=build_aux_split_rppo,
)
