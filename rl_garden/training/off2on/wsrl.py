"""WSRL offline-to-online training registration."""
from dataclasses import dataclass

from rl_garden.common.cli_args import VisionWSRLTrainingArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on._registry import registry


@dataclass
class WSRLOff2OnArgs(VisionWSRLTrainingArgs, EnvBackendArgs):
    """WSRL args; visual defaults. For state obs pass --obs_mode state."""


def run_wsrl(args: WSRLOff2OnArgs) -> None:
    from rl_garden.training.off2on._wsrl_runner import run_wsrl as run

    run(args)


registry.register("wsrl", WSRLOff2OnArgs, run_wsrl)
