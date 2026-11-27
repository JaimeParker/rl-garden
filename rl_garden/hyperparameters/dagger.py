"""DAgger policy-update hyperparameters."""
from dataclasses import dataclass

from rl_garden.hyperparameters.bc import BCParams


@dataclass
class DAggerParams(BCParams):
    """BC policy updates; collection rounds remain constructor keywords."""
