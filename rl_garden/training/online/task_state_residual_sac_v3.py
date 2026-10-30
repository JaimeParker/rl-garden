"""120D task state residual: v2 state plus six rigid-part collision extents."""
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from rl_garden.common.env_args import RoboTwinConfig
from rl_garden.training.online.task_state_residual_sac import (
    TaskStateResidualSACArgs, run_task_state_residual_sac,
)
from rl_garden.training.online._registry import registry


@dataclass
class TaskStateResidualSACV3Args(TaskStateResidualSACArgs):
    STATE_VERSION: ClassVar[int] = 3
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    robotwin: RoboTwinConfig = field(default_factory=lambda: RoboTwinConfig(
        extra_state=("open_laptop_robot_v3",)))


registry.register("task_state_residual_sac_v3", TaskStateResidualSACV3Args,
                  run_task_state_residual_sac)
