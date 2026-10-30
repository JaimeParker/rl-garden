"""146D state residual on frozen RGB ACT, with truncated bootstrap."""
from dataclasses import dataclass, field
from typing import ClassVar

from rl_garden.common.env_args import RoboTwinConfig
from rl_garden.training.online.task_state_residual_sac_v3 import TaskStateResidualSACV3Args
from rl_garden.training.online.task_state_residual_sac import run_task_state_residual_sac
from rl_garden.training.online._registry import registry


@dataclass
class TaskStateResidualSACV5Args(TaskStateResidualSACV3Args):
    STATE_VERSION: ClassVar[int] = 5
    robotwin: RoboTwinConfig = field(default_factory=lambda: RoboTwinConfig(
        extra_state=('open_laptop_robot_v5',)))


registry.register('task_state_residual_sac_v5', TaskStateResidualSACV5Args,
                  run_task_state_residual_sac)
