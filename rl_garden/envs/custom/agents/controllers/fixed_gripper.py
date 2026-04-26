"""Fixed gripper controller that keeps gripper always closed (0-dim action space)."""

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
)
from mani_skill.utils.structs.types import Array


class FixedGripperController(PDJointPosMimicController):
    """Gripper controller with 0-dim action space; gripper is always driven to closed position."""

    config: "FixedGripperControllerConfig"

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            np.empty(0), np.empty(0), dtype=np.float32
        )

    def _set_fixed_gripper_target(self):
        closed_value = torch.as_tensor(
            self.config.fixed_target,
            device=self.device,
            dtype=self._target_qpos.dtype,
        )
        self._target_qpos[:, self.control_joint_indices] = closed_value
        self._target_qpos[:, self.mimic_joint_indices] = (
            self._target_qpos[:, self.mimic_control_joint_indices]
            * self._multiplier[None, :]
            + self._offset[None, :]
        )
        self.set_drive_targets(self._target_qpos)

    def reset(self):
        super().reset()
        self._set_fixed_gripper_target()

    def set_action(self, action: Array):
        self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos.clone()
        self._target_qpos = self._start_qpos.clone()
        self._set_fixed_gripper_target()


@dataclass
class FixedGripperControllerConfig(PDJointPosMimicControllerConfig):
    """Config for FixedGripperController. Gripper is always at fixed_target (closed)."""

    controller_cls = FixedGripperController
    fixed_target: float = 0
    """Target position for the control joint when closed (Panda default: 0)."""
