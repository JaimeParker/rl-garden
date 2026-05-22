from __future__ import annotations

import torch

from robot_infra.controller.core.impedance_torch import ImpedanceCommand
from robot_infra.controller.core.pose_torch import (
    axis_angle_to_quat,
    compose_pose,
    normalize_quat,
)


class DeltaPoseCommandAdapter:
    """Maintains an EE equilibrium pose and applies local delta-pose actions."""

    def __init__(self):
        self.target_pos: torch.Tensor | None = None
        self.target_quat: torch.Tensor | None = None

    def reset(self, pos: torch.Tensor, quat: torch.Tensor):
        self.target_pos = pos.clone()
        self.target_quat = normalize_quat(quat.clone())

    def set_action(self, action: torch.Tensor):
        if self.target_pos is None or self.target_quat is None:
            raise RuntimeError("Adapter must be reset before set_action.")
        delta_pos = action[..., :3]
        delta_quat = axis_angle_to_quat(action[..., 3:6])
        self.target_pos, self.target_quat = compose_pose(
            self.target_pos, self.target_quat, delta_pos, delta_quat
        )

    def command(self, nullspace_qpos: torch.Tensor | None = None) -> ImpedanceCommand:
        if self.target_pos is None or self.target_quat is None:
            raise RuntimeError("Adapter must be reset before command.")
        return ImpedanceCommand(
            target_pos=self.target_pos,
            target_quat=self.target_quat,
            nullspace_qpos=nullspace_qpos,
        )


class TwistCommandAdapter(DeltaPoseCommandAdapter):
    """Maintains an EE equilibrium pose and applies EE body-frame twist increments."""

    def set_action(self, action: torch.Tensor):
        # For one control interval, a small twist increment has the same SE(3)
        # representation as a local delta pose [v, omega].
        super().set_action(action)

