"""SpaceMouse-backed teleop source, adapted from HIL-SERL's
``SpacemouseIntervention`` gym wrapper (``franka_env/envs/wrappers.py``) to
this repo's ``TeleOpSample``-producing device-source interface (matching
``EETwistTeleOpWrapper``'s ``reset()``/``poll() -> TeleOpSample`` shape, so
``TeleopInterventionWrapper`` can use either backend interchangeably).

Unlike Pico (absolute hand pose over ZMQ, needs ``HandPoseToEETwist``'s
bind+diff logic), a SpaceMouse is a rate/velocity-control device -- its raw
6-axis reading already *is* a twist, matching HIL-SERL's own
``SpacemouseIntervention`` which uses the raw reading directly with no pose
integration.
"""
from __future__ import annotations

import numpy as np

from robot_infra.teleop.spacemouse.spacemouse_expert import SpaceMouseExpert
from robot_infra.teleop.utils.telo_op_control_twist import TeleOpSample

DEFAULT_INTERVENTION_THRESHOLD = 1e-3


class SpaceMouseTeleOpWrapper:
    def __init__(self, intervention_threshold: float = DEFAULT_INTERVENTION_THRESHOLD) -> None:
        self.expert = SpaceMouseExpert()
        self.intervention_threshold = float(intervention_threshold)
        self.last_gripper = 1.0

    def reset(self, *, episode_end_pressed: bool = False) -> None:
        del episode_end_pressed
        self.last_gripper = 1.0

    def poll(self) -> TeleOpSample:
        twist, buttons = self.expert.get_action()
        twist = np.asarray(twist, dtype=np.float32)

        gripper_intervened = False
        if len(buttons) >= 1 and buttons[0]:
            self.last_gripper = -1.0  # close
            gripper_intervened = True
        elif len(buttons) >= 2 and buttons[1]:
            self.last_gripper = 1.0  # open
            gripper_intervened = True

        intervened = bool(np.linalg.norm(twist) > self.intervention_threshold) or gripper_intervened
        action = np.concatenate([twist, [self.last_gripper]]).astype(np.float32)
        return TeleOpSample(
            action=action,
            twist=twist,
            gripper=self.last_gripper,
            bind_pressed=False,
            episode_end=False,
            intervened=intervened,
        )

    def get_ee_twist(self) -> np.ndarray:
        return self.poll().twist

    def get_action(self) -> np.ndarray:
        return self.poll().action

    def close(self) -> None:
        self.expert.close()
