"""Fixed, invertible physical-unit scaling of the 146D open_laptop state.

Only residual network features are transformed. Raw observations/replay, ACT,
warmup, physics and RNG are untouched. Constants are not running statistics.
"""
import math

import torch

from .selected_state import SelectedStateExtractor
from rl_garden.envs.robotwin.state_sources import SOURCE_FIELDS


PROFILE = "open_laptop_aloha_physical_v1"
KEYS = ("state", "state_open_laptop_robot_v5")


def scaling_spec():
    """Explicit field classification; fail closed if the schema is extended."""
    fields, divisors, units = [], [], []
    for side in ("left", "right"):
        for j in range(6):
            fields.append(f"command_state.{side}.arm_{j}")
            divisors.append(math.pi)
            units.append("rad")
        fields.append(f"command_state.{side}.gripper")
        divisors.append(1.)
        units.append("dimensionless")
    for field in SOURCE_FIELDS["open_laptop_robot_v5"]:
        pieces = field.split(".")
        tail = pieces[-1]
        if len(pieces) == 3 and pieces[0] in ("left", "right") and pieces[1] in ("position", "velocity"):
            if tail.startswith("arm_"):
                divisor, unit = math.pi, "rad" if pieces[1] == "position" else "rad/s"
            elif tail in ("finger_0", "finger_1"):
                # Verified aloha-agilex URDF: four prismatic joints, 0..0.04765m.
                divisor, unit = (0.04765, "m") if pieces[1] == "position" else (0.1, "m/s")
            else:
                raise ValueError(f"Unclassified joint: {field}")
        elif field in ("hinge.position", "hinge.lower", "hinge.upper", "hinge.velocity"):
            divisor, unit = math.pi, "rad/s" if tail == "velocity" else "rad"
        elif tail.startswith("extent_") or tail.startswith("geometry_center_") or ".to_contact_" in field:
            divisor, unit = 0.25, "m"
        elif tail in ("qw", "qx", "qy", "qz") or field.startswith("hinge.axis."):
            divisor, unit = 1., "dimensionless"
        elif tail == "command_valid" or tail.endswith("_any_tick") or tail.endswith("_last_tick"):
            divisor, unit = 1., "binary"
        elif tail in ("x", "y", "z"):
            divisor, unit = 1., "m"
        else:
            raise ValueError(f"Unclassified state field: {field}")
        fields.append(field)
        divisors.append(divisor)
        units.append(unit)
    if len(fields) != 146 or len(set(fields)) != 146:
        raise ValueError("Physical-v1 requires the exact 146D schema")
    return fields, divisors, units


class ScaledTaskStateExtractor(SelectedStateExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, keys=KEYS)
        if self.dims != (14, 132):
            raise ValueError("Physical-v1 requires 14+132 state dimensions")
        _, divisors, _ = scaling_spec()
        self.register_buffer("state_divisor", torch.tensor(divisors, dtype=torch.float32))

    def forward(self, obs):
        # Out-of-place: ACT/warmup and replay retain their native-unit inputs.
        # No clipping: unusually large velocities remain distinguishable.
        return super().forward(obs) / self.state_divisor
