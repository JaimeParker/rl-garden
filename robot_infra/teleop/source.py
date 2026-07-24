from __future__ import annotations

from typing import Literal, Optional, Protocol

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper, TeleOpSample


class TeleOpSource(Protocol):
    def poll(self) -> TeleOpSample:
        ...

    def reset(self, *, episode_end_pressed: bool = False) -> None:
        ...

    def close(self) -> None:
        ...


def make_teleop_source(
    *,
    device: Literal["pico", "spacemouse"],
    zmq_url: str = "tcp://192.168.6.2:7777",
    hand: Literal["left", "right"] = "right",
    pos_scale: Optional[float] = None,
    rot_scale: Optional[float] = None,
    twist_limit: Optional[float] = None,
    intervention_threshold: float = 1e-4,
    init_timeout_s: float = 120.0,
    spacemouse_index: int = 0,
) -> TeleOpSource:
    """Create a teleoperation source with the repo's TeleOpSample interface."""
    if device == "pico":
        teleop_kwargs = dict(
            zmq_url=zmq_url,
            hand=hand,
            device="pico",
            intervention_threshold=intervention_threshold,
            init_timeout_s=init_timeout_s,
        )
        if pos_scale is not None:
            teleop_kwargs["pos_scale"] = pos_scale
        if rot_scale is not None:
            teleop_kwargs["rot_scale"] = rot_scale
        if twist_limit is not None:
            teleop_kwargs["twist_limit"] = twist_limit
        return EETwistTeleOpWrapper(**teleop_kwargs)

    if device == "spacemouse":
        from robot_infra.teleop.spacemouse import SpaceMouseTeleOpWrapper

        return SpaceMouseTeleOpWrapper(
            intervention_threshold=intervention_threshold,
            spacemouse_index=spacemouse_index,
        )

    raise ValueError(f"device must be 'pico' or 'spacemouse', got {device!r}.")
