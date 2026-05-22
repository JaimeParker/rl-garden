import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper


@dataclass
class Args:
    zmq_url: str = "tcp://192.168.6.2:7777"
    device: Literal["pico", "spacemouse"] = "pico"
    hand: Literal["left", "right"] = "right"
    dt: float = 1 / 30
    pos_scale: Optional[float] = None
    rot_scale: Optional[float] = None
    twist_limit: Optional[float] = None
    intervention_threshold: float = 1e-4


def main():
    args = tyro.cli(Args)
    np.set_printoptions(precision=4, suppress=True)

    teleop_kwargs = dict(
        zmq_url=args.zmq_url,
        hand=args.hand,
        device=args.device,
        intervention_threshold=args.intervention_threshold,
    )
    if args.pos_scale is not None:
        teleop_kwargs["pos_scale"] = args.pos_scale
    if args.rot_scale is not None:
        teleop_kwargs["rot_scale"] = args.rot_scale
    if args.twist_limit is not None:
        teleop_kwargs["twist_limit"] = args.twist_limit

    teleop = EETwistTeleOpWrapper(**teleop_kwargs)
    print(f"Listening on {args.zmq_url}, device={args.device}, hand={args.hand}")
    try:
        while True:
            sample = teleop.poll()
            print(
                "received:",
                teleop.last_received,
                "twist:",
                sample.twist,
                "gripper:",
                sample.gripper,
                "bind:",
                sample.bind_pressed,
                "episode_end:",
                sample.episode_end,
                "intervened:",
                sample.intervened,
            )
            time.sleep(args.dt)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
