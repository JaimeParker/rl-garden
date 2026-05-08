import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Print hand data and computed ee_twist.")
    parser.add_argument("--zmq-url", type=str, default="tcp://192.168.6.2:7777")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--dt", type=float, default=1 / 30)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--twist-limit", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    teleop = EETwistTeleOpWrapper(
        zmq_url=args.zmq_url,
        hand=args.hand,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        twist_limit=args.twist_limit,
    )
    print(f"Listening on {args.zmq_url}, hand={args.hand}")
    try:
        while True:
            sample = teleop.poll()
            if sample is None:
                print("No teleop data received.")
            else:
                print("raw:", sample["raw"])
                print(
                    "twist:",
                    sample["twist"],
                    "gripper:",
                    sample["gripper"],
                    "button:",
                    sample["button_state"].name,
                )
            time.sleep(args.dt)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
