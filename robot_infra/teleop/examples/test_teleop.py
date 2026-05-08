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

TWIST_TRACKING_RATIO = 0.39
TWIST_TRACKING_COMPENSATION = 1.0 / TWIST_TRACKING_RATIO
RAW_TWIST_LIMIT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teleoperate PegInsertionSidePegOnly-v1 with ee_twist actions."
    )
    parser.add_argument("--zmq-url", type=str, default="tcp://192.168.6.2:7777")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--env-id", type=str, default="PegInsertionSidePegOnly-v1")
    parser.add_argument(
        "--robot-uids", type=str, default="panda_wristcam_gripper_closed_wo_norm"
    )
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="gpu")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dt", type=float, default=1/30)
    parser.add_argument("--pos-scale", type=float, default=TWIST_TRACKING_COMPENSATION)
    parser.add_argument("--rot-scale", type=float, default=TWIST_TRACKING_COMPENSATION)
    parser.add_argument("--twist-limit", type=float, default=RAW_TWIST_LIMIT * TWIST_TRACKING_COMPENSATION)
    return parser.parse_args()

def main():
    args = parse_args()
    import torch
    from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env
    from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=1,
        obs_mode="rgb",
        include_state=True,
        control_mode="pd_ee_twist",
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode="normalized_dense",
        # robot_uids=args.robot_uids,
        # fix_peg_pose=False,
        # fix_box=True,
        ignore_terminations=False,
    )
    env = make_maniskill_env(env_cfg)
    teleop = None
    try:
        env.auto_reset = False
        twist_normalizer = get_twist_normalizer(env)
        if twist_normalizer is None:
            print("normalize_action=False: using raw ee_twist actions.")
        else:
            print("normalize_action=True: scaling ee_twist actions to [-1, 1].")
        print(
            f"twist compensation: pos_scale={args.pos_scale:.3f}, "
            f"rot_scale={args.rot_scale:.3f}, twist_limit={args.twist_limit:.3f}"
        )
        teleop = EETwistTeleOpWrapper(
            zmq_url=args.zmq_url,
            hand=args.hand,
            pos_scale=args.pos_scale,
            rot_scale=args.rot_scale,
            twist_limit=args.twist_limit,
        )
        obs, info = env.reset()
        step = 0
        while args.max_steps <= 0 or step < args.max_steps:
            teleop_action = normalize_twist(teleop.get_action(), twist_normalizer)
            teleop_action = fit_action(teleop_action, env.action_space.shape)
            action = torch.as_tensor(
                teleop_action, dtype=torch.float32, device=obs_device(obs)
            ).reshape(env.action_space.shape)
            print("action:", action)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            print(step)
            time.sleep(args.dt)
    finally:
        if teleop is not None:
            teleop.close()
        env.close()


def obs_device(obs):
    if isinstance(obs, dict):
        return next(iter(obs.values())).device
    return obs.device


def fit_action(action, shape):
    size = 1
    for dim in shape:
        size *= dim
    if action.size >= size:
        return action[:size]
    import numpy as np

    return np.pad(action, (0, size - action.size))


def get_twist_normalizer(env):
    controller = env.unwrapped.agent.controller
    arm_controller = getattr(controller, "controllers", {}).get("arm", controller)
    cfg = arm_controller.config
    normalize_action = getattr(arm_controller, "_normalize_action", False)
    if not normalize_action:
        return None

    lower = np.broadcast_to(np.asarray(cfg.twist_lower, dtype=np.float32), 6)
    upper = np.broadcast_to(np.asarray(cfg.twist_upper, dtype=np.float32), 6)
    if np.any(upper <= lower):
        raise ValueError(f"Invalid twist bounds: lower={lower}, upper={upper}")
    return lower, upper


def normalize_twist(action, normalizer):
    if normalizer is None:
        return action

    lower, upper = normalizer
    action = action.copy()
    raw_twist = np.clip(action[:6], lower, upper)
    action[:6] = 2.0 * (raw_twist - lower) / (upper - lower) - 1.0
    return action


if __name__ == "__main__":
    main()
