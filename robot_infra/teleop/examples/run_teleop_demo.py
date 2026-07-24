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


@dataclass
class Args:
    zmq_url: str = "tcp://192.168.6.2:7777"
    device: Literal["pico", "spacemouse"] = "pico"
    hand: Literal["left", "right"] = "right"
    env_id: str = "PegInsertionSidePegOnly-v1"
    control_mode: str = "pd_ee_twist"
    robot_uids: str = "panda_wristcam_gripper_closed_wo_norm"
    sim_backend: str = "cpu"
    render_backend: str = "gpu"
    max_steps: int = -1
    dt: float = 1 / 30
    pos_scale: Optional[float] = None
    rot_scale: Optional[float] = None
    twist_limit: Optional[float] = None
    intervention_threshold: float = 1e-4
    init_timeout_s: float = 120.0
    spacemouse_index: int = 0
    spacemouse_scale: float = 1.0


def main():
    args = tyro.cli(Args)
    import torch
    from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=1,
        obs_mode="rgb",
        include_state=True,
        control_mode=args.control_mode,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode="normalized_dense",
        robot_uids=args.robot_uids,
        fix_peg_pose=False,
        fix_box=True,
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
        teleop = make_demo_teleop(args)
        obs, info = env.reset()
        step = 0
        while args.max_steps <= 0 or step < args.max_steps:
            sample = teleop.poll()
            teleop_action = normalize_twist(sample.action, twist_normalizer)
            teleop_action = fit_action(teleop_action, env.action_space.shape)
            action = torch.as_tensor(
                teleop_action, dtype=torch.float32, device=obs_device(obs)
            ).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            print(
                "step:",
                step,
                "twist:",
                sample.twist,
                "gripper:",
                sample.gripper,
                "intervened:",
                sample.intervened,
            )
            time.sleep(args.dt)
    finally:
        if teleop is not None:
            teleop.close()
        env.close()


def obs_device(obs):
    if isinstance(obs, dict):
        return next(iter(obs.values())).device
    return obs.device


def make_demo_teleop(args):
    if args.device == "spacemouse":
        from robot_infra.teleop.spacemouse import SpaceMouseTeleOpWrapper

        return SpaceMouseTeleOpWrapper(
            intervention_threshold=args.intervention_threshold,
            spacemouse_index=args.spacemouse_index,
            spacemouse_scale=args.spacemouse_scale,
        )

    from robot_infra.teleop.source import make_teleop_source

    return make_teleop_source(
        zmq_url=args.zmq_url,
        hand=args.hand,
        device=args.device,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        twist_limit=args.twist_limit,
        intervention_threshold=args.intervention_threshold,
        init_timeout_s=args.init_timeout_s,
        spacemouse_index=args.spacemouse_index,
    )


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
