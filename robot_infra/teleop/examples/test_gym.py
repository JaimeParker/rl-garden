import os
import sys
import argparse

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PegInsertionSidePegOnly-v1 environment with zero actions."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PegInsertionSidePegOnly-v1",
        help="Environment ID to visualize.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_ee_delta_pose",
        help="Control mode.",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="cpu",
        help="ManiSkill simulation backend.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default="gpu",
        help="ManiSkill render backend.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum total steps to run. Set <= 0 to run forever.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env

    cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=1,
        obs_mode="rgb",
        include_state=True,
        control_mode=args.control_mode,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode="normalized_dense",
        robot_uids="panda_wristcam_gripper_closed_wo_norm",
        fix_peg_pose=False,
        fix_box=True,
        ignore_terminations=False,
    )
    env = make_maniskill_env(cfg)
    env.auto_reset = False
    try:
        obs, info = env.reset()
        step = 0
        while args.max_steps <= 0 or step < args.max_steps:
            action = torch.zeros(env.action_space.shape, device=obs_device(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            print(step)
    finally:
        env.close()


def obs_device(obs):
    if isinstance(obs, dict):
        return next(iter(obs.values())).device
    return obs.device


if __name__ == "__main__":
    main()
