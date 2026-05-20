"""Adapter from RoboTwin task instances to rl-garden step/reset semantics."""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rl_garden.envs.robotwin.config import RoboTwinEnvConfig
from rl_garden.envs.robotwin.rewards import build_task_reward


@dataclass
class StepResult:
    obs: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def ensure_robotwin_importable(robotwin_root: Optional[str]) -> None:
    if robotwin_root is not None:
        root = os.path.abspath(robotwin_root)
        if root not in sys.path:
            sys.path.insert(0, root)


def make_task(task_name: str, robotwin_root: Optional[str] = None):
    ensure_robotwin_importable(robotwin_root)
    try:
        module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import RoboTwin task module. Install RoboTwin or set "
            "RoboTwinEnvConfig.robotwin_root to the RoboTwin repository root."
        ) from exc
    try:
        cls = getattr(module, task_name)
    except AttributeError as exc:
        raise AttributeError(
            f"RoboTwin task class {task_name!r} not found in envs.{task_name}."
        ) from exc
    return cls()


class RoboTwinTaskAdapter:
    """Owns one RoboTwin task instance and exposes one-env step/reset."""

    def __init__(
        self,
        env_id: int,
        cfg: RoboTwinEnvConfig,
        task_args: dict[str, Any],
        env_seed: Optional[int] = None,
    ) -> None:
        self.env_id = env_id
        self.cfg = cfg
        self.task_name = cfg.task_name
        self.task_args = dict(task_args)
        self.env_seed = env_seed if env_seed is not None else cfg.seed + env_id
        self.task = None
        self.elapsed_steps = 0
        self.last_dense_reward = 0.0

    def reset(self, env_seed: Optional[int] = None) -> dict[str, Any]:
        if env_seed is not None:
            self.env_seed = int(env_seed)
        self.close(
            clear_cache=(
                self.elapsed_steps > 0
                and self.elapsed_steps % self.cfg.clear_cache_freq == 0
            )
        )
        self.task = make_task(self.task_name, self.cfg.robotwin_root)
        args = dict(self.task_args)
        args.setdefault(
            "step_lim", self.cfg.step_lim or self.cfg.max_episode_steps or 400
        )
        args.setdefault("planner_backend", self.cfg.planner_backend)
        args.setdefault("embodiment", self.cfg.embodiment)
        args.setdefault("render_freq", 0)
        args.setdefault("eval_mode", True)
        args.setdefault("eval_video_log", False)
        args.setdefault("save_path", "./data")
        args.setdefault("clear_cache_freq", self.cfg.clear_cache_freq)
        _prepare_robotwin_task_args(args)
        self.task.setup_demo(now_ep_num=self.env_seed, seed=self.env_seed, **args)
        self.task.step_lim = int(args["step_lim"])
        self.task.take_action_cnt = 0
        self.task.run_steps = 0
        self.task.reward_step = 0
        self.task.eval_success = False
        self._install_helpers()
        if self.cfg.reward_mode == "dense":
            build_task_reward(self.task_name, self.task)
        self.elapsed_steps = 0
        self.last_dense_reward = 0.0
        return self.get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        if self.task is None:
            self.reset()
        assert self.task is not None
        action = self._to_robotwin_action(action)
        self._begin_reward_trace()
        self.task.take_action(action, action_type="qpos")
        self._end_reward_trace()
        self.elapsed_steps += 1
        success = bool(self.task.check_success())
        if success:
            self.task.eval_success = True
        truncated = self.elapsed_steps >= int(self.task.step_lim)
        reward = self._compute_reward(success)
        info = {
            "success": success,
            "instruction": self._instruction(),
            "env_seed": self.env_seed,
        }
        return StepResult(
            obs=self.get_obs(),
            reward=reward,
            terminated=success,
            truncated=bool(truncated),
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RoboTwin task has not been reset.")
        return _extract_robotwin_obs(self.task.get_obs(), self._instruction())

    def close(self, clear_cache: bool = True) -> None:
        if self.task is not None:
            self.task.close_env(clear_cache=clear_cache)
            self.task = None

    def _to_robotwin_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.cfg.action_dim:
            raise ValueError(
                f"Expected action_dim={self.cfg.action_dim}, got {action.shape[0]}."
            )
        if self.cfg.control_mode == "joint_pos":
            return action
        if self.cfg.control_mode != "delta_joint_pos":
            raise ValueError(
                f"Unsupported RoboTwin control mode {self.cfg.control_mode!r}."
            )
        assert self.task is not None
        left = np.asarray(self.task.robot.get_left_arm_jointState(), dtype=np.float32)
        right = np.asarray(self.task.robot.get_right_arm_jointState(), dtype=np.float32)
        left_arm_dim = len(left) - 1
        right_arm_dim = len(right) - 1
        if left_arm_dim + right_arm_dim + 2 != self.cfg.action_dim:
            raise ValueError(
                "RoboTwin robot joint state does not match action_dim: "
                f"{left_arm_dim}+{right_arm_dim}+2 != {self.cfg.action_dim}."
            )
        out = np.empty_like(action)
        out[:left_arm_dim] = (
            left[:left_arm_dim] + action[:left_arm_dim] * self.cfg.joint_delta_scale
        )
        out[left_arm_dim] = np.clip(
            left[left_arm_dim] + action[left_arm_dim] * self.cfg.gripper_delta_scale,
            0.0,
            1.0,
        )
        r0 = left_arm_dim + 1
        out[r0 : r0 + right_arm_dim] = (
            right[:right_arm_dim]
            + action[r0 : r0 + right_arm_dim] * self.cfg.joint_delta_scale
        )
        out[-1] = np.clip(
            right[-1] + action[-1] * self.cfg.gripper_delta_scale, 0.0, 1.0
        )
        return out

    def _compute_reward(self, success: bool) -> float:
        if self.cfg.reward_mode == "sparse":
            reward = 1.0 if success else 0.0
        elif success:
            reward = 1.0
        else:
            reward_obj = getattr(self.task, "reward", None)
            reward_obj.update()
            if reward_obj.is_fail():
                reward = 0.0
            else:
                reward = float(reward_obj.compute_reward())
        reward = reward * self.cfg.reward_scale + self.cfg.reward_bias
        if self.cfg.use_relative_reward:
            diff = reward - self.last_dense_reward
            self.last_dense_reward = reward
            return float(diff)
        self.last_dense_reward = reward
        return float(reward)

    def _begin_reward_trace(self) -> None:
        assert self.task is not None
        robot = self.task.robot
        self.task.episode_left_eef_poses = [
            np.asarray(robot.get_left_ee_pose(), dtype=np.float32)
        ]
        self.task.episode_right_eef_poses = [
            np.asarray(robot.get_right_ee_pose(), dtype=np.float32)
        ]
        self.task.episode_left_joint_states = [
            np.asarray(robot.get_left_arm_jointState(), dtype=np.float32)
        ]
        self.task.episode_right_joint_states = [
            np.asarray(robot.get_right_arm_jointState(), dtype=np.float32)
        ]

    def _end_reward_trace(self) -> None:
        assert self.task is not None
        robot = self.task.robot
        self.task.episode_left_eef_poses.append(
            np.asarray(robot.get_left_ee_pose(), dtype=np.float32)
        )
        self.task.episode_right_eef_poses.append(
            np.asarray(robot.get_right_ee_pose(), dtype=np.float32)
        )
        self.task.episode_left_joint_states.append(
            np.asarray(robot.get_left_arm_jointState(), dtype=np.float32)
        )
        self.task.episode_right_joint_states.append(
            np.asarray(robot.get_right_arm_jointState(), dtype=np.float32)
        )

    def _install_helpers(self) -> None:
        assert self.task is not None
        if not hasattr(self.task, "is_in_hand"):
            self.task.is_in_hand = types.MethodType(_is_in_hand, self.task)

    def _instruction(self) -> Optional[str]:
        if self.task is None:
            return None
        getter = getattr(self.task, "get_instruction", None)
        if callable(getter):
            return getter()
        return getattr(self.task, "instruction", None)


def _extract_robotwin_obs(
    raw_obs: dict[str, Any], instruction: Optional[str]
) -> dict[str, Any]:
    observation = raw_obs.get("observation", raw_obs)
    head = observation.get("head_camera", {})
    left = observation.get("left_camera", {})
    right = observation.get("right_camera", {})
    joint = raw_obs.get("joint_action", {})
    state = joint.get("vector")
    if state is None:
        state = raw_obs.get("state")
    if state is None:
        raise KeyError(
            "RoboTwin observation does not contain joint_action.vector/state."
        )
    return {
        "rgb": head.get("rgb"),
        "rgb_left_wrist": left.get("rgb"),
        "rgb_right_wrist": right.get("rgb"),
        "state": np.asarray(state, dtype=np.float32),
        "instruction": instruction,
    }


def _is_in_hand(task, actor):
    contacts = task.scene.get_contacts()
    left_count = 0
    right_count = 0
    actor_name = actor.get_name()
    for contact in contacts:
        names = [
            contact.bodies[0].entity.get_name(),
            contact.bodies[1].entity.get_name(),
        ]
        if actor_name not in names:
            continue
        other = names[1] if names[0] == actor_name else names[0]
        has_impulse = any(
            not all(impulse == 0 for impulse in point.impulse)
            for point in contact.points
        )
        if not has_impulse:
            continue
        if other in ("fl_link7", "fl_link8"):
            left_count += 1
        if other in ("fr_link7", "fr_link8"):
            right_count += 1
    return (
        left_count >= 2 and task.robot.is_left_gripper_close(),
        right_count >= 2 and task.robot.is_right_gripper_close(),
    )


def _prepare_robotwin_task_args(args: dict[str, Any]) -> None:
    args.setdefault(
        "domain_randomization",
        {
            "random_background": True,
            "cluttered_table": True,
            "clean_background_rate": 0.02,
            "random_head_camera_dis": 0,
            "random_table_height": 0.03,
            "random_light": True,
            "crazy_random_light_rate": 0.02,
        },
    )
    args.setdefault(
        "camera",
        {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
    )
    args.setdefault(
        "data_type",
        {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": False,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
    )
    args.setdefault("pcd_down_sample_num", 1024)
    args.setdefault("pcd_crop", True)
    if "left_robot_file" in args and "right_robot_file" in args:
        return
    try:
        import yaml
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH, ROOT_PATH
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboTwin task arguments require RoboTwin's envs package. "
            "Install RoboTwin or set robotwin_root."
        ) from exc

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        left_name = right_name = embodiment[0]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_name, right_name, args["embodiment_dis"] = embodiment
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(
            "RoboTwin embodiment must contain one item or left/right/distance."
        )

    def robot_file(name: str) -> str:
        rel = embodiment_types[name]["file_path"]
        return os.path.abspath(os.path.join(ROOT_PATH, rel))

    def robot_config(path: str) -> dict[str, Any]:
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_robot_file"] = robot_file(left_name)
    args["right_robot_file"] = robot_file(right_name)
    args["left_embodiment_config"] = robot_config(args["left_robot_file"])
    args["right_embodiment_config"] = robot_config(args["right_robot_file"])
    args["embodiment_name"] = (
        left_name if left_name == right_name else f"{left_name}_{right_name}"
    )
