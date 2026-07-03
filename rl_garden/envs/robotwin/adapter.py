"""Adapter from RoboTwin task instances to rl-garden step/reset semantics."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from rl_garden.envs.robotwin.config import RoboTwinEnvConfig
from rl_garden.envs.robotwin.rewards import build_task_reward


LOGGER = logging.getLogger(__name__)
UNSTABLE_RETRY_WARNING_INTERVAL = 20


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


@contextmanager
def robotwin_workdir(robotwin_root: Optional[str]) -> Iterator[None]:
    if robotwin_root is None:
        yield
        return
    previous = os.getcwd()
    os.chdir(os.path.abspath(robotwin_root))
    try:
        yield
    finally:
        os.chdir(previous)


def make_task(task_name: str, robotwin_root: Optional[str] = None):
    ensure_robotwin_importable(robotwin_root)
    try:
        with robotwin_workdir(robotwin_root):
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
        self._eval_video_index = 0

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
        args.setdefault("render_every_control_step", self.cfg.render_every_control_step)
        args.setdefault("need_topp", not self.cfg.disable_topp)
        if (
            self.cfg.control_step_cap is not None
            and args.get("control_step_cap") is None
        ):
            args["control_step_cap"] = self.cfg.control_step_cap
        has_domain_randomization = "domain_randomization" in args
        has_camera_config = "camera" in args
        _prepare_robotwin_task_args(args)
        domain_randomization = args["domain_randomization"]
        if has_domain_randomization:
            domain_randomization.setdefault("random_light", self.cfg.random_light)
            domain_randomization.setdefault(
                "crazy_random_light_rate", self.cfg.crazy_random_light_rate
            )
        else:
            domain_randomization["random_light"] = self.cfg.random_light
            domain_randomization["crazy_random_light_rate"] = (
                self.cfg.crazy_random_light_rate
            )
        camera_cfg = args["camera"]
        if has_camera_config:
            camera_cfg.setdefault("head_camera_type", self.cfg.head_camera_type)
            camera_cfg.setdefault("wrist_camera_type", self.cfg.wrist_camera_type)
        else:
            camera_cfg["head_camera_type"] = self.cfg.head_camera_type
            camera_cfg["wrist_camera_type"] = self.cfg.wrist_camera_type
        if "collect_wrist_camera" not in camera_cfg:
            camera_cfg["collect_wrist_camera"] = self.cfg.include_wrist_cameras
        trial_seed = self.env_seed
        retry_count = 0
        while True:
            try:
                with robotwin_workdir(self.cfg.robotwin_root):
                    self.task.setup_demo(now_ep_num=trial_seed, seed=trial_seed, **args)
                break
            except Exception as exc:
                if type(exc).__name__ != "UnStableError":
                    raise
                retry_count += 1
                LOGGER.warning(
                    "RoboTwin reset hit UnStableError; env_id=%s task=%s "
                    "seed=%s next_seed=%s error=%s",
                    self.env_id,
                    self.task_name,
                    trial_seed,
                    trial_seed + 1,
                    exc,
                )
                self.task.close_env(clear_cache=True)
                if retry_count % UNSTABLE_RETRY_WARNING_INTERVAL == 0:
                    LOGGER.warning(
                        "RoboTwin reset is still retrying after %s unstable seeds; "
                        "env_id=%s task=%s current_seed=%s",
                        retry_count,
                        self.env_id,
                        self.task_name,
                        trial_seed,
                    )
                trial_seed += 1
        self.env_seed = trial_seed
        self.task.step_lim = int(args["step_lim"])
        self.task.take_action_cnt = 0
        self.task.run_steps = 0
        self.task.reward_step = 0
        self.task.eval_success = False
        if self.cfg.profile_timing:
            from envs.utils.step_timer import StepTimer
            self.task._step_timer = StepTimer(
                enabled=True, log_interval=self.cfg.profile_interval
            )
        self._install_helpers()
        if self.cfg.reward_mode == "dense":
            build_task_reward(self.task_name, self.task)
        self.elapsed_steps = 0
        self.last_dense_reward = 0.0
        obs = self.get_obs()
        obs["_env_seed"] = self.env_seed
        self._start_eval_video_if_needed(obs.get("rgb"))
        return obs

    def step(self, action: np.ndarray) -> StepResult:
        if self.task is None:
            self.reset()
        assert self.task is not None
        action = self._to_robotwin_action(action)
        self._begin_reward_trace()
        self.task.take_action(action, action_type=self._robotwin_action_type())
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
        if success or bool(truncated):
            self._stop_eval_video_if_needed()

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
            self._stop_eval_video_if_needed()
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
        if self.cfg.control_mode == "ee_delta_pose":
            return self._to_robotwin_delta_ee_action(action)
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

    def _to_robotwin_delta_ee_action(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] != 14:
            raise ValueError(
                "ee_delta_pose expects 14 dims: "
                "left xyz+rotvec+gripper and right xyz+rotvec+gripper."
            )
        assert self.task is not None
        robot = self.task.robot
        left_gripper = _get_gripper_val(robot, "left")
        right_gripper = _get_gripper_val(robot, "right")
        out = np.empty(16, dtype=np.float32)
        out[0:3] = action[0:3] * self.cfg.ee_delta_pos_scale
        out[3:7] = _rotvec_to_wxyz(action[3:6] * self.cfg.ee_delta_rot_scale)
        out[7] = np.clip(
            left_gripper + action[6] * self.cfg.gripper_delta_scale,
            0.0,
            1.0,
        )
        out[8:11] = action[7:10] * self.cfg.ee_delta_pos_scale
        out[11:15] = _rotvec_to_wxyz(action[10:13] * self.cfg.ee_delta_rot_scale)
        out[15] = np.clip(
            right_gripper + action[13] * self.cfg.gripper_delta_scale,
            0.0,
            1.0,
        )
        return out

    def _robotwin_action_type(self) -> str:
        if self.cfg.control_mode == "ee_delta_pose":
            return "ee"
        return "qpos"

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

    def _start_eval_video_if_needed(self, rgb: Any) -> None:
        assert self.task is not None
        video_dir = getattr(self.task, "eval_video_path", None)
        if video_dir is None:
            return

        rgb_array = np.asarray(rgb)
        if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
            raise ValueError(
                "RoboTwin eval video expects an RGB head-camera image with shape HxWx3."
            )
        os.makedirs(os.fspath(video_dir), exist_ok=True)
        h, w = rgb_array.shape[:2]
        out_path = os.path.join(
            os.fspath(video_dir),
            f"episode_env{self.env_id}_seed{self.env_seed}_{self._eval_video_index}.mp4",
        )
        self._eval_video_index += 1
        ffmpeg = subprocess.Popen(
            [
                _ffmpeg_executable(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{w}x{h}",
                "-framerate",
                "10",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )
        self.task._set_eval_video_ffmpeg(ffmpeg)

    def _stop_eval_video_if_needed(self) -> None:
        assert self.task is not None
        if hasattr(self.task, "eval_video_ffmpeg"):
            self.task._del_eval_video_ffmpeg()

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


def _get_gripper_val(robot, arm: str) -> float:
    getter = getattr(robot, f"get_{arm}_gripper_val", None)
    if callable(getter):
        return float(getter())
    joint_state = getattr(robot, f"get_{arm}_arm_jointState")()
    return float(joint_state[-1])


def _rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = rotvec / angle
    half_angle = 0.5 * angle
    quat = np.empty(4, dtype=np.float32)
    quat[0] = np.cos(half_angle)
    quat[1:] = axis * np.sin(half_angle)
    return quat


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


def _prepare_robotwin_task_args(args: dict[str, Any]) -> None:
    args.setdefault(
        "domain_randomization",
        {
            "random_background": True,
            "cluttered_table": True,
            "clean_background_rate": 0.02,
            "random_head_camera_dis": 0,
            "random_table_height": 0.03,
            "random_light": False,
            "crazy_random_light_rate": 0.0,
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
        if isinstance(args["embodiment_dis"], str):
            args["embodiment_dis"] = float(args["embodiment_dis"])
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
