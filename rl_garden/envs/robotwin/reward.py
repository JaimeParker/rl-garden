"""Dense reward primitives for RoboTwin tasks.

Portions are adapted from RoboTwin's ``RLinf_support`` branch
(``envs/reward.py``). The code is kept local so rl-garden can maintain reward
definitions independently while using an installed RoboTwin runtime.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable, Optional

import numpy as np


def process_pose(pose: Iterable, dim: Optional[int | Iterable] = None) -> np.ndarray:
    pose = np.array(pose)
    if dim is None:
        return pose
    if isinstance(dim, int):
        return pose[:dim]
    mask = np.array(dim)
    if len(mask) < len(pose):
        mask = np.concatenate([mask, np.zeros(len(pose) - len(mask))])
    return pose[mask.astype(bool)]


class BaseTask(abc.ABC):
    @abc.abstractmethod
    def compute_reward(self) -> float: ...

    @abc.abstractmethod
    def is_success(self) -> bool: ...

    @abc.abstractmethod
    def is_fail(self) -> bool: ...

    @abc.abstractmethod
    def update(self) -> None: ...


class SubTask(BaseTask):
    def __init__(self, base=None, max_reward: float = 1.0, **_: Any) -> None:
        self.base = base
        self.max_reward = max_reward

    def compute_reward(self) -> float:
        return 0.0

    def is_success(self) -> bool:
        return False

    def is_fail(self) -> bool:
        return False

    def update(self) -> None:
        return None


class SerialTask(BaseTask):
    def __init__(self, subtasks: list[BaseTask], transition_rewards: list[float]) -> None:
        if len(transition_rewards) != len(subtasks) - 1:
            raise ValueError("transition_rewards must be len(subtasks) - 1")
        self.subtasks = subtasks
        self.transition_rewards = transition_rewards
        self.current_idx = 0

    def _total_max_reward(self) -> float:
        return sum(
            t.max_reward if isinstance(t, SubTask) else t._total_max_reward()
            for t in self.subtasks
        ) + sum(self.transition_rewards)

    def _preceding_max_reward(self) -> float:
        return sum(
            t.max_reward if isinstance(t, SubTask) else t._total_max_reward()
            for t in self.subtasks[: self.current_idx]
        ) + sum(self.transition_rewards[: self.current_idx])

    def compute_reward(self, normalized: bool = False) -> float:
        if self.current_idx >= len(self.subtasks):
            return 0.0
        reward = self.subtasks[self.current_idx].compute_reward()
        if reward < 0:
            return 0.0
        raw = self._preceding_max_reward() + reward
        total = self._total_max_reward()
        return raw / total if normalized and total > 0 else raw

    def update(self) -> None:
        current = self.subtasks[self.current_idx]
        current.update()
        if (
            current.is_success()
            and self.current_idx < len(self.subtasks) - 1
            and not isinstance(current, Success)
        ):
            self.current_idx += 1

    def is_success(self) -> bool:
        return self.current_idx == len(self.subtasks) - 1 and self.subtasks[-1].is_success()

    def is_fail(self) -> bool:
        return self.subtasks[self.current_idx].is_fail()


class ParallelTask(BaseTask):
    def __init__(self, subtasks: list[BaseTask], weights: list[float]) -> None:
        if len(subtasks) != len(weights):
            raise ValueError("weights must match subtasks length")
        self.subtasks = subtasks
        self.weights = weights
        self.success = [False for _ in subtasks]

    def _total_max_reward(self) -> float:
        total_weight = sum(self.weights)
        return sum(
            w * (t.max_reward if isinstance(t, SubTask) else t._total_max_reward())
            for w, t in zip(self.weights, self.subtasks)
        ) / total_weight

    def compute_reward(self) -> float:
        total_weight = sum(self.weights)
        return sum(w * t.compute_reward() for w, t in zip(self.weights, self.subtasks)) / total_weight

    def update(self) -> None:
        for task in self.subtasks:
            task.update()

    def is_success(self) -> bool:
        for idx, task in enumerate(self.subtasks):
            if task.is_success():
                self.success[idx] = True
        return all(self.success)

    def is_fail(self) -> bool:
        return all(task.is_fail() for task in self.subtasks)


class Reward:
    @staticmethod
    def build(config: dict[str, Any] | BaseTask) -> BaseTask:
        if isinstance(config, BaseTask):
            return config
        if not isinstance(config, dict):
            raise ValueError(f"Invalid reward config type: {type(config)}")
        task_type = config.get("type", "Serial")
        if task_type == "Serial":
            subtasks = [Reward.build(s) for s in config["subtasks"]]
            return SerialTask(
                subtasks=subtasks,
                transition_rewards=config.get("transition_rewards", [0.0] * (len(subtasks) - 1)),
            )
        if task_type == "Parallel":
            subtasks = [Reward.build(s) for s in config["subtasks"]]
            return ParallelTask(subtasks=subtasks, weights=config.get("weights", [1.0] * len(subtasks)))
        if task_type == "Success":
            return Success()
        raise ValueError(f"Unknown reward task type: {task_type!r}")

    @staticmethod
    def build_top(config: dict[str, Any]) -> BaseTask:
        task = Reward.build(config)

        class TopLevelTaskWrapper(BaseTask):
            def compute_reward(self, normalize: bool = True) -> float:
                if isinstance(task, SerialTask):
                    return task.compute_reward(normalized=normalize)
                return task.compute_reward()

            def update(self) -> None:
                task.update()

            def is_success(self) -> bool:
                return task.is_success()

            def is_fail(self) -> bool:
                return task.is_fail()

        return TopLevelTaskWrapper()


class Success(SubTask):
    def __init__(self, base=None, max_reward: float = 0.0, **task_params: Any) -> None:
        super().__init__(base, max_reward, **task_params)

    def is_success(self) -> bool:
        return True


class Pick(SubTask):
    def __init__(
        self,
        base,
        max_reward: float = 4.0,
        entity: Optional[Any] = None,
        dist: float = 0.18,
        eef_dim: Optional[int | Iterable] = 3,
        joint_dim: Optional[int | Iterable] = None,
        a_d: float = 3.0,
        a_g: float = 1.0,
        c_d: float = 2.0,
        c_g: float = 2.0,
        thresh_eef: Optional[float] = 0.02,
        thresh_joint: Optional[float] = 0.1,
        arm_tag: Optional[str | int] = None,
    ) -> None:
        super().__init__(base, max_reward, entity=entity)
        self.entity = entity
        self.gripper_dist = dist
        self.eef_dim = eef_dim
        self.joint_dim = joint_dim
        self.a_d = a_d
        self.a_g = a_g
        if abs(c_d + c_g - max_reward) >= 1e-5:
            raise ValueError("c_d + c_g must equal max_reward")
        self.c_d = c_d
        self.c_g = c_g
        self.thresh_eef = thresh_eef
        self.thresh_joint = thresh_joint
        self.arm_tag = _arm_index(arm_tag)

    def compute_reward(self, action_punishment: bool = False) -> float:
        base = self.base
        start_left = process_pose(base.episode_left_eef_poses[0], self.eef_dim)
        start_right = process_pose(base.episode_right_eef_poses[0], self.eef_dim)
        end_left = process_pose(base.episode_left_eef_poses[-1], self.eef_dim)
        end_right = process_pose(base.episode_right_eef_poses[-1], self.eef_dim)
        entity_pose = process_pose(self.entity.get_pose().p, self.eef_dim)
        dists = [np.linalg.norm(end_left - entity_pose), np.linalg.norm(end_right - entity_pose)]
        gripper = int(np.argmin(dists)) if self.arm_tag is None else self.arm_tag
        gripper_angle = base.robot.left_gripper_val if gripper == 0 else base.robot.right_gripper_val
        if action_punishment:
            start = start_left if gripper == 0 else start_right
            end = end_left if gripper == 0 else end_right
            if np.linalg.norm(end - start) < float(self.thresh_eef or 0.0):
                return -1.0
        dist_reward = 1 - np.tanh(dists[gripper] * self.a_d)
        gripper_reward = 1 - np.tanh(gripper_angle * self.a_g)
        return float(dist_reward * self.c_d + gripper_reward * (dists[gripper] < self.gripper_dist) * self.c_g)

    def is_success(self) -> bool:
        grabs = self.base.is_in_hand(self.entity)
        return any(grabs) if self.arm_tag is None else bool(grabs[self.arm_tag])


class Contact(Pick):
    def __init__(self, *args, entity_name: Optional[str] = None, entity_idx: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if entity_name is None or entity_idx is None:
            raise ValueError("entity_name and entity_idx are required")
        self.entity_name = entity_name
        self.entity_idx = entity_idx

    def compute_reward(self, action_punishment: bool = False) -> float:
        entity_pose = process_pose(self.entity.get_contact_point(self.entity_idx)[:3], self.eef_dim)
        end_left = process_pose(self.base.episode_left_eef_poses[-1], self.eef_dim)
        end_right = process_pose(self.base.episode_right_eef_poses[-1], self.eef_dim)
        dists = [np.linalg.norm(end_left - entity_pose), np.linalg.norm(end_right - entity_pose)]
        gripper = int(np.argmin(dists)) if self.arm_tag is None else self.arm_tag
        gripper_angle = self.base.robot.left_gripper_val if gripper == 0 else self.base.robot.right_gripper_val
        return float((1 - np.tanh(dists[gripper] * self.a_d)) * self.c_d + (1 - np.tanh(gripper_angle * self.a_g)) * self.c_g)

    def is_success(self) -> bool:
        return bool(self.base.get_gripper_actor_contact_position(self.entity_name))


class Place(SubTask):
    def __init__(
        self,
        base,
        max_reward: float = 4.0,
        entity: Optional[Any] = None,
        target: Optional[Any] = None,
        dist: float = 0.15,
        eef_dim: Optional[int | Iterable] = 3,
        joint_dim: Optional[int | Iterable] = None,
        a_d: float = 3.0,
        a_g: float = 1.5,
        c_d: float = 2.0,
        c_g: float = 2.0,
        thresh_eef: Optional[float] = 0.02,
        thresh_joint: Optional[float] = 0.1,
        arm_tag: Optional[str | int] = None,
        eps: Optional[Any] = None,
        eps_mask: Optional[Any] = None,
        is_function_point=None,
        name: str = "base",
    ) -> None:
        super().__init__(base, max_reward, entity=entity)
        self.entity = entity
        self.target = target
        self.dist = dist
        self.eef_dim = eef_dim
        self.joint_dim = joint_dim
        self.a_d = a_d
        self.a_g = a_g
        if abs(c_d + c_g - max_reward) >= 1e-5:
            raise ValueError("c_d + c_g must equal max_reward")
        self.c_d = c_d
        self.c_g = c_g
        self.thresh_eef = thresh_eef
        self.thresh_joint = thresh_joint
        self.arm_tag = _arm_index(arm_tag)
        self.eps = eps
        self.eps_mask = eps_mask
        self.is_function_point = is_function_point
        self.name = name

    def _target_pose(self, dim) -> np.ndarray:
        if isinstance(self.target, tuple):
            target = self.target[0].get_functional_point(self.target[1])
            return process_pose(target, dim)
        if isinstance(self.target, list):
            return process_pose(self.target, dim)
        if isinstance(self.target, np.ndarray):
            return process_pose(self.target, dim)
        raise ValueError("target must be a tuple/list/ndarray")

    def compute_reward(self, action_punishment: bool = False) -> float:
        entity_pose = process_pose(self.entity.get_pose().p, self.eef_dim)
        target_pose = self._target_pose(self.eef_dim)
        end_left = process_pose(self.base.episode_left_eef_poses[-1], self.eef_dim)
        end_right = process_pose(self.base.episode_right_eef_poses[-1], self.eef_dim)
        dists = [np.linalg.norm(end_left - entity_pose), np.linalg.norm(end_right - entity_pose)]
        gripper = int(np.argmin(dists)) if self.arm_tag is None else self.arm_tag
        gripper_angle = self.base.robot.left_gripper_val if gripper == 0 else self.base.robot.right_gripper_val
        entity_to_target = np.linalg.norm(target_pose - entity_pose)
        dist_reward = 1 - np.tanh(entity_to_target * self.a_d)
        gripper_reward = 1 - np.tanh((1 - gripper_angle) * self.a_g)
        return float(dist_reward * self.c_d + gripper_reward * (entity_to_target < self.dist) * self.c_g)

    def is_success(self) -> bool:
        if self.eps is None:
            return bool(self.base.check_success())
        entity_pose = process_pose(np.concatenate([self.entity.get_pose().p, self.entity.get_pose().q]), self.eps_mask)
        return bool(np.all(np.abs(entity_pose - self._target_pose(self.eps_mask)) < self.eps))

    def is_fail(self) -> bool:
        left_grab, right_grab = self.base.is_in_hand(self.entity)
        return (
            not self.is_success()
            and self.base.robot.is_left_gripper_open()
            and self.base.robot.is_right_gripper_open()
            and not self.base.check_success()
            and not left_grab
            and not right_grab
        )


class Endpose(SubTask):
    def __init__(self, base, max_reward: float = 1.0, left_target=None, right_target=None) -> None:
        super().__init__(base, max_reward, left_target=left_target, right_target=right_target)
        self.left_target = left_target
        self.right_target = right_target

    def compute_reward(self, action_punishment: bool = True) -> float:
        left_dist = np.linalg.norm(self.base.episode_left_eef_poses[-1][:3] - np.array(self.left_target[:3]))
        right_dist = np.linalg.norm(self.base.episode_right_eef_poses[-1][:3] - np.array(self.right_target[:3]))
        return float(((1 - np.tanh(left_dist * 5)) + (1 - np.tanh(right_dist * 5))) / 2)

    def is_success(self) -> bool:
        return bool(self.base.check_success())


class OpenArticulation(SubTask):
    """Shape reaching an articulation handle and opening its target joint."""

    def __init__(
        self,
        base,
        max_reward: float = 4.0,
        entity=None,
        joint_idx: int = 0,
        contact_point_idx: int = 1,
        target_fraction: float = 0.4,
        arm_tag: Optional[str | int] = None,
        a_d: float = 5.0,
        c_d: float = 1.0,
        c_qpos: float = 3.0,
    ) -> None:
        super().__init__(base, max_reward, entity=entity)
        if abs(c_d + c_qpos - max_reward) >= 1e-5:
            raise ValueError("c_d + c_qpos must equal max_reward")
        if not 0.0 <= target_fraction <= 1.0:
            raise ValueError("target_fraction must be in [0, 1]")

        self.entity = entity
        self.joint_idx = joint_idx
        self.contact_point_idx = contact_point_idx
        self.arm_tag = _arm_index(arm_tag)
        self.a_d = a_d
        self.c_d = c_d
        self.c_qpos = c_qpos

        limits = np.asarray(entity.get_qlimits(), dtype=np.float64)[joint_idx]
        lower, upper = float(limits[0]), float(limits[1])
        if upper <= lower:
            raise ValueError("articulation joint upper limit must exceed lower limit")
        self.initial_qpos = float(np.asarray(entity.get_qpos())[joint_idx])
        self.target_qpos = lower + (upper - lower) * target_fraction

    def compute_reward(self) -> float:
        contact_pose = np.asarray(
            self.entity.get_contact_point(self.contact_point_idx), dtype=np.float64
        )
        tcp_poses = [
            np.asarray(self.base.robot.get_left_tcp_pose(), dtype=np.float64),
            np.asarray(self.base.robot.get_right_tcp_pose(), dtype=np.float64),
        ]
        dists = [np.linalg.norm(tcp[:3] - contact_pose[:3]) for tcp in tcp_poses]
        arm_idx = int(np.argmin(dists)) if self.arm_tag is None else self.arm_tag
        reach_reward = 1.0 - np.tanh(dists[arm_idx] * self.a_d)

        qpos = float(np.asarray(self.entity.get_qpos())[self.joint_idx])
        if self.initial_qpos >= self.target_qpos:
            progress = float(qpos >= self.target_qpos)
        else:
            progress = float(
                np.clip(
                    (qpos - self.initial_qpos)
                    / (self.target_qpos - self.initial_qpos),
                    0.0,
                    1.0,
                )
            )
        return float(reach_reward * self.c_d + progress * self.c_qpos)

    def is_success(self) -> bool:
        return bool(self.base.check_success())


class Rank(SubTask):
    def __init__(self, base, max_reward: float = 4.0, dist_dim: int | Iterable = 2, entities=None, eps=None, a_ds=None, c_ds=None) -> None:
        super().__init__(base, max_reward, entities=entities)
        self.dist_dim = dist_dim
        self.entities = entities if entities is not None else []
        self.eps = eps if eps is not None else [0.05]
        self.a_ds = a_ds if a_ds is not None else [3.0] * (len(self.entities) - 1)
        self.c_ds = c_ds if c_ds is not None else [max_reward / (len(self.entities) - 1)] * (len(self.entities) - 1)

    def compute_reward(self) -> float:
        poses = [np.array(e.get_pose().p) for e in self.entities]
        rewards = []
        for i in range(len(poses) - 1):
            dist = np.linalg.norm(poses[i][: self.dist_dim] - poses[i + 1][: self.dist_dim])
            correct = poses[i][0] < poses[i + 1][0]
            rewards.append((1 - np.tanh(dist * self.a_ds[i])) * correct * self.c_ds[i])
        return float(sum(rewards))

    def is_success(self) -> bool:
        poses = [np.array(e.get_pose().p) for e in self.entities]
        eps = np.array(self.eps)
        ok = []
        for i in range(len(poses) - 1):
            ok.append(poses[i][0] < poses[i + 1][0] and np.all(np.abs(poses[i][: len(eps)] - poses[i + 1][: len(eps)]) < eps))
        return bool((all(ok) and self.base.is_left_gripper_open() and self.base.is_right_gripper_open()) or self.base.check_success())


class Stack(SubTask):
    """RLinf-style vertical-stack shaping with pairwise XY alignment."""

    def __init__(
        self,
        base,
        max_reward: float = 4.0,
        entities=None,
        eps=None,
        a_ds=None,
        c_ds=None,
        target_pose=None,
        z_threshold: float = 0.02,
    ) -> None:
        super().__init__(base, max_reward, entities=entities)
        self.entities = entities if entities is not None else []
        self.eps = eps if eps is not None else [0.05, 0.03]
        num_pairs = len(self.entities) - 1
        self.a_ds = a_ds if a_ds is not None else [3.0] * num_pairs
        self.c_ds = c_ds if c_ds is not None else [max_reward / num_pairs] * num_pairs
        self.target_pose = (
            target_pose if target_pose is not None else np.array([0.5, 0.2, 0.0])
        )
        self.z_threshold = z_threshold

    def compute_reward(self) -> float:
        poses = [np.asarray(entity.get_pose().p) for entity in self.entities]
        pair_rewards = []
        for i, (lower, upper) in enumerate(zip(poses, poses[1:])):
            dz = upper[2] - lower[2]
            z_reward = 1 - np.tanh(abs(dz - self.z_threshold) * self.a_ds[i])
            xy_dist = np.linalg.norm(upper[:2] - lower[:2])
            xy_alignment = np.exp(-np.square(xy_dist / self.eps[1]))
            pair_rewards.append(
                z_reward * (dz > self.z_threshold) * xy_alignment * self.c_ds[i]
            )

        target_dists = [
            np.linalg.norm(pose[:2] - self.target_pose[:2]) for pose in poses
        ]
        target_reward = 0.5 * np.exp(
            -np.mean(np.square(target_dists) / np.square(self.eps[0]))
        )
        return float(sum(pair_rewards) + target_reward)

    def is_success(self) -> bool:
        return bool(self.base.check_success())


class SparseExtra(SubTask):
    def __init__(self, base, max_reward: float = 4.0, entity=None, target_entitys=None, dist: float = 0.15, arm_tag=None) -> None:
        super().__init__(base, max_reward, entity=entity)
        self.entity = entity
        self.target_entitys = target_entitys or []

    def compute_reward(self) -> float:
        for contact in self.base.scene.get_contacts():
            names = {contact.bodies[0].entity.get_name(), contact.bodies[1].entity.get_name()}
            if self.entity.get_name() in names and any(e.get_name() in names for e in self.target_entitys):
                return self.max_reward
        return 0.0

    def is_success(self) -> bool:
        return self.compute_reward() != 0


def _arm_index(arm_tag: Optional[str | int]) -> Optional[int]:
    if arm_tag in ("left", 0):
        return 0
    if arm_tag in ("right", 1):
        return 1
    if arm_tag is None:
        return None
    raise ValueError("arm_tag must be 'left', 'right', 0, 1, or None")
