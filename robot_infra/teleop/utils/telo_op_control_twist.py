from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


RIGHT_ALIGN = (
    np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
)
LEFT_ALIGN = (
    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    @ RIGHT_ALIGN
)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _se3_log(omega: np.ndarray, pos: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    omega_hat = _skew(omega)
    if theta < 1e-8:
        v_inv = np.eye(3) - 0.5 * omega_hat + (omega_hat @ omega_hat) / 12.0
    else:
        a = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        v_inv = np.eye(3) - 0.5 * omega_hat + a * (omega_hat @ omega_hat)
    return np.concatenate([v_inv @ pos, omega]).astype(np.float32)


def _quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("Invalid hand quaternion.")
    return quat / norm


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[:3]
    w = q[3]
    return (
        2.0 * np.dot(xyz, v) * xyz
        + (w * w - np.dot(xyz, xyz)) * v
        + 2.0 * w * np.cross(xyz, v)
    )


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = _quat(q)
    if q[3] < 0.0:
        q = -q
    vec = q[:3]
    sin_half = np.linalg.norm(vec)
    if sin_half < 1e-8:
        return 2.0 * vec
    angle = 2.0 * np.arctan2(sin_half, q[3])
    return vec * (angle / sin_half)


class ButtonStatus(Enum):
    RELEASED = 0
    JUST_PRESSED = 1
    PRESSED = 2
    JUST_RELEASED = 3


@dataclass
class HandPoseToEETwist:
    hand: str = "right"
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    twist_limit: float = 0.1

    def __post_init__(self):
        if self.hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'.")
        self.align = LEFT_ALIGN if self.hand == "left" else RIGHT_ALIGN
        self.bound = False
        self._last_pos: np.ndarray | None = None
        self._last_quat: np.ndarray | None = None

    def bind(self, hand_pos: np.ndarray, hand_quat: np.ndarray) -> None:
        self._last_pos = np.asarray(hand_pos, dtype=np.float64).reshape(3).copy()
        self._last_quat = _quat(hand_quat)
        self.bound = True

    def compute(self, hand_pos: np.ndarray, hand_quat: np.ndarray):
        pos = np.asarray(hand_pos, dtype=np.float64).reshape(3)
        quat = _quat(hand_quat)
        if not self.bound or self._last_pos is None or self._last_quat is None:
            return np.zeros(6, dtype=np.float32)

        last_inv = _quat_conj(self._last_quat)
        delta_pos_hand = _quat_rotate(last_inv, pos - self._last_pos)
        delta_rotvec_hand = _quat_to_rotvec(_quat_mul(last_inv, quat))
        self._last_pos = pos.copy()
        self._last_quat = quat

        delta_pos_ee = self.align @ delta_pos_hand
        delta_rotvec_ee = self.align @ delta_rotvec_hand
        twist = _se3_log(delta_rotvec_ee, delta_pos_ee)
        twist[:3] *= self.pos_scale
        twist[3:] *= self.rot_scale
        return np.clip(twist, -self.twist_limit, self.twist_limit).astype(np.float32)


class EETwistTeleOpWrapper:
    def __init__(
        self,
        zmq_url: str = "tcp://localhost:7777",
        hand: str = "right",
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        twist_limit: float = 0.1,
    ):
        import zmq

        self.zmq = zmq
        self.hand = hand
        self.mapper = HandPoseToEETwist(hand, pos_scale, rot_scale, twist_limit)
        self.button_state = ButtonStatus.PRESSED
        self.last_gripper = 1.0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(zmq_url)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    @staticmethod
    def _gripper_action(cmd: float) -> float:
        return -1.0 if cmd == 1 else 1.0

    @staticmethod
    def _update_button_state(data: np.ndarray, button_state: ButtonStatus):
        if data[11] == 1 and button_state in [ButtonStatus.RELEASED]:
            return ButtonStatus.JUST_PRESSED
        if data[11] == 1 and button_state in [ButtonStatus.JUST_PRESSED, ButtonStatus.PRESSED]:
            return ButtonStatus.PRESSED
        if data[11] == 0 and button_state in [ButtonStatus.PRESSED]:
            return ButtonStatus.JUST_RELEASED
        if data[11] == 0 and button_state in [ButtonStatus.JUST_RELEASED, ButtonStatus.RELEASED]:
            return ButtonStatus.RELEASED
        return button_state

    def _update_mapper_state(self, hand_data: np.ndarray) -> None:
        if self.button_state in [ButtonStatus.JUST_PRESSED, ButtonStatus.PRESSED]:
            self.mapper.bind(hand_data[:3], hand_data[6:10])

    def _recv_latest(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                latest = self.socket.recv_json(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
        return None if latest is None else np.asarray(latest, dtype=np.float32)

    def _select_hand_data(self, data: np.ndarray | None) -> np.ndarray | None:
        if data is None:
            return None

        if data.size == 13:
            return data
        if data.size >= 26:
            return data[:13] if self.hand == "left" else data[13:26]
        raise ValueError(f"Expected 13 or 26 teleop values, got {data.size}.")

    def poll(self) -> dict[str, np.ndarray | float | ButtonStatus] | None:
        data = self._recv_latest()
        if data is None:
            return None

        hand_data = self._select_hand_data(data)
        self.button_state = self._update_button_state(hand_data, self.button_state)
        self._update_mapper_state(hand_data)
        twist = self.mapper.compute(hand_data[:3], hand_data[6:10])
        if hand_data[11] == 1:
            twist = np.zeros(6, dtype=np.float32)
        self.last_gripper = self._gripper_action(hand_data[10])
        return {
            "raw": data,
            "hand": hand_data,
            "twist": twist,
            "gripper": self.last_gripper,
            "button_state": self.button_state,
        }

    def get_ee_twist(self) -> np.ndarray:
        sample = self.poll()
        if sample is None:
            return np.zeros(6, dtype=np.float32)
        return sample["twist"]

    def get_action(self) -> np.ndarray:
        sample = self.poll()
        if sample is None:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.last_gripper],
                dtype=np.float32,
            )
        return np.concatenate([sample["twist"], [sample["gripper"]]]).astype(np.float32)

    def close(self) -> None:
        self.socket.close()
        self.context.term()
