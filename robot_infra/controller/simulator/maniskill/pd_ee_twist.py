from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, DriveMode

from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController


class PDEETwistController(PDJointPosController):
    """PD EE Twist controller that accepts 6D twist [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    in the **end-effector body frame** and converts to joint position targets.

    The twist is mapped to a desired EE pose via the SE(3) exponential map (exact, not
    a linear approximation). The desired pose is then resolved to joint displacements:
      - GPU: pose error computed in world frame (p_servo equivalent), then rotated to
             arm base frame via R6 adjoint, and resolved via pinv(J_base).
      - CPU: desired pose passed to pinocchio iterative IK solver.

    Low-level execution is delegated to pd_joint_pos (PD position tracking on joints).
    """

    config: "PDEETwistControllerConfig"

    # ------------------------------------------------------------------
    # Static helpers for SE(3) / so(3) operations
    # ------------------------------------------------------------------

    @staticmethod
    def _twist_hat(xi: torch.Tensor) -> torch.Tensor:
        """Batched 6D twist -> 4x4 se(3) Lie algebra element.

        xi: (B, 6) with layout [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        Returns: (B, 4, 4)  [ [omega]_x  v ]
                             [    0       0 ]
        """
        B = xi.shape[0]
        v = xi[:, :3]
        omega = xi[:, 3:]
        mat = torch.zeros(B, 4, 4, device=xi.device, dtype=xi.dtype)
        # skew-symmetric [omega]_x
        mat[:, 0, 1] = -omega[:, 2]
        mat[:, 0, 2] = omega[:, 1]
        mat[:, 1, 0] = omega[:, 2]
        mat[:, 1, 2] = -omega[:, 0]
        mat[:, 2, 0] = -omega[:, 1]
        mat[:, 2, 1] = omega[:, 0]
        # translation
        mat[:, :3, 3] = v
        return mat

    @staticmethod
    def _twist_to_SE3(xi: torch.Tensor) -> torch.Tensor:
        """Batched 6D twist -> 4x4 SE(3) via matrix exponential (exact).

        Equivalent to scipy.linalg.expm(twist_hat(xi)).
        """
        return torch.matrix_exp(PDEETwistController._twist_hat(xi))

    @staticmethod
    def _pose_to_matrix(pose: Pose) -> torch.Tensor:
        """Convert ManiSkill Pose (p, q) to batched 4x4 homogeneous matrix."""
        R = quaternion_to_matrix(pose.q)  # (B, 3, 3)
        B = R.shape[0]
        T = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
        T[:, :3, :3] = R
        T[:, :3, 3] = pose.p
        T[:, 3, 3] = 1.0
        return T

    @staticmethod
    def _rotation_matrix_to_angle_axis(R: torch.Tensor) -> torch.Tensor:
        """Batched rotation matrix -> angle-axis (logarithmic map).

        R: (B, 3, 3)  Returns: (B, 3) where ||result|| = angle, direction = axis.
        """
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = torch.acos(cos_angle)  # (B,)

        # anti-symmetric part = 2 * sin(angle) * axis
        skew = torch.stack([
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ], dim=-1)  # (B, 3)

        sin_angle = torch.sin(angle)
        small = sin_angle.abs() < 1e-8
        safe_sin = torch.where(small, torch.ones_like(sin_angle), sin_angle)
        # angle / (2*sin(angle)) -> 0.5 when angle -> 0  (L'Hopital)
        scale = torch.where(
            small,
            0.5 * torch.ones_like(angle),
            angle / (2.0 * safe_sin),
        )
        return skew * scale.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_joints(self):
        super()._initialize_joints()
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )
        self.ee_link = self.kinematics.end_link
        if self.config.root_link_name is not None:
            self.root_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.root_link_name
            )
        else:
            self.root_link = self.articulation.root

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.twist_lower, 6))
        high = np.float32(np.broadcast_to(self.config.twist_upper, 6))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.root_link.pose.inv()
        return to_base * self.ee_pose

    # ------------------------------------------------------------------
    # Core control
    # ------------------------------------------------------------------

    def _compute_desired_T_at_base(self, twist: torch.Tensor) -> torch.Tensor:
        """Compute desired 4x4 EE transform in base frame from twist in EE body frame.

        desired_T = current_T @ expm(twist_hat)   (right-multiply = body-frame increment)
        """
        delta_T = self._twist_to_SE3(twist)                     # (B, 4, 4)
        current_T = self._pose_to_matrix(self.ee_pose_at_base)  # (B, 4, 4)
        return torch.bmm(current_T, delta_T)

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        if self.kinematics.use_gpu_ik:
            self._target_qpos = self._compute_target_qpos_gpu(action)
        else:
            self._target_qpos = self._compute_target_qpos_cpu(action)

        if self._target_qpos is None:
            self._target_qpos = self._start_qpos

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def _compute_target_qpos_gpu(self, twist: torch.Tensor) -> torch.Tensor:
        """GPU path — mirrors twist_in_EE_frame_to_action:
        1. twist (EE frame) -> delta_T via SE(3) exp
        2. desired_T_world = current_T_world @ delta_T   (right-multiply = body frame)
        3. pose error (pos + angle-axis rot) in world frame   (p_servo, Gain=1)
        4. R6 adjoint: world frame -> arm base frame
        5. delta_q = pinv(J_base) @ error_base
        6. target_q = current_q + delta_q
        """
        delta_T = self._twist_to_SE3(twist)  # (B, 4, 4)

        # current & desired EE poses in world frame
        current_T_world = self._pose_to_matrix(self.ee_pose)       # (B, 4, 4)
        desired_T_world = torch.bmm(current_T_world, delta_T)      # (B, 4, 4)

        # pose error in world frame (equivalent to p_servo with Gain=1, method='angle-axis')
        pos_error_world = (
            desired_T_world[:, :3, 3] - current_T_world[:, :3, 3]
        )  # (B, 3)
        R_error = torch.bmm(
            desired_T_world[:, :3, :3],
            current_T_world[:, :3, :3].transpose(1, 2),
        )  # (B, 3, 3)
        rot_error_world = self._rotation_matrix_to_angle_axis(R_error)  # (B, 3)

        # transform 6D error from world frame to arm base frame
        # R_base_in_world is the rotation part of the arm base pose in world
        R_base_in_world = quaternion_to_matrix(self.root_link.pose.q)   # (B, 3, 3)
        R_world_to_base = R_base_in_world.transpose(1, 2)              # (B, 3, 3)
        pos_error_base = torch.bmm(
            R_world_to_base, pos_error_world.unsqueeze(-1)
        ).squeeze(-1)  # (B, 3)
        rot_error_base = torch.bmm(
            R_world_to_base, rot_error_world.unsqueeze(-1)
        ).squeeze(-1)  # (B, 3)
        error_base = torch.cat([pos_error_base, rot_error_base], dim=-1)  # (B, 6)

        # Jacobian (spatial, in base frame) for controlled joints
        q0 = self.articulation.get_qpos()
        q0_ancestors = q0[:, self.kinematics.active_ancestor_joint_idxs]
        jacobian = self.kinematics.pk_chain.jacobian(q0_ancestors)[
            :, :, self.kinematics.qmask
        ]  # (B, 6, n_controlled)

        # pseudo-inverse: delta_q = pinv(J) @ error_base
        delta_q = torch.bmm(
            torch.linalg.pinv(jacobian), error_base.unsqueeze(-1)
        ).squeeze(-1)  # (B, n_controlled)

        return self.qpos + delta_q

    def _compute_target_qpos_cpu(
        self, twist: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """CPU path:
        1. twist (EE frame) -> desired pose via SE(3) exp
        2. desired pose -> pinocchio iterative IK
        """
        desired_T = self._compute_desired_T_at_base(twist)
        target_pos = desired_T[:, :3, 3]
        target_quat = matrix_to_quaternion(desired_T[:, :3, :3])
        target_pose = Pose.create_from_pq(target_pos, target_quat)

        return self.kinematics.compute_ik(
            pose=target_pose,
            q0=self.articulation.get_qpos(),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dof={self.single_action_space.shape[0]}, "
            f"active_joints={len(self.joints)}, "
            f"end_link={self.config.ee_link}, "
            f"joints=({', '.join([x.name for x in self.joints])}))"
        )


@dataclass
class PDEETwistControllerConfig(ControllerConfig):
    """Configuration for the PD EE Twist controller.

    Accepts 6D twist actions [v_x, v_y, v_z, omega_x, omega_y, omega_z] in the
    **end-effector body frame**. The twist is converted to a desired pose via SE(3)
    exponential map and then resolved to joint position targets.
    """

    twist_lower: Union[float, Sequence[float]] = -0.1
    """Lower bound for twist action. Scalar is broadcast to all 6 dimensions."""
    twist_upper: Union[float, Sequence[float]] = 0.1
    """Upper bound for twist action. Scalar is broadcast to all 6 dimensions."""

    stiffness: Union[float, Sequence[float]] = 1e3
    damping: Union[float, Sequence[float]] = 1e2
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    ee_link: str = None
    """The end-effector link name for Jacobian computation."""
    urdf_path: str = None
    """Path to the URDF file for kinematics."""
    root_link_name: Optional[str] = None
    """Optionally set a different root link for the base frame."""

    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDEETwistController
