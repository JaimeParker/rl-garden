from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import Array

from robot_infra.controller.adapters import DeltaPoseCommandAdapter, TwistCommandAdapter
from robot_infra.controller.core import (
    ImpedanceConfig,
    ImpedanceControllerState,
    compute_impedance_torque,
)


class ImpedanceEEDeltaPoseController(BaseController):
    """ManiSkill wrapper for a torch Cartesian impedance torque controller."""

    config: "ImpedanceEEDeltaPoseControllerConfig"
    sets_target_qpos = False
    sets_target_qvel = False
    adapter_cls = DeltaPoseCommandAdapter

    def _initialize_joints(self):
        super()._initialize_joints()
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )
        self.ee_link = self.kinematics.end_link
        if self.config.root_link_name is None:
            self.root_link = self.articulation.root
        else:
            self.root_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.root_link_name
            )
        self.adapter = self.adapter_cls()
        self._impedance_state = ImpedanceControllerState()
        self._nullspace_qpos: Optional[torch.Tensor] = None

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.action_lower, 6))
        high = np.float32(np.broadcast_to(self.config.action_upper, 6))
        self.single_action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_drive_property(self):
        # Torque is applied through qf. Disable drive tracking on controlled joints.
        n = len(self.joints)
        friction = np.broadcast_to(self.config.friction, n)
        for i, joint in enumerate(self.joints):
            joint.set_drive_properties(0.0, 0.0, force_limit=0.0, mode="force")
            joint.set_friction(float(friction[i]))

    @property
    def qpos(self):
        return self.articulation.get_qpos()[..., self.active_joint_indices]

    @property
    def qvel(self):
        return self.articulation.get_qvel()[..., self.active_joint_indices]

    @property
    def ee_pose_at_root(self):
        return self.root_link.pose.inv() * self.ee_link.pose

    def reset(self):
        super().reset()
        ee_pose = self.ee_pose_at_root
        self.adapter.reset(ee_pose.p, ee_pose.q)
        self._nullspace_qpos = self.qpos.clone()
        self._impedance_state = ImpedanceControllerState(
            previous_torque=torch.zeros_like(self.qpos)
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self.adapter.set_action(action)

    def before_simulation_step(self):
        tau = self._compute_torque()
        self._apply_qf(tau)

    def _compute_torque(self) -> torch.Tensor:
        ee_pose = self.ee_pose_at_root
        command = self.adapter.command(nullspace_qpos=self._nullspace_qpos)
        config = ImpedanceConfig(
            cartesian_stiffness=self.config.cartesian_stiffness,
            cartesian_damping=self.config.cartesian_damping,
            nullspace_stiffness=self.config.nullspace_stiffness,
            nullspace_damping=self.config.nullspace_damping,
            pose_error_clip=self.config.pose_error_clip,
            torque_limit=self.config.torque_limit,
            torque_rate_limit=self.config.torque_rate_limit,
        )
        tau, self._impedance_state = compute_impedance_torque(
            qpos=self.qpos,
            qvel=self.qvel,
            ee_pos=ee_pose.p,
            ee_quat=ee_pose.q,
            jacobian=self._compute_jacobian(),
            command=command,
            config=config,
            state=self._impedance_state,
        )
        return tau

    def _compute_jacobian(self) -> torch.Tensor:
        if self.kinematics.use_gpu_ik:
            qpos = self.articulation.get_qpos()
            qpos_ancestors = qpos[:, self.kinematics.active_ancestor_joint_idxs]
            return self.kinematics.pk_chain.jacobian(qpos_ancestors)[
                :, :, self.kinematics.qmask
            ]
        return self._compute_cpu_jacobian()

    def _compute_cpu_jacobian(self) -> torch.Tensor:
        qpos = self.articulation.get_qpos()
        qpos_model = qpos[:, self.kinematics.pmodel_active_joint_indices]
        jacobians = []
        for row in qpos_model:
            self.kinematics.pmodel.compute_full_jacobian(row.cpu().numpy())
            jacobian = self.kinematics.pmodel.get_link_jacobian(
                self.kinematics.end_link_idx
            )
            jacobian = torch.as_tensor(
                jacobian, device=self.device, dtype=qpos.dtype
            )
            jacobians.append(jacobian[:, self.kinematics.qmask])
        return torch.stack(jacobians, dim=0)

    def _apply_qf(self, tau: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            qf = self.articulation.px.cuda_articulation_qf.torch()
            rows = self.articulation._data_index
            qf[rows[:, None], self.active_joint_indices[None, :]] = tau
            self.articulation.px.gpu_apply_articulation_qf()
            return

        full_qf = torch.zeros_like(self.articulation.get_qf())
        full_qf[:, self.active_joint_indices] = tau
        self.articulation.set_qf(full_qf)

    def get_state(self) -> dict:
        if not self.config.expose_state:
            return {}
        state = self._impedance_state
        output = {}
        if state.target_pos is not None:
            output["target_pos"] = state.target_pos
        if state.target_quat is not None:
            output["target_quat"] = state.target_quat
        if state.pose_error is not None:
            output["pose_error"] = state.pose_error
        if state.last_torque is not None:
            output["last_torque"] = state.last_torque
        return output

    def set_state(self, state: dict):
        if "target_pos" in state and "target_quat" in state:
            self.adapter.reset(state["target_pos"], state["target_quat"])
        if "last_torque" in state:
            self._impedance_state.previous_torque = state["last_torque"]


class ImpedanceEETwistController(ImpedanceEEDeltaPoseController):
    adapter_cls = TwistCommandAdapter


@dataclass
class ImpedanceEEDeltaPoseControllerConfig(ControllerConfig):
    action_lower: Union[float, Sequence[float]] = -0.1
    action_upper: Union[float, Sequence[float]] = 0.1
    cartesian_stiffness: Union[float, Sequence[float]] = (
        200.0,
        200.0,
        200.0,
        20.0,
        20.0,
        20.0,
    )
    cartesian_damping: Optional[Union[float, Sequence[float]]] = None
    nullspace_stiffness: Union[float, Sequence[float]] = 20.0
    nullspace_damping: Optional[Union[float, Sequence[float]]] = None
    pose_error_clip: Optional[Union[float, Sequence[float]]] = (
        0.1,
        0.1,
        0.1,
        0.3,
        0.3,
        0.3,
    )
    torque_limit: Optional[Union[float, Sequence[float]]] = 100.0
    torque_rate_limit: Optional[Union[float, Sequence[float]]] = 1.0
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = ""
    urdf_path: str = ""
    root_link_name: Optional[str] = None
    normalize_action: bool = True
    expose_state: bool = False
    # TODO(robot-infra): add integral gains once state reset, clamp, and safety
    # semantics are shared across sim and real backends.
    controller_cls = ImpedanceEEDeltaPoseController


@dataclass
class ImpedanceEETwistControllerConfig(ImpedanceEEDeltaPoseControllerConfig):
    controller_cls = ImpedanceEETwistController

