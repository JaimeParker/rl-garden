from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from robot_infra.controller.core.pose_torch import pose_error_current_minus_target


ScalarOrSequence = float | Sequence[float] | torch.Tensor


@dataclass
class ImpedanceConfig:
    cartesian_stiffness: ScalarOrSequence = (200.0, 200.0, 200.0, 20.0, 20.0, 20.0)
    cartesian_damping: Optional[ScalarOrSequence] = None
    nullspace_stiffness: ScalarOrSequence = 20.0
    nullspace_damping: Optional[ScalarOrSequence] = None
    pose_error_clip: Optional[ScalarOrSequence] = (0.1, 0.1, 0.1, 0.3, 0.3, 0.3)
    torque_limit: Optional[ScalarOrSequence] = None
    torque_rate_limit: Optional[ScalarOrSequence] = 1.0


@dataclass
class ImpedanceCommand:
    target_pos: torch.Tensor
    target_quat: torch.Tensor
    nullspace_qpos: Optional[torch.Tensor] = None
    wrench: Optional[torch.Tensor] = None


@dataclass
class ImpedanceControllerState:
    previous_torque: Optional[torch.Tensor] = None
    target_pos: Optional[torch.Tensor] = None
    target_quat: Optional[torch.Tensor] = None
    pose_error: Optional[torch.Tensor] = None
    last_torque: Optional[torch.Tensor] = None


def _as_gain(
    value: Optional[ScalarOrSequence],
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    default_from_stiffness: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if value is None:
        if default_from_stiffness is None:
            return None
        return 2.0 * torch.sqrt(default_from_stiffness.clamp_min(0.0))
    out = torch.as_tensor(value, device=device, dtype=dtype)
    if out.ndim == 0:
        out = out.repeat(width)
    if out.numel() != width:
        raise ValueError(f"Expected gain width {width}, got shape {tuple(out.shape)}")
    return out.reshape(1, width)


def _clip_symmetric(value: torch.Tensor, limit: Optional[torch.Tensor]) -> torch.Tensor:
    if limit is None:
        return value
    return value.clamp(min=-limit, max=limit)


def compute_impedance_torque(
    *,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    jacobian: torch.Tensor,
    command: ImpedanceCommand,
    config: ImpedanceConfig,
    state: Optional[ImpedanceControllerState] = None,
) -> tuple[torch.Tensor, ImpedanceControllerState]:
    """Compute batched Cartesian impedance torques with torch tensors.

    All tensors must be on the same device. Quaternion convention is wxyz.
    Jacobian shape is ``(batch, 6, dof)`` with linear rows first.
    """
    if jacobian.ndim != 3:
        raise ValueError(f"Expected jacobian shape (B, 6, dof), got {jacobian.shape}")
    batch, rows, dof = jacobian.shape
    if rows != 6:
        raise ValueError(f"Expected 6 Jacobian rows, got {rows}")
    if qpos.shape != (batch, dof) or qvel.shape != (batch, dof):
        raise ValueError(
            f"qpos/qvel must have shape {(batch, dof)}, got {qpos.shape}/{qvel.shape}"
        )

    device = qpos.device
    dtype = qpos.dtype
    kx = _as_gain(config.cartesian_stiffness, 6, device=device, dtype=dtype)
    dx = _as_gain(
        config.cartesian_damping,
        6,
        device=device,
        dtype=dtype,
        default_from_stiffness=kx,
    )
    pose_clip = _as_gain(config.pose_error_clip, 6, device=device, dtype=dtype)
    torque_limit = _as_gain(config.torque_limit, dof, device=device, dtype=dtype)
    torque_rate_limit = _as_gain(
        config.torque_rate_limit, dof, device=device, dtype=dtype
    )

    pose_error = pose_error_current_minus_target(
        ee_pos, ee_quat, command.target_pos, command.target_quat
    )
    pose_error = _clip_symmetric(pose_error, pose_clip)
    ee_twist = torch.bmm(jacobian, qvel.unsqueeze(-1)).squeeze(-1)
    wrench = torch.zeros((batch, 6), device=device, dtype=dtype)
    if command.wrench is not None:
        wrench = command.wrench.to(device=device, dtype=dtype)
    task_wrench = -kx * pose_error - dx * ee_twist + wrench
    tau_task = torch.bmm(jacobian.transpose(1, 2), task_wrench.unsqueeze(-1)).squeeze(-1)

    tau_nullspace = torch.zeros_like(tau_task)
    if command.nullspace_qpos is not None:
        kq = _as_gain(config.nullspace_stiffness, dof, device=device, dtype=dtype)
        dq_gain = _as_gain(
            config.nullspace_damping,
            dof,
            device=device,
            dtype=dtype,
            default_from_stiffness=kq,
        )
        joint_cmd = kq * (command.nullspace_qpos.to(device=device, dtype=dtype) - qpos)
        joint_cmd = joint_cmd - dq_gain * qvel
        jacobian_t = jacobian.transpose(1, 2)
        jacobian_t_pinv = torch.linalg.pinv(jacobian_t)
        eye = torch.eye(dof, device=device, dtype=dtype).expand(batch, dof, dof)
        nullspace_projector = eye - torch.bmm(jacobian_t, jacobian_t_pinv)
        tau_nullspace = torch.bmm(
            nullspace_projector, joint_cmd.unsqueeze(-1)
        ).squeeze(-1)

    tau = tau_task + tau_nullspace
    if state is not None and state.previous_torque is not None:
        previous = state.previous_torque.to(device=device, dtype=dtype)
        delta = _clip_symmetric(tau - previous, torque_rate_limit)
        tau = previous + delta
    tau = _clip_symmetric(tau, torque_limit)

    next_state = ImpedanceControllerState(
        previous_torque=tau.detach(),
        target_pos=command.target_pos.detach(),
        target_quat=command.target_quat.detach(),
        pose_error=pose_error.detach(),
        last_torque=tau.detach(),
    )
    return tau, next_state

