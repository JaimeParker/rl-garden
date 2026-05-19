from __future__ import annotations

import torch


def normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat((zeros, v), dim=-1)
    return quat_multiply(quat_multiply(q, v_as_quat), quat_conjugate(q))[..., 1:]


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices."""
    q = normalize_quat(q)
    r, i, j, k = q.unbind(dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return mat.reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotvec(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert rotation matrices to angle-axis vectors."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    skew = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    scale = torch.where(
        sin_angle.abs() < eps,
        torch.full_like(angle, 0.5),
        angle / (2.0 * sin_angle),
    )
    return skew * scale.unsqueeze(-1)


def pose_error_current_minus_target(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Return 6D pose error in the shared/root frame.

    The returned vector is [current_pos - target_pos, current_rot_minus_target].
    It is intended for impedance laws of the form ``-K * error``.
    """
    current_rot = quat_to_matrix(current_quat)
    target_rot = quat_to_matrix(target_quat)
    rot_error = matrix_to_rotvec(torch.bmm(current_rot, target_rot.transpose(1, 2)))
    return torch.cat((current_pos - target_pos, rot_error), dim=-1)


def compose_pose(
    pos: torch.Tensor,
    quat: torch.Tensor,
    delta_pos: torch.Tensor,
    delta_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-compose a local delta pose onto a batched pose."""
    quat = normalize_quat(quat)
    delta_quat = normalize_quat(delta_quat)
    next_pos = pos + quat_apply(quat, delta_pos)
    next_quat = normalize_quat(quat_multiply(quat, delta_quat))
    return next_pos, next_quat


def axis_angle_to_quat(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(eps)
    half_angle = 0.5 * angle
    quat = torch.cat((torch.cos(half_angle), axis * torch.sin(half_angle)), dim=-1)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(angle <= eps, identity, normalize_quat(quat))

