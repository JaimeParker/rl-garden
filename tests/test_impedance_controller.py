from __future__ import annotations

import torch
import pytest

from robot_infra.controller.adapters import DeltaPoseCommandAdapter, TwistCommandAdapter
from robot_infra.controller.core import (
    ImpedanceCommand,
    ImpedanceConfig,
    ImpedanceControllerState,
    compute_impedance_torque,
)


def _identity_quat(batch: int, device: torch.device | str = "cpu"):
    quat = torch.zeros(batch, 4, device=device)
    quat[:, 0] = 1.0
    return quat


def test_zero_error_and_zero_velocity_produce_zero_torque():
    qpos = torch.zeros(2, 7)
    qvel = torch.zeros(2, 7)
    jacobian = torch.randn(2, 6, 7)
    ee_pos = torch.zeros(2, 3)
    ee_quat = _identity_quat(2)
    command = ImpedanceCommand(
        target_pos=ee_pos.clone(),
        target_quat=ee_quat.clone(),
        nullspace_qpos=qpos.clone(),
    )

    tau, state = compute_impedance_torque(
        qpos=qpos,
        qvel=qvel,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        command=command,
        config=ImpedanceConfig(torque_rate_limit=None),
    )

    assert torch.allclose(tau, torch.zeros_like(tau), atol=1e-6)
    assert torch.allclose(state.pose_error, torch.zeros(2, 6), atol=1e-6)


def test_cartesian_wrench_maps_through_jacobian_transpose():
    qpos = torch.zeros(1, 6)
    qvel = torch.zeros(1, 6)
    jacobian = torch.eye(6).reshape(1, 6, 6)
    ee_pos = torch.zeros(1, 3)
    ee_quat = _identity_quat(1)
    wrench = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    command = ImpedanceCommand(
        target_pos=ee_pos.clone(),
        target_quat=ee_quat.clone(),
        wrench=wrench,
    )

    tau, _ = compute_impedance_torque(
        qpos=qpos,
        qvel=qvel,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        command=command,
        config=ImpedanceConfig(
            cartesian_stiffness=0.0,
            cartesian_damping=0.0,
            nullspace_stiffness=0.0,
            torque_rate_limit=None,
        ),
    )

    assert torch.allclose(tau, wrench)


def test_torque_rate_and_torque_limits_are_applied():
    qpos = torch.zeros(1, 6)
    qvel = torch.zeros(1, 6)
    jacobian = torch.eye(6).reshape(1, 6, 6)
    ee_pos = torch.zeros(1, 3)
    ee_quat = _identity_quat(1)
    command = ImpedanceCommand(
        target_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        target_quat=ee_quat.clone(),
    )

    tau, _ = compute_impedance_torque(
        qpos=qpos,
        qvel=qvel,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        command=command,
        config=ImpedanceConfig(
            cartesian_stiffness=10.0,
            cartesian_damping=0.0,
            nullspace_stiffness=0.0,
            pose_error_clip=None,
            torque_limit=2.0,
            torque_rate_limit=0.5,
        ),
        state=ImpedanceControllerState(previous_torque=torch.zeros(1, 6)),
    )

    assert torch.allclose(tau, torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_delta_pose_adapter_updates_target_pose():
    adapter = DeltaPoseCommandAdapter()
    adapter.reset(torch.zeros(1, 3), _identity_quat(1))
    adapter.set_action(torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    command = adapter.command()

    assert torch.allclose(command.target_pos, torch.tensor([[0.1, 0.0, 0.0]]))
    assert torch.allclose(command.target_quat, _identity_quat(1))


def test_twist_adapter_uses_same_increment_contract_as_delta_pose():
    adapter = TwistCommandAdapter()
    adapter.reset(torch.zeros(1, 3), _identity_quat(1))
    adapter.set_action(torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.0]]))

    command = adapter.command()

    assert torch.allclose(command.target_pos, torch.tensor([[0.0, 0.2, 0.0]]))
    assert command.target_quat.shape == (1, 4)


def test_impedance_core_preserves_cuda_device_when_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    try:
        qpos = torch.zeros(1, 7, device=device)
        qvel = torch.zeros(1, 7, device=device)
        jacobian = torch.zeros(1, 6, 7, device=device)
        ee_pos = torch.zeros(1, 3, device=device)
        ee_quat = _identity_quat(1, device=device)
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            pytest.skip(f"CUDA device is out of memory: {err}")
        raise
    command = ImpedanceCommand(target_pos=ee_pos.clone(), target_quat=ee_quat.clone())

    tau, state = compute_impedance_torque(
        qpos=qpos,
        qvel=qvel,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        jacobian=jacobian,
        command=command,
        config=ImpedanceConfig(torque_rate_limit=None),
    )

    assert tau.is_cuda
    assert state.last_torque.is_cuda
