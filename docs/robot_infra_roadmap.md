# Robot Infra Roadmap

This document tracks the intended direction for `robot_infra` after the V1
ManiSkill Franka impedance controller.

## V1 Boundary

- Keep controller math in pure torch modules under `robot_infra/controller/core`.
- Keep simulator-specific object access and command application in backend wrappers.
- Treat ManiSkill as the first backend, not the owner of the control algorithm.
- Keep GPU training paths batched and tensor-native.

## Near-Term Extensions

- Add a dynamics provider interface for optional Coriolis, gravity, mass matrix,
  and inverse dynamics terms.
- Implement a CPU Pinocchio provider for smoke tests and a GPU-compatible provider
  only when the required tensors are available without CPU transfer.
- Add an integral impedance term (`Ki`) with explicit reset, clamp, and anti-windup
  behavior shared across sim and real backends.
- Add a joint-delta adapter that maps joint commands into nullspace targets or a
  dedicated joint impedance command.

## Backend Roadmap

- `maniskill`: current wrapper, qf torque application, pytorch_kinematics Jacobian.
- `mujoco`: future wrapper for state/Jacobian/torque command extraction.
- `real_franka`: future wrapper with hardware safety gates, torque-rate limits,
  fault handling, and calibrated model/dynamics providers.

## Safety And Observability

- Add backend-level torque, velocity, workspace, and pose target guards before any
  real robot deployment.
- Keep controller diagnostics opt-in so training observation spaces do not change
  accidentally.
- Standardize logging for target pose, pose error, commanded torque, torque clamp,
  and optional measured wrench.

