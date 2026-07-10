# Robot Infra Roadmap

This document tracks the intended direction for `robot_infra` after the V1
ManiSkill Franka impedance controller.

**Note on the real-world RL v1 bridge:** `robot_infra/controller/real/franka_bridge.py`
forwards to the ROS-side `serl_franka_controllers` instead of a project-owned
real-time controller. This is an explicit transitional solution (see
`docs/superpowers/specs/2026-07-09-real-world-rl-design.md`) to get a
real-world RL training loop running quickly; the ROS dependency is confined
to that one file, and `FrankaRealEnv` only talks to it through an HTTP
client, so swapping the controller implementation later requires no change
above that boundary. Either way, the real-hardware control loop stays a
real-time C++ process, not a torch module under `controller/core` -- see the
`real_franka` entry below.

## V1 Boundary

- Keep controller math in pure torch modules under `robot_infra/controller/core`
  for GPU-batched simulation backends, where the point is avoiding GPU<->CPU
  handoffs across many parallel `env.step()` calls. This does not extend to
  real hardware: `num_envs` is always 1 on a physical robot, so there is no
  batching benefit to recoup, and `libfranka`'s control loop has a hard
  real-time deadline (1kHz) that a Python/torch process cannot reliably meet
  regardless of how fast the tensor ops are.
- Keep simulator-specific object access and command application in backend wrappers.
- Treat ManiSkill as the first backend, not the owner of the control algorithm.
- Keep GPU training paths batched and tensor-native.

## Near-Term Extensions

- Add a dynamics provider interface for optional Coriolis, gravity, mass matrix,
  and inverse dynamics terms.
- Implement a CPU Pinocchio provider for smoke tests and a GPU-compatible provider
  only when the required tensors are available without CPU transfer.
- Add an integral impedance term (`Ki`) with explicit reset, clamp, and anti-windup
  behavior in `controller/core`, mirrored (not physically shared -- see V1
  Boundary) in the real-time controller for behavioral parity.
- Add a joint-delta adapter that maps joint commands into nullspace targets or a
  dedicated joint impedance command.

## Backend Roadmap

- `maniskill`: current wrapper, qf torque application, pytorch_kinematics Jacobian.
- `mujoco`: future wrapper for state/Jacobian/torque command extraction.
- `real_franka`: future real-time controller providing hardware safety gates,
  torque-rate limits, fault handling, and calibrated model/dynamics providers.
  Runs as its own real-time C++ process (today: `serl_franka_controllers`;
  potentially a project-owned successor later) -- never a torch module under
  `controller/core`. `robot_infra/controller/real/` owns only the non-real-time
  interface into it.

## Safety And Observability

- Add backend-level torque, velocity, workspace, and pose target guards before any
  real robot deployment.
- Keep controller diagnostics opt-in so training observation spaces do not change
  accidentally.
- Standardize logging for target pose, pose error, commanded torque, torque clamp,
  and optional measured wrench.

## Real-World RL Training-Layer TODOs (`rl_garden` side)

These are open design questions on the training side (`rl_garden/real_world/`,
`rl_garden/training/real_world/`), not the `robot_infra` controller boundary
above. Surfaced while researching `3rd_party/RLinf`'s real-world path; not yet
decided.

- **Buffer layer**: still open, and still not needed. `hil_serl` (landed --
  see below) targets only HIL-SERL's `train_rlpd.py` capability set (online
  RLPD + demo mixing + HITL + reward classifier), which uses RLPD's existing
  static `offline_dataset_path` loading, same as SERL. HG-DAgger's
  iterative, growing human-correction dataset (which would actually need
  this redesign) was explicitly scoped out of the `hil_serl` migration.
  `LearnerLoop._refresh_offline_data()` (added pre-emptively during the SERL
  v1 base-class work, reserved for exactly this) is still an unused no-op in
  both `serl` and `hil_serl`. RLinf's real-world DAgger/SFT path (a
  disk-file dataset pipeline decoupled from the in-memory RL replay buffer:
  a LeRobot-format writer, a separate offline ETL step, and a
  lazily-reloaded PyTorch `Dataset`) remains the reference design for
  whenever HG-DAgger is actually migrated.
- **Runner/launcher layer**: done for both SERL and HIL-SERL, fully
  mirroring `rl_garden/training/online/`'s shape (registry mechanism, not
  just its "one package per concern" spirit): `rl_garden/real_world/` holds
  base classes (`ActorLoop`/`LearnerLoop`/sync) plus `serl/` and `hil_serl/`
  subpackages of concrete (currently empty) subclasses;
  `rl_garden/training/real_world/` holds a shared `RealWorldFrankaArgs`
  base, a `RealWorldAlgorithmRegistry` (`_registry.py`), and one flat file
  per method (`serl.py`, `hil_serl.py`, both shaped like
  `training/online/rlpd.py`) that self-registers at import time;
  `examples/train_real_world.py` is a generic `registry.run_cli()`
  launcher, method selected via subcommand. `hil_serl.py` explicitly
  composes the env wrappers from Component 3
  (`TeleopInterventionWrapper`/`RewardClassifierWrapper`/
  `FWBWResetFreeWrapper`) in its own `_build_env` (no generic flag-driven
  wrapper composer). Its hybrid continuous+discrete gripper action space is
  a new algorithm, `RLPDHybrid` (`rl_garden/algorithms/rlpd_hybrid.py`),
  registered as an ordinary online algorithm
  (`rl_garden/training/online/rlpd_hybrid.py`, usable via `--algo
  rlpd_hybrid` in simulation too) and reused by `hil_serl.py` the same way
  `serl.py` reuses plain `RLPD`. Not migrated this round: BC (HIL-SERL's own
  code confirms it's never combined with the online RLPD loss, so it's a
  fully independent future addition). The reward-classifier model,
  inference loader, training script, and data-collection script now live
  under `rl_garden/models/reward/success/` (moved from
  `rl_garden/envs/franka_real/classifier.py` and completed -- see
  `docs/hil_serl_roadmap.md` item 3), alongside a reorganized
  `rl_garden/models/reward/` that also holds the pre-existing offline
  HDF5-labeled classifiers (`classifiers/`).

