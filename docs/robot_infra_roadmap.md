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

- **Buffer layer**: `LearnerLoop`'s `agent.replay_buffer.add()` path assumes
  either live online transitions or a static offline dataset loaded once at
  startup (RLPD's `offline_dataset_path`, e.g. Minari). Upcoming HIL-SERL
  migration needs iterative, growing human-demonstration/correction data
  (BC/DAgger-style), which does not fit that model. RLinf's real-world
  DAgger/SFT path instead uses a disk-file dataset pipeline decoupled from its
  in-memory RL replay buffer (a LeRobot-format writer, a separate offline ETL
  step, and a lazily-reloaded PyTorch `Dataset` that picks up newly written
  data). Design this data path -- don't try to force growing demo data through
  the existing in-memory buffer interface.
- **Runner/launcher layer**: done for the SERL side, now fully mirroring
  `rl_garden/training/online/`'s shape (not just its "one package per
  concern" spirit, but the exact registry mechanism): `rl_garden/real_world/`
  holds base classes (`ActorLoop`/`LearnerLoop`/sync) plus a `serl/`
  subpackage of concrete (currently empty) subclasses;
  `rl_garden/training/real_world/` holds a shared `RealWorldFrankaArgs` base,
  a `RealWorldAlgorithmRegistry` (`_registry.py`), and one flat file per
  method (`serl.py`, shaped like `training/online/rlpd.py`) that
  self-registers at import time; `examples/train_real_world.py` is a generic
  `registry.run_cli()` launcher, algorithm selected via subcommand. `hil_serl.py`
  is not built yet -- its algorithm-layer components (BC/DAgger, reward
  classifier training) don't exist in `rl_garden/algorithms/` yet, so
  scaffolding an empty sibling file now would be speculative. When it lands,
  it explicitly composes the env wrappers from Component 3 in its own
  `_run_actor`/`_run_learner` (no generic flag-driven wrapper composer -- see
  the design spec).

