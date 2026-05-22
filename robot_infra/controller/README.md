# Robot Infra Controllers

`robot_infra.controller` separates controller math from simulator and robot
bindings. The goal is to keep control algorithms reusable while allowing each
backend to handle its own state extraction and command application details.

## Layout

- `core/`: pure torch controller algorithms and math utilities. Code here must
  not import ManiSkill, SAPIEN, Gymnasium, or robot-specific wrappers.
- `adapters/`: command adapters that convert policy or teleop actions into
  controller commands. For example, EE delta pose and EE twist adapters update a
  Cartesian impedance equilibrium target.
- `simulator/`: simulator-specific wrappers. These modules may import simulator
  APIs and are responsible for reading simulator state, computing backend
  kinematics, and applying commands.
- `simulator/maniskill/`: ManiSkill controller wrappers for the torch impedance
  core. These wrappers integrate with ManiSkill `ControllerConfig`, read Panda
  state/Jacobians, and apply joint torques through qf.

## Import Boundaries

Keep dependencies flowing in one direction:

```text
simulator wrappers -> adapters -> core
```

The core and adapter layers should stay backend-agnostic. If a change requires
ManiSkill or SAPIEN objects, it belongs in `simulator/maniskill` or another
backend wrapper.

## Current Impedance Controller

The V1 Franka impedance controller is GPU-first and torque-level in simulation.
It computes batched torch torques from current joint state, EE pose, Jacobian,
and a Cartesian target command. The ManiSkill wrapper writes the resulting arm
torques to qf without modifying ManiSkill source code.

Current limitations:

- No Coriolis or gravity compensation on the GPU path.
- No integral impedance term yet.
- EE delta pose and EE twist adapters are implemented; joint-delta adapters are
  planned but not part of V1.
- Controller diagnostics are opt-in and should not be added to task
  observations by default.

See `docs/robot_infra_roadmap.md` for planned backend, dynamics, safety, and
observability work.

