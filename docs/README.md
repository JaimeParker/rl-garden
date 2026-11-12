# Documentation Index

## Guides

Operational how-to for running and configuring the framework.

- [Checkpoint Save & Load](guides/checkpoint.md)
- [Configuration System](guides/configuration.md)
- [D4RL Legacy Manipulation Baselines](guides/d4rl-legacy-expansion.md)
- [Decision-Level (Semi-Markov) PPO](guides/decision-level-ppo.md)
- [Extending rl-garden from Your Own Package](guides/extending.md)
- [IsaacLab Camera Training Stall (resolved on Isaac Lab 3.0)](guides/isaaclab-camera-stall.md)
- [IsaacLab Custom Tasks](guides/isaaclab-custom-tasks.md)
- [Meta-World Integration](guides/metaworld-integration.md)
- [Offline Training Acceleration](guides/offline-acceleration.md)
- [OGBench Integration](guides/ogbench-integration.md)
- [Policy Graphs (YAML-Declared PPO Policies)](guides/policy-graphs.md)
- [RLBench Integration](guides/rlbench-integration.md)
- [robomimic Integration](guides/robomimic-integration.md)
- [RoboTwin Integration](guides/robotwin.md)
- [Teleoperation and Recording](https://github.com/JaimeParker/real-world-rl/blob/main/docs/teleop.md) (in `real-world-rl`)
- [WSRL Reproduction](guides/wsrl-reproduction.md)

## Design

Architecture and rationale behind specific subsystems.

- [WSRL Overview](design/wsrl-overview.md)
- [IQL Overview](design/iql-overview.md)
- [RLinf Integration](design/rlinf-integration.md)
- [RNG and Numerical Determinism](design/rng-numerical-determinism.md)

## Robot Infra and Real-World RL

Both live in the separate downstream repository `real-world-rl`
(`real_world_rl/`, including the device layer under
`real_world_rl/{devices,server,discovery,setup,supervise,doctor}` and teleop under
`real_world_rl/teleop/<device>/`), which carries
rl-garden as a submodule under `rl-garden/`; see the "Robot Infrastructure and
Reward Models" section in the top-level [README.md](../README.md).

- [Installation](https://github.com/JaimeParker/real-world-rl/blob/main/docs/tutorial/01-install.md) (`real-world-rl` 0.1.0, `[learning]` extra)
- [rwrl Command Reference](https://github.com/JaimeParker/real-world-rl/blob/main/docs/cli.md)
- [Device Layer Roadmap](https://github.com/JaimeParker/real-world-rl/blob/main/docs/devices/robot-infra-roadmap.md)
- [HIL-SERL Roadmap](https://github.com/JaimeParker/real-world-rl/blob/main/docs/hil-serl-roadmap.md)
