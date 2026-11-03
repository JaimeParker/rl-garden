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
(`real_world_rl/` and its `robot_infra/` device layer), which carries
rl-garden as a submodule under `rl-garden/`; see the "Robot Infrastructure and
Reward Models" section in the top-level [README.md](../README.md).

- [Robot Infra Roadmap](https://github.com/JaimeParker/real-world-rl/blob/main/robot_infra/docs/robot-infra-roadmap.md)
- [HIL-SERL Roadmap](https://github.com/JaimeParker/real-world-rl/blob/main/docs/hil-serl-roadmap.md)
