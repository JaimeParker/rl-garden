# Repository Map

Top-level layout of `rl-garden`. Read `AGENTS.md` first for the project-level
agent rules; this file is a plain orientation reference, not a rules doc.

- `rl_garden/algorithms/` — online, offline, and off-to-online algorithms.
- `rl_garden/policies/` — policy composition and actor/critic modules.
- `rl_garden/buffers/` — tensor, dict, Monte-Carlo, and rollout buffers.
- `rl_garden/encoders/` — state, CNN, RGBD/proprio, pooling, FiLM, and ResNet encoders.
- `rl_garden/networks/` — actor, critic, value, and MLP backbone builders.
- `rl_garden/common/` — logging, shared CLI arguments, environment arguments,
  checkpoint I/O, optimization, types, and utilities.
- `rl_garden/envs/` — backend registry and implementations, environment factories,
  wrappers, and custom environments.
- `rl_garden/models/` — ACT and reward models.
- `rl_garden/real_world/` — real-robot actor/learner loops and HIL-SERL/SERL
  integration.
- `rl_garden/training/` — registry base and independent `online/`, `offline/`, and
  `off2on/` packages.
- `robot_infra/` — controllers, teleoperation, and real-robot support.
- `examples/` — thin training dispatchers and specialized experiment entrypoints.
- `configs/` — reusable preset configs for training.
- `scripts/` — shell launchers with experiment defaults.
- `tools/` — standalone utilities: `conversion/` (checkpoint/dataset format
  conversion), `diagnostics/` (Q-value and parity probes), `reproductions/`
  (third-party baseline reproduction runners).
- `pretrained/` — externally pretrained weights (ResNet, ACT), outside the
  importable package tree.
- `tests/` — unit tests and accelerator/backend integration smoke tests.
- `docs/` — public documentation, split into `guides/` (operational how-to),
  `design/` (architecture and rationale), and `roadmaps/` (migration-tracking
  notes).
- `3rd_party/` — vendored reference submodules and external clones; read-only,
  do not edit unless explicitly requested.
