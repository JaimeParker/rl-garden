# AGENTS.md

Operating guide for coding agents working on `rl-garden`. Read [README.md](README.md)
for user-facing installation, features, and quick-start documentation.

## Required Context

- Before changing training entrypoints, argument ownership, algorithm registration,
  environment backends, or replay/device behavior, read
  [`.agents/rules/training-development.md`](.agents/rules/training-development.md).
- Before adding a new environment backend, read
  [`.agents/rules/adding-env-backend.md`](.agents/rules/adding-env-backend.md).
- Before adding a new algorithm or training entrypoint, read
  [`.agents/rules/adding-algorithm.md`](.agents/rules/adding-algorithm.md).
- Before launching training or evaluation, read
  [`.agents/runbooks/training.md`](.agents/runbooks/training.md).
- Before saving, loading, or resuming checkpoints, read
  [`.agents/runbooks/checkpoint.md`](.agents/runbooks/checkpoint.md).
- Before remote training, evaluation, or debugging, read
  [`.agents/rules/remote-training-sop.md`](.agents/rules/remote-training-sop.md).
- Before changing or repairing Mutagen sync, read
  [`.agents/rules/mutagen-sync-sop.md`](.agents/rules/mutagen-sync-sop.md).

Machine-specific server, Docker, path, Python environment, and Mutagen bindings
belong in ignored `.agents/local/personal_config.md`. Before any remote command,
read that file. If it is missing, stop and ask the user to create it from
`.agents/local/personal_config.md.example`.

## Project Constraints

`rl-garden` is a PyTorch-native robot-learning (especially for RL) framework for simulation,
offline datasets, and real-robot systems. Its environment backend architecture is
designed to support additional platforms without coupling algorithms or training
entrypoints to a specific simulator.

- Training is GPU-first. Normal rollouts, replay buffers, sampled batches,
  inference, and updates should stay on CUDA tensors.
- Avoid NumPy handoffs in rollout, replay, and update hot paths.
- Replay buffers use torch tensors with `(T, N, ...)` layout. Keep storage and
  sample devices explicit.
- Keep device transfers explicit for every environment backend. CPU-backed
  environments are supported, but must not introduce implicit CPU copies into a
  GPU training path.
- RGBD actor and critic share the encoder; actor updates detach encoder features.
- Keep optional environment and hardware dependencies lazy so core package imports,
  registry discovery, and configuration inspection work without every backend
  installed.

## Repository Map

- `rl_garden/algorithms/` — online, offline, off-to-online, and residual algorithms.
- `rl_garden/policies/` — policy composition and actor/critic modules.
- `rl_garden/buffers/` — tensor, dict, Monte-Carlo, residual, and rollout buffers.
- `rl_garden/encoders/` — state, CNN, RGBD/proprio, pooling, FiLM, and ResNet encoders.
- `rl_garden/networks/` — actor, critic, value, and MLP backbone builders.
- `rl_garden/common/` — logging, shared CLI arguments, environment arguments,
  checkpoint I/O, optimization, types, and utilities.
- `rl_garden/envs/` — backend registry and implementations, environment factories,
  wrappers, and custom environments.
- `rl_garden/training/` — registry base and independent `online/`, `offline/`, and
  `off2on/` packages. Each phase owns shared parameters in `_args.py`; public
  algorithm modules provide final Args composition, run function, and registration.
- `robot_infra/` — controllers, teleoperation, and real-robot support.
- `examples/` — thin training dispatchers and specialized experiment entrypoints.
- `scripts/` — launchers with experiment defaults.
- `tests/` — unit tests and accelerator/backend integration smoke tests.
- `docs/` — public design, workflow, and operational documentation.
- `3rd_party/` — reference submodules and external clones; do not edit unless the
  user explicitly requests it.

## Development Rules

- Inspect existing patterns with `rg` before editing.
- Make surgical changes. Do not refactor adjacent code or reformat unrelated files.
- Prefer subclass overrides over changing parent classes. Add a parent hook only
  when it is generic, has a no-op or trivially correct default, and benefits more
  than one implementation.
- Keep algorithm-specific fields in subclasses. If parent-side variability is
  unavoidable, prefer an overridable class attribute over `hasattr` branches.
- Keep optimizer ownership explicit for actor, critic, encoder, entropy coefficient,
  and value networks.
- Add focused tests for changed behavior and relevant edge cases.

## Verification

Before finishing code changes:

1. Run the smallest relevant test set first.
2. For environment or training changes, run a tiny smoke test where dependencies
   and hardware permit it.
3. Report exact commands and results.
4. If an accelerator, renderer, or optional backend dependency is unavailable,
   report the traceback and device status instead of silently changing execution
   semantics.
5. Run `git status --short` before and after changes and preserve unrelated work.
