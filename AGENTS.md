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
- Before navigating an unfamiliar area of the codebase, read
  [`.agents/rules/repository-map.md`](.agents/rules/repository-map.md).
- Before installing or running an official JAX baseline (Cal-QL, wsrl,
  IQL-jax) for numeric comparison against rl-garden, read
  [`.agents/runbooks/baseline-install.md`](.agents/runbooks/baseline-install.md).

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
- Keep optional environment and hardware dependencies lazy so core package imports,
  registry discovery, and configuration inspection work without every backend
  installed.

## Development Rules

- Inspect existing patterns with `rg` before editing.
- Make surgical changes. Do not refactor adjacent code or reformat unrelated files.
- Maximize reuse via subclassing: override in a subclass rather than editing a
  shared parent. Add a parent hook only if ≥2 *existing* subclasses need
  genuinely different behavior there (never a hypothetical future one), with a
  no-op/trivial default for the rest -- otherwise implement it in that one
  subclass.
- Keep algorithm-specific fields in subclasses. If parent-side variability is
  unavoidable, prefer an overridable class attribute over `hasattr` branches.
- Keep optimizer ownership explicit for actor, critic, encoder, entropy coefficient,
  and value networks.
- Add focused tests for changed behavior and relevant edge cases.
- Code(name classes, functions, and variables) should be self-explanatory.
- Comment when necessary, but avoid redundant comments.

## Verification

Before finishing code changes:

1. Run the smallest relevant test set first, prefer to run pytest on remote.
2. For environment or training changes, run a tiny smoke test where dependencies
   and hardware permit it.
3. Report exact commands and results.
4. If an accelerator, renderer, or optional backend dependency is unavailable,
   report the traceback and device status instead of silently changing execution
   semantics.
5. Run `git status --short` before and after changes and preserve unrelated work.
