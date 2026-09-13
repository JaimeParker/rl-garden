# Repository Map

Top-level layout of `rl-garden`. Read `AGENTS.md` first for the project-level
agent rules; this file is a plain orientation reference, not a rules doc.

- `rl_garden/algorithms/` — online, offline, and off-to-online algorithms.
  `_observation.py` holds `ObservationEncoderMixin` (Layer C: turns an
  algorithm's `encoder_config`/`obs_groups`/`critic_encoder_config`/
  `encoder_sharing` into built feature extractors via `_policy_extractor_kwargs()`).
- `rl_garden/observations/` — the observation composition contract (Layer A):
  `ObservationConfig` ("what is observed" — state/rgb/depth cameras/extra_state
  auxiliary keys, image size, frame stacking), `ObservationSchema`/`Modality`/
  `ObsEntry` (derived from an observation space), `ObsGroups`/`resolve_obs_groups`
  (asymmetric actor/critic observation keys), and the strict `state`/`state_<name>`/
  `rgb_<cam>`/`depth_<cam>` key-vocabulary validation every backend/dataset loader
  must satisfy.
- `rl_garden/policies/` — policy composition and actor/critic modules.
  `BasePolicy` owns the extractor contract: policies subclass it, call
  `super().__init__(..., actor_extractor=..., critic_extractor=...,
  encoder_sharing=...)`, and use `extract_actor_features()`/`extract_critic_features()`
  and their dimension properties for head construction.
- `rl_garden/buffers/` — tensor, dict, Monte-Carlo, and rollout buffers.
- `rl_garden/encoders/` — state, CNN, RGBD/proprio, pooling, FiLM, and ResNet
  encoders (Layer B — "how observations are encoded"). `config.py` holds
  `EncoderConfig` (the `--encoder.*` CLI surface); `registry.py` holds
  `ENCODER_REGISTRY`/`EncoderSpec` (backbone name -> factory); `factory.py`
  holds `build_observation_encoder(observation_space, encoder_config, ...)`,
  the one entry point that builds a `FlattenExtractor` (state-only) or
  `CombinedExtractor` (Dict with images) feature extractor.
- `rl_garden/networks/` — actor, critic, value, and MLP backbone builders.
- `rl_garden/common/` — logging, shared CLI arguments, environment arguments,
  checkpoint I/O, optimization, types, and utilities.
- `rl_garden/envs/` — backend registry and implementations, environment factories,
  wrappers, and custom environments.
- `rl_garden/models/` — ACT and reward models.
- `rl_garden/training/` — registry base and independent `online/`, `offline/`, and
  `off2on/` packages.
- `robot_infra/` — optional submodule ([rlgarden-robot-infra](https://github.com/JaimeParker/rlgarden-robot-infra)):
  controllers, teleoperation, real-robot support. No dependency on `rl_garden`.
- Real-robot actor/learner loops, the `franka_real` env backend, and
  HIL-SERL/SERL integration live in a separate repo,
  [rlgarden-real-world](https://github.com/JaimeParker/rlgarden-real-world) —
  not under `rl_garden/` at all. It imports `rl_garden` as a library and
  registers `franka_real` through the `rlgarden.env_backends` entry-point
  group (`rl_garden/envs/backend_registry.py`).
- `examples/` — thin training dispatchers and specialized experiment entrypoints.
- `configs/` — reusable preset configs for training.
- `scripts/` — shell launchers with experiment defaults.
- `tools/` — standalone utilities: `conversion/` (checkpoint/dataset format
  conversion), `diagnostics/` (Q-value and parity probes), `reproductions/`
  (third-party baseline reproduction runners that *patch* a specific idea into
  a copied source tree, e.g. `run_iql_fixed_mixing.py` — different purpose
  from `baselines/`, see below).
- `baselines/` — top-level package for running *unmodified* official JAX
  baseline repos (Cal-QL, wsrl, IQL-jax) against rl-garden's canonical
  environments, for pure numeric comparison. See
  [`.agents/runbooks/baseline-install.md`](.agents/runbooks/baseline-install.md).
  Not to be confused with `rl_garden/integrations/rlinf/`, which is the
  opposite direction — rl-garden's own algorithms running as workers under
  RLinf.
- `pretrained/` — externally pretrained weights (ResNet, ACT), outside the
  importable package tree.
- `tests/` — unit tests and accelerator/backend integration smoke tests.
- `docs/` — public documentation, split into `guides/` (operational how-to),
  `design/` (architecture and rationale), and `roadmaps/` (migration-tracking
  notes).
- `3rd_party/` — vendored reference submodules and external clones; read-only,
  do not edit unless explicitly requested. `Cal-QL`, `wsrl`, and
  `implicit_q_learning` are registered as real git submodules (see
  `baselines/baselines.yaml`); the rest are untracked reference clones.
