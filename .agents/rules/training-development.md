# Training Development Rules

Read this file before changing training entrypoints, configuration ownership,
algorithm registration, environment backends, or replay/device behavior.

## Entrypoints and Registration

- `examples/train_online.py`, `examples/pretrain_offline.py`, and
  `examples/train_off2on.py` are thin registry dispatchers. Do not edit them to add
  an algorithm.
- Add a public algorithm module under the appropriate
  `rl_garden/training/{online,offline,off2on}/` package. It must define the final
  Args composition, run function, and package-local registry call.
- Put parameters shared by algorithms in the phase-local `_args.py`. Keep only
  genuinely cross-phase CLI primitives and helpers in `rl_garden/common/cli_args.py`.
- `EnvRunArgs` and backend configuration are cross-phase environment concerns and
  belong in `rl_garden/common/env_args.py`.
- Observation/encoder CLI ownership is centralized: every algorithm that
  accepts observations mixes in `ObservationArgs`
  (`rl_garden/common/cli_args.py`, adds `obs`/`encoder`/`obs_groups`/
  `critic_encoder`/`encoder_sharing`) rather than declaring its own vision
  fields. Build the `EnvRequest` for a training entrypoint through the one
  shared `make_env_request(args, run_name=None, *, create_eval_env=None)`
  (`rl_garden/common/env_args.py`) instead of a per-algorithm
  `_<algo>_env_request` helper; it reads `args.obs` (falling back to a
  state-only `ObservationConfig()` when the args class has no `obs` field)
  into `EnvRequest.observation`.
- Keep imports lazy during registry discovery so listing algorithms and
  `--print-config` do not eagerly load optional simulator dependencies.

## Environment Backends

- Add a backend in `rl_garden/envs/backends/<name>.py` by subclassing `EnvBackend`,
  implementing `make_train_env(req)` and `make_eval_env(req)`, and calling
  `register_env_backend("<name>", MyBackend)`.
- The backend module is auto-discovered by `discover_env_backends()` via
  `pkgutil` — nothing is imported from `rl_garden/envs/backends/__init__.py`.
  Only add the config dataclass field on `EnvBackendArgs` in
  `rl_garden/common/env_args.py`.
- Training run functions access backend-specific settings through
  `EnvRequest.backend_config`; they must not call `make_maniskill_env()` directly.
- Extend `ManiSkillEnvConfig` and `make_maniskill_env()` for shared ManiSkill
  behavior instead of duplicating wrappers in examples.

## Algorithms, Policies, and Encoders

- Reuse `OffPolicyAlgorithm` when its rollout/replay/update contract fits.
- Prefer subclass method overrides. Add a generic parent hook only when it has a
  no-op or trivially correct default and no algorithm-specific concepts in its
  signature.
- Algorithm-specific fields such as `base_actions` or `mc_returns` belong in the
  subclass. Prefer an overridable class attribute for extra batch keys over parent
  `hasattr` branches.
- Implement feature extractors under `rl_garden/encoders/` as
  `BaseFeaturesExtractor` subclasses and inject them through `policy_kwargs`.
- Never branch on `isinstance(obs_space, spaces.Box/Dict)` in an algorithm or
  policy. Observation space is always `spaces.Dict` (state-only normalizes to
  `Dict({"state": Box})`); resolve encoders through `ObservationEncoderMixin`
  (`rl_garden/algorithms/_observation.py`) — see
  [`.agents/rules/adding-algorithm.md`](adding-algorithm.md) §5.
- Add focused tests for construction, one update step, shapes/devices, and relevant
  edge cases. CPU tests validate compatibility behavior, not the preferred path.

## Device and Replay Invariants

- Preserve the CUDA-first rollout, replay, inference, and update path.
- Do not introduce CPU or NumPy copies in the normal hot path.
- `buffer_device` controls replay storage; samples move to the algorithm device.
- Keep replay layout `(T, N, ...)` and dict observation keys stable unless the
  requested change explicitly modifies that contract.
- Actor/critic encoder sharing is the per-algorithm-class `encoder_sharing`
  attribute (`"shared_critic_grad"` | `"shared"` | `"separate"`), not a fixed
  behavior: `"shared_critic_grad"` (off-policy default, e.g. SAC/CQL/IQL) is
  one encoder, actor updates detach encoder features, critic updates train
  it; `"shared"` (on-policy default, e.g. PPO) is one encoder trained by both
  losses; `"separate"` is two independent encoders, required when
  `obs_groups.actor != obs_groups.critic` or a distinct
  `critic_encoder_config` is given.

## Specialized Defaults

- Never add named, task-specific fields to a backend's `EnvBackendArgs` config
  (e.g. `ManiSkillConfig`) — it is shared across every task the backend runs
  and must stay generic. Customized or task-specific env parameters go one of
  two ways: change them directly in the env's own file (the vendored task
  class's constructor defaults), or pass them in via the backend's generic
  JSON passthrough (e.g. `--maniskill.env-kwargs-json '{"key": value}'`,
  forwarded to `ManiSkillEnvConfig.env_kwargs`, which takes precedence over
  any named field).
