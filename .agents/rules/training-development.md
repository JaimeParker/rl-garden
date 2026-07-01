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
- Keep imports lazy during registry discovery so listing algorithms and
  `--print-config` do not eagerly load optional simulator dependencies.

## Environment Backends

- Add a backend in `rl_garden/envs/backends/<name>.py` by subclassing `EnvBackend`,
  implementing `make_train_env(req)` and `make_eval_env(req)`, and calling
  `register_env_backend("<name>", MyBackend)`.
- Import the backend module from `rl_garden/envs/backends/__init__.py` and add its
  config dataclass to `EnvBackendArgs` in `rl_garden/common/env_args.py`.
- Training run functions access backend-specific settings through
  `EnvRequest.backend_config`; they must not call `make_maniskill_env()` directly.
- Extend `ManiSkillEnvConfig` and `make_maniskill_env()` for shared ManiSkill
  behavior instead of duplicating wrappers in examples.
- Register custom ManiSkill environments lazily through
  `rl_garden.envs.register_custom_envs()`.

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
- Add focused tests for construction, one update step, shapes/devices, and relevant
  edge cases. CPU tests validate compatibility behavior, not the preferred path.

## Device and Replay Invariants

- Preserve the CUDA-first rollout, replay, inference, and update path.
- Do not introduce CPU or NumPy copies in the normal hot path.
- `buffer_device` controls replay storage; samples move to the algorithm device.
- Keep replay layout `(T, N, ...)` and dict observation keys stable unless the
  requested change explicitly modifies that contract.
- RGBD actor and critic share an encoder. Actor updates detach encoder features;
  critic updates train the encoder.

## Specialized Defaults

- Environment- or experiment-specific defaults belong in a dedicated example or
  launcher, not in a generic registry dispatcher.
- Peg defaults belong in `examples/train_sac_rgbd_peg.py` and its launcher.
- `PegInsertionSidePegOnly-v1` exposes `base_camera` and `hand_camera`. The peg
  RGBD entrypoint uses per-camera keys and `image_fusion_mode="per_key"` so
  pretrained three-channel ResNet weights remain compatible.
