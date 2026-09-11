# Adding a New Environment Backend

This document is the authoritative guide for adding a new simulator or
real-robot backend to rl-garden. Read it before touching `rl_garden/envs/`,
`rl_garden/common/env_args.py`, or any file that calls `register_env_backend`.

## 1. Overview

The backend system decouples training code from simulators. A run function fills
a backend-neutral `EnvRequest`; the registry translates it into
simulator-specific calls.

```
run_online(args, make_env_request=..., build_agent=...)
    │
    ├─ make_env_request(args, run_name) → EnvRequest   ← training/online/my_algo.py
    │
    └─ make_training_envs(args.env_backend, req)
           │
           └─ _get_backend("my_backend")               ← auto-discovered on first call
                  │
                  ├─ MyBackend.make_train_env(req)
                  └─ MyBackend.make_eval_env(req)       ← only if req.create_eval_env
```

Offline algorithms never call `make_train_env` — spaces come from the dataset
via `OfflineEnvSpec`, and the backend is only touched for an optional live eval
env through `make_evaluation_env` (see §8).

## 2. Discovery

`discover_env_backends()` (`rl_garden/envs/backend_registry.py`) runs lazily on
the first `make_training_envs` / `make_evaluation_env` call and does two things:

1. `pkgutil.iter_modules` over `rl_garden/envs/backends/` imports every
   non-`_`-prefixed module in that package. Each module's top-level
   `register_env_backend(...)` call fires as an import side effect — no explicit
   import elsewhere is required. **`rl_garden/envs/backends/__init__.py` stays a
   docstring-only file; nothing is imported there.**
2. `importlib.metadata.entry_points(group="rlgarden.env_backends")` loads every
   entry point in that group, letting a backend live entirely outside
   `rl_garden` — the real example is `franka_real`, registered by the separate
   `rlgarden-real-world` repo (see
   [`.agents/rules/repository-map.md`](repository-map.md)) via its own
   `pyproject.toml`:

```toml
[project.entry-points."rlgarden.env_backends"]
franka_real = "rlgarden_real_world.backend_module"
```

`rlgarden_real_world.backend_module` calls `register_env_backend(...)` at import
time, same as an in-tree module does. Use this path only when the backend
depends on hardware or code that must not live in `rl_garden` itself; otherwise
add an in-tree module as in Step 1.

## 3. Step 1 — Backend adapter (`rl_garden/envs/backends/my_backend.py`)

Create a new file in `rl_garden/envs/backends/`. The file name becomes the
backend key users pass to `--env_backend`. `rl_garden/envs/backends/custom.py`
is the reference to copy.

```python
"""MyBackend env backend — registered as ``"my_backend"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import EnvBackend, EnvRequest, register_env_backend


class MyBackend(EnvBackend):
    api_version = 2
    config_field = "my_backend"   # must equal the registry name below

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.my_backend.config import MyBackendConfig

        mb = req.backend_config  # MyBackendConfig or None
        return MyBackendConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=mb.device if mb is not None else "cpu",
            # ... translate every relevant EnvRequest field
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.my_backend import make_my_backend_env

        return make_my_backend_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.my_backend import make_my_backend_env

        return make_my_backend_env(cls.resolve_config(req, is_eval=True))


register_env_backend("my_backend", MyBackend)   # ← fires on import
```

`register_env_backend` enforces API v2 exactly, at discovery time, before any
simulator is touched:
- `cls.__dict__.get("api_version") == 2` — declared directly on the class, not
  inherited from another backend.
- `cls.__dict__.get("config_field")` is a `str` and `cls.config_field == name` —
  the registry key passed to `register_env_backend(...)`.
- `resolve_config`, `make_train_env`, `make_eval_env` are each present in
  `cls.__dict__` — defined directly in the class body. Inheriting one from
  another `EnvBackend` subclass does not satisfy the check; subclassing an
  existing backend is not a shortcut for a new one.

`EnvBackend.config_from_args(args)` (base class, not overridden per backend)
does `getattr(args, cls.config_field)` — what `resolve_backend_config()` on
`EnvBackendArgs` calls to fetch your backend's CLI config.

**Rules:**
- `resolve_config` must be side-effect-free: it builds the concrete train/eval
  config used by `--print-config`/`--dry-run` without initializing a simulator.
- `make_train_env`/`make_eval_env` must return a vectorised env exposing
  `num_envs`, `single_observation_space`, `single_action_space`, and
  `step`/`reset` returning **GPU torch tensors** (or CPU tensors for CPU-backed
  backends; transfer cost must not enter the hot path invisibly).
- Keep the backend library import inside the classmethods, not module level —
  importing `rl_garden.envs` must not fail when the backend isn't installed.

**Escape hatch:** `make_eval_env` may raise `NotImplementedError` when the
backend cannot run two live instances at once — `isaaclab.py` does this because
`AppLauncher` only supports one Isaac Sim per process, telling the user to pass
`--eval_freq 0`. Document the requirement in the module docstring like isaaclab
does.

## 4. Step 2 — Backend implementation package (`rl_garden/envs/my_backend/`)

Create a sub-package holding the config dataclass and the env factory. Three
reference tiers, pick the closest: `rl_garden/envs/maniskill/` (minimal — wraps
an existing GPU simulator with a thin config-translation + wrapper-application
layer), `rl_garden/envs/robotwin/` (heavier — wraps a simulator behind a
threaded executor), or `rl_garden/envs/custom/` for a brand-new environment
authored from scratch. Copy `custom/` whole to start; it's a runnable,
fully-wired template (`--env_backend custom --env_id PointReach-v0`) — see its
`__init__.py` docstring.

```python
# rl_garden/envs/my_backend/config.py
from dataclasses import dataclass

@dataclass
class MyBackendConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cpu"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # ... all fields the factory needs

# rl_garden/envs/my_backend/__init__.py
from rl_garden.envs.my_backend.config import MyBackendConfig
from rl_garden.envs.my_backend.env import make_my_backend_env

__all__ = ["MyBackendConfig", "make_my_backend_env"]
```

### Wrapping a numpy gymnasium env

When the new environment is a standard, single-instance, numpy-based
`gymnasium.Env` (the `custom` package's shape — `point_reach_env.py` authors a
plain `gym.Env`; `env.py` does the rest), the factory follows one fixed
pipeline, `rl_garden/envs/custom/env.py`:

```python
def _make_env_fn():
    def _env_fn():
        # Import here, not at module scope: under AsyncVectorEnv + spawn,
        # workers don't inherit the parent's imports, so gym.register(...)
        # must run fresh in whichever process calls gym.make().
        import gymnasium as gym

        import rl_garden.envs.my_backend.my_task_env  # noqa: F401

        return gym.make("MyTask-v0")

    return _env_fn


def make_my_backend_env(cfg: MyBackendConfig):
    from gymnasium.vector import AutoresetMode, SyncVectorEnv  # or AsyncVectorEnv (process isolation)
    from rl_garden.envs.vector_env import TorchVectorEnvAdapter

    env_fns = [_make_env_fn() for _ in range(cfg.num_envs)]
    vec_env = SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter = TorchVectorEnvAdapter(vec_env, device=cfg.device)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
```

**Env contract** — algorithms depend on this surface regardless of backend,
checked by `tests/test_vector_env_contract.py`:
- `num_envs`, `single_observation_space`, `single_action_space`, and a batched
  `action_space` (off-policy's random-exploration phase reads its `.shape`
  directly, not `single_action_space`)
- `reset(seed=...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, infos)`, all of
  `obs`/`reward`/`terminated`/`truncated` torch tensors (or a dict of torch
  tensors for Dict obs) on the backend's configured device
- `metadata["autoreset_mode"] = AutoresetMode.SAME_STEP`
- `infos["final_observation"]`/`"_final_observation"`/`"final_info"`/`"_final_info"`
  — matching `ManiSkillVectorEnv`'s key set exactly

`TorchVectorEnvAdapter` provides all of this for free once handed a numpy
`gymnasium.vector.VectorEnv`; skip it only if the backend already returns torch
tensors natively (e.g. ManiSkill).

## 5. Step 3 — Register the config in `EnvBackendArgs`

Edit `rl_garden/common/env_args.py`:

```python
# rl_garden/common/env_args.py

@dataclass
class MyBackendConfig:
    """MyBackend-specific env settings. CLI prefix: ``--my-backend.<field>``"""
    device: str = "cpu"
    # JSON-encoded dict forwarded verbatim to the task constructor — escape
    # hatch for task-specific kwargs without adding a named field per task.
    env_kwargs_json: str = "{}"
    # Only for CPU gymnasium-vector backends: "sync" (single process) or
    # "async" (one OS process per env, e.g. for isolated render contexts).
    vectorization: str = "sync"
    backend_specific_knob: Optional[str] = None


@dataclass
class EnvBackendArgs:
    env_backend: str = "maniskill"
    maniskill: ManiSkillConfig = field(default_factory=ManiSkillConfig)
    # ... existing backends (robotwin, minari, d4rl_legacy, mujoco,
    # mujoco_warp, isaaclab, custom, robomimic, ogbench, rlbench, metaworld)
    my_backend: MyBackendConfig = field(default_factory=MyBackendConfig)  # ← add

    def resolve_backend_config(self):
        from rl_garden.envs.backend_registry import resolve_backend_config
        return resolve_backend_config(self.env_backend, self)
```

**Rules:**
- The field name (`my_backend`) must match `MyBackend.config_field`.
- All fields must have defaults so the dataclass can be constructed without the
  backend installed (lazy loading invariant).
- `device` and `env_kwargs_json: str = "{}"` are the standing conventions for a
  device or task-kwargs passthrough (see `ManiSkillConfig`, `OGBenchConfig`,
  `IsaacLabConfig`, etc.); `vectorization: "sync" | "async"` is the convention
  for CPU gymnasium-vector backends (`MujocoConfig`, `OGBenchConfig`,
  `RLBenchConfig`, `MetaWorldConfig`).
- Backend config in `EnvBackendArgs` is for **CLI-tuneable, cross-task** knobs
  only. Task-specific values (one task's fixed object pose, a robot variant)
  don't belong here, even as `Optional` fields defaulting to `None` — put them
  in the implementation-layer config's own field defaults, or behind a generic
  passthrough already on that config (`env_kwargs_json`), never as a new named
  field on the CLI-facing config.
- The CLI accepts both dash and underscore field names for a sub-config
  (`--maniskill.sim-backend` and `--maniskill.sim_backend` both parse).

## 6. Step 4 — Wrappers

**Backend-owned** — applied inside `make_my_backend_env()`, before the registry
gets the env:

| Wrapper | When to use |
|---------|-------------|
| `ImageFrameStackWrapper` | Visual obs with `frame_stack > 1` |
| `PerCameraRGBDWrapper` | Multi-camera envs where each camera feeds a separate encoder |
| `RewardScaleBiasWrapper` | Dense reward normalisation (`r * scale + bias`) on a single-env `gym.Wrapper` chain, pre-vectorization |
| `RewardScaleBiasVectorWrapper` | Same, applied to a `TorchVectorEnvAdapter` output post-vectorization |

**Algorithm-owned** — applied in the run file's `build_*` function, after the
registry returns the env, since they encode algorithm-specific semantics the
backend shouldn't know about:

| Wrapper | When to use |
|---------|-------------|
| `ActionChunkWrapper` | Action-chunking algorithms (e.g. `rl_garden/training/online/dppo.py`'s `build_dppo`) |
| `SkillActionWrapper` | Skill-conditioned action spaces |

Other reusable wrappers under `rl_garden/envs/wrappers/`, by name only:
`RewardClassifierWrapper`, `MultiStageBinaryRewardClassifierWrapper`,
`GAILRewardWrapper`, `FWBWResetFreeWrapper`.

## 7. `EnvRequest` field reference

| Field | Type | Notes |
|-------|------|-------|
| `env_id` | `str` | Environment identifier |
| `num_envs` | `int` | Training parallel envs |
| `obs_mode` | `str` | `"state"` / `"rgb"` / `"rgbd"` |
| `control_mode` | `str` | e.g. `"pd_joint_delta_pos"` |
| `render_mode` | `str` | e.g. `"rgb_array"` |
| `seed` | `int` | |
| `camera_width` / `camera_height` | `Optional[int]` | `None` when `obs_mode == "state"` |
| `include_state` | `bool` | Whether to append proprioceptive state to visual obs |
| `per_camera_rgbd` | `bool` | Keep each camera as separate obs keys |
| `frame_stack` | `int` | 1 = no stacking |
| `reward_scale` / `reward_bias` | `float` | Applied by `RewardScaleBiasWrapper` if non-trivial |
| `num_eval_envs` | `int` | Parallel eval envs |
| `eval_record_dir` | `Optional[str]` | Path for video recording; `None` = no recording |
| `capture_video` | `bool` | |
| `video_fps` | `int` | |
| `num_eval_steps` | `int` | Steps per eval episode |
| `create_eval_env` | `bool` | `False` → `make_training_envs` returns `(train_env, None)` |
| `backend_config` | `Any` | Passed through from `args.resolve_backend_config()` |

## 8. Offline-only path

Offline algorithms (`OfflineRLAlgorithm`) never build a train env from a backend
— observation/action spaces come from the dataset via
`OfflineEnvSpec(observation_space, action_space, num_envs)`
(`rl_garden/algorithms/offline.py`), which has no `reset`/`step`. The backend is
touched only for an optional live eval env, gated on both
(`rl_garden/training/offline/_runner.py`):

```python
if args.env_id is not None and should_create_eval_env(args):
    eval_env = make_evaluation_env(args.env_backend, _eval_env_request(args))
```

`should_create_eval_env(args)` (`rl_garden/envs/backend_registry.py`) is the
shared "was periodic evaluation actually requested" check, used identically by
online, offline, and off2on.

Offline **dataset** loading (not env construction) has its own separate
registry, `rl_garden/buffers/dataset_backend_registry.py`
(`register_dataset_backend`).

## 9. Verification

```bash
# Config inspection (no env created):
python examples/train_online.py sac \
    --env_backend my_backend \
    --my_backend.backend_specific_knob value \
    --print-config 2>/dev/null | python -m json.tool | grep my_backend

# Registry discovery:
python -c "
from rl_garden.envs.backend_registry import discover_env_backends, _REGISTRY
discover_env_backends()
print(sorted(_REGISTRY))
"

# Smoke test (requires the backend to be installed):
python examples/train_online.py sac \
    --env_backend my_backend \
    --env_id MyEnv-v0 \
    --total_timesteps 100 \
    --learning_starts 64 \
    --batch_size 32 \
    --num_envs 2 \
    --num_eval_envs 2
```

Add `tests/test_my_backend_env.py` following `tests/test_custom_env_backend.py`,
whose five tests are the checklist:
`test_point_reach_env_satisfies_gymnasium_api` (gymnasium API conformance for
the single-instance env),
`test_make_custom_env_returns_torch_tensors_on_configured_device`,
`test_backend_registers_with_api_version_2`,
`test_resolve_config_is_side_effect_free_and_splits_train_eval_num_envs`, and
`test_make_train_and_eval_env_construct_working_vectorized_envs`. Keep
`tests/test_training_registry.py::test_help_does_not_import_simulator_backends`
and `::test_backend_discovery_validates_v2_without_importing_simulators` green —
they enforce the lazy-import invariant across every backend.

## 10. Current backends

| Backend | Guide |
|---------|-------|
| `custom` | — (this doc's own template, `rl_garden/envs/custom/`) |
| `d4rl_legacy` | [`docs/guides/d4rl-legacy-expansion.md`](../../docs/guides/d4rl-legacy-expansion.md) |
| `isaaclab` | [`docs/guides/isaaclab-custom-tasks.md`](../../docs/guides/isaaclab-custom-tasks.md) |
| `maniskill` | — |
| `metaworld` | [`docs/guides/metaworld-integration.md`](../../docs/guides/metaworld-integration.md) |
| `minari` | — |
| `mujoco` | — |
| `mujoco_warp` | — |
| `ogbench` | [`docs/guides/ogbench-integration.md`](../../docs/guides/ogbench-integration.md) |
| `rlbench` | [`docs/guides/rlbench-integration.md`](../../docs/guides/rlbench-integration.md) |
| `robomimic` | [`docs/guides/robomimic-integration.md`](../../docs/guides/robomimic-integration.md) |
| `robotwin` | [`docs/guides/robotwin.md`](../../docs/guides/robotwin.md) |
| `franka_real` (external, `rlgarden-real-world` repo, entry-point registered) | — |

