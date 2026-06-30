# Adding a New Environment Backend

This document is the authoritative guide for adding a new simulator or real-robot
backend to rl-garden. Read it before touching `rl_garden/envs/`, `rl_garden/common/env_args.py`,
or any file that calls `register_env_backend`.

## Overview

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
                  └─ MyBackend.make_eval_env(req)
```

**Discovery mechanism:** `discover_env_backends()` (called lazily on first
`make_training_envs` / `make_evaluation_env`) uses `pkgutil.iter_modules` to
import every non-`_`-prefixed module under `rl_garden/envs/backends/`. Each
module's top-level `register_env_backend(...)` call fires as a side-effect of
the import. No explicit import elsewhere is required.

---

## Step 1 — Backend adapter (`rl_garden/envs/backends/my_backend.py`)

Create a new file in `rl_garden/envs/backends/`. The file name becomes the
backend key users pass to `--env_backend`.

```python
"""MyBackend env backend — registered as ``"my_backend"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import EnvBackend, EnvRequest, register_env_backend


class MyBackend(EnvBackend):
    config_field = "my_backend"   # must match the field name on EnvBackendArgs

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.my_backend import MyBackendConfig, make_my_backend_env

        cfg = MyBackendConfig(
            env_id=req.env_id,
            num_envs=req.num_envs,
            obs_mode=req.obs_mode,
            # ... translate every relevant EnvRequest field
            backend_specific_knob=req.backend_config.backend_specific_knob
            if req.backend_config is not None
            else None,
        )
        return make_my_backend_env(cfg)

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.my_backend import MyBackendConfig, make_my_backend_env

        cfg = MyBackendConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs,   # note: eval uses num_eval_envs
            # reconfiguration_freq=1,      # reset between episodes if supported
            record_dir=req.eval_record_dir,
            save_video=req.capture_video,
            video_fps=req.video_fps,
            max_steps_per_video=req.num_eval_steps,
            # ...
        )
        return make_my_backend_env(cfg)


register_env_backend("my_backend", MyBackend)   # ← fires on import
```

**Rules:**
- The `config_field` string must match the attribute name you add to
  `EnvBackendArgs` in Step 3.
- `make_train_env` / `make_eval_env` must return a vectorised env that exposes
  `env.num_envs`, `env.single_observation_space`, `env.single_action_space`,
  and `step` / `reset` returning **GPU torch tensors** (or CPU tensors for
  CPU-backed backends; transfer cost must not enter the hot path invisibly).
- Keep the import of the backend library inside the classmethods — not at module
  level. This preserves lazy loading: importing `rl_garden.envs` must not fail
  when the backend is not installed.

---

## Step 2 — Backend implementation package (`rl_garden/envs/my_backend/`)

Create a sub-package that holds the config dataclass and the env factory. The
maniskill package (`rl_garden/envs/maniskill/`) is the minimal reference; robotwin
(`rl_garden/envs/robotwin/`) is a heavier example with a threading executor.

### `rl_garden/envs/my_backend/__init__.py`

```python
from rl_garden.envs.my_backend.config import MyBackendConfig
from rl_garden.envs.my_backend.env import make_my_backend_env

__all__ = ["MyBackendConfig", "make_my_backend_env"]
```

### `rl_garden/envs/my_backend/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MyBackendConfig:
    env_id: str
    num_envs: int
    obs_mode: str
    control_mode: str
    render_mode: str
    seed: int
    # ... all fields the factory needs
    record_dir: Optional[str] = None
    save_video: bool = False
    video_fps: int = 30
    max_steps_per_video: Optional[int] = None
    # backend-specific knobs:
    backend_specific_knob: Optional[str] = None
```

### `rl_garden/envs/my_backend/env.py`

```python
def make_my_backend_env(cfg: MyBackendConfig):
    import my_simulator  # lazy — do not import at module level

    env = my_simulator.make(cfg.env_id, num_envs=cfg.num_envs, ...)

    # Apply standard wrappers where they fit:
    from rl_garden.envs.wrappers.frame_stack import ImageFrameStackWrapper
    from rl_garden.envs.wrappers.per_camera_rgbd import PerCameraRGBDWrapper
    from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper

    if cfg.frame_stack > 1:
        env = ImageFrameStackWrapper(env, cfg.frame_stack)
    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        env = RewardScaleBiasWrapper(env, cfg.reward_scale, cfg.reward_bias)

    return env
```

---

## Step 3 — Register the config in `EnvBackendArgs`

Edit `rl_garden/common/env_args.py`:

1. Add a `MyBackendConfig` dataclass (backend-level CLI knobs, e.g. task-specific
   flags). Keep its import **lazy inside the field's `default_factory`** or add
   it as a direct import — but guard it so the file can be imported without the
   backend installed.

2. Add the field to `EnvBackendArgs`:

```python
# rl_garden/common/env_args.py

@dataclass
class MyBackendConfig:
    """MyBackend-specific env settings. CLI prefix: ``--my_backend.<field>``"""
    backend_specific_knob: Optional[str] = None
    # add only settings that users should be able to tune via CLI

@dataclass
class EnvBackendArgs:
    env_backend: str = "maniskill"
    maniskill: ManiSkillConfig = field(default_factory=ManiSkillConfig)
    robotwin: RoboTwinConfig = field(default_factory=RoboTwinConfig)
    my_backend: MyBackendConfig = field(default_factory=MyBackendConfig)  # ← add

    def resolve_backend_config(self):
        from rl_garden.envs.backend_registry import resolve_backend_config
        return resolve_backend_config(self.env_backend, self)
```

**Rules:**
- The field name (`my_backend`) must match `MyBackend.config_field`.
- All fields must have defaults so the dataclass can be constructed without
  the backend installed (lazy loading invariant).
- Backend config in `EnvBackendArgs` is for **CLI-tuneable** knobs only.
  Internal implementation details belong in `MyBackendConfig` inside the
  implementation package, not here.

---

## Step 4 — Wrappers (optional)

The following wrappers in `rl_garden/envs/wrappers/` are available for reuse:

| Wrapper | When to use |
|---------|-------------|
| `ImageFrameStackWrapper` | Visual obs with `frame_stack > 1` |
| `PerCameraRGBDWrapper` | Multi-camera envs where each camera feeds a separate encoder |
| `RewardScaleBiasWrapper` | Dense reward normalisation (`r * scale + bias`) |
| `SuccessRewardOverrideWrapper` | Replace terminal reward with a fixed value on success |

Wrappers are applied inside `make_my_backend_env()` — the backend owns its
wrapper stack. Run functions and algorithms are unaware of wrappers.

---

## `EnvRequest` field reference

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

---

## Verification

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
