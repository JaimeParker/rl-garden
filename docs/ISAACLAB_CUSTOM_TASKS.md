# Writing Custom IsaacLab Tasks

This guide is for adding a new IsaacLab task to rl-garden. It covers the
`RLGardenDirectRLEnv` scaffold, which is the recommended way to write new
tasks: you implement four methods, and boilerplate scene setup/reset is
generic.

If you only want to train PPO on a task that already exists in
`isaaclab_tasks` (official IsaacLab tasks, e.g. `Isaac-Cartpole-v0`,
`Isaac-Ant-v0`), you don't need any of this — those run directly through
`--env_backend isaaclab --env_id <task-id>`, see
[Training an existing task](#training-an-existing-task) below.

## The four methods you write

`RLGardenDirectRLEnv` (`rl_garden/envs/isaaclab/direct_env.py`) is a subclass
of IsaacLab's `DirectRLEnv` that provides generic implementations of
`_setup_scene`, `_reset_idx`, and `_pre_physics_step` for the common case: one
primary robot (an `Articulation`) plus an optional single camera. You only
implement the task-specific logic:

| Method | What it does |
|---|---|
| `_apply_action(self) -> None` | Send `self.actions` (already scaled by `cfg.action_scale`) to the robot, e.g. `self.robot.set_joint_effort_target(...)`. |
| `_get_observations(self) -> dict` | Return the observation dict. See [Observation key convention](#observation-key-convention). |
| `_get_rewards(self) -> torch.Tensor` | Return a `(num_envs,)` reward tensor. |
| `_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]` | Return `(terminated, time_out)`, each `(num_envs,)` bool tensors. |

Everything else — spawning the robot, ground plane, light, cloning
environments across `num_envs`, and restoring default joint state on reset —
is handled by the base class.

## Minimal example

```python
# rl_garden/envs/isaaclab/tasks/my_task.py
from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv

MY_ROBOT_CFG: ArticulationCfg = ...  # from isaaclab_assets, or your own


@configclass
class MyTaskEnvCfg(RLGardenDirectEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1.0
    action_space = 1          # dim of the action your _apply_action expects
    observation_space = 4     # dim of your flat state vector (state-only tasks)
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )


class MyTaskEnv(RLGardenDirectRLEnv):
    cfg: MyTaskEnvCfg

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        return {"state": torch.cat([self.robot.data.joint_pos, self.robot.data.joint_vel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return -torch.abs(self.robot.data.joint_pos).sum(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out


gym.register(
    id="RlGarden-MyTask-v0",
    entry_point=f"{__name__}:MyTaskEnv",
    disable_env_checker=True,
    # make_isaaclab_env() resolves the cfg through IsaacLab's own
    # parse_env_cfg()/load_cfg_from_registry(), matching the convention every
    # native isaaclab_tasks registration uses -- always a string, not an
    # instance, and always this exact kwarg name.
    kwargs={"env_cfg_entry_point": f"{__name__}:MyTaskEnvCfg"},
)
```

Then register the module so its `gym.register(...)` call actually runs:

```python
# rl_garden/envs/isaaclab/tasks/__init__.py
from rl_garden.envs.isaaclab.tasks.my_task import MyTaskEnv
__all__ = [..., "MyTaskEnv"]
```

**This import must stay in `rl_garden/envs/isaaclab/tasks/__init__.py`, not
anywhere at package-import time.** IsaacLab requires its Kit application to
be booted (`AppLauncher`) before any `isaaclab.*` subclass — including your
task — can be imported. `make_isaaclab_env()` in `rl_garden/envs/isaaclab/env.py`
already imports `rl_garden.envs.isaaclab.tasks` at the right point (after
`get_or_launch_app()`); you don't need to add anything there, just add your
task's import to that `tasks/__init__.py`.

Full worked reference: `rl_garden/envs/isaaclab/tasks/cartpole_direct.py`
reimplements IsaacLab's official Cartpole task end-to-end (action scaling,
reward terms, termination, reset-time pole-angle randomization via the
`_sample_reset_state` hook — see below).

## Reset-time randomization: `_sample_reset_state`

`_reset_idx` restores default joint pos/vel by default (no randomization). If
your task needs reset-time randomization (e.g. a random initial pose), override
the `_sample_reset_state` hook instead of `_reset_idx` itself:

```python
def _sample_reset_state(self, env_ids, joint_pos, joint_vel):
    joint_pos[:, self._some_joint_idx] += sample_uniform(-0.1, 0.1, ..., joint_pos.device)
    return joint_pos, joint_vel
```

If your `_get_observations()`/`_get_rewards()`/`_get_dones()` read a *cached*
copy of joint state (as `cartpole_direct.py` does, for a small perf win) rather
than calling `self.robot.data.joint_pos` fresh each time, update that cache
inside `_sample_reset_state` too — see the comment in
`cartpole_direct.py::_sample_reset_state` for why (it's about `_get_observations`
running after `_reset_idx` within the same `step()` call, so the cache must
reflect the just-written reset state, not the pre-reset one).

## Observation key convention

Unlike native IsaacLab tasks (which return `{"policy": obs}`),
`_get_observations()` here must return rl-garden's own cross-backend keys:

- `"state"` — flat proprioceptive tensor, shape `(num_envs, D)`.
- `"rgb"` / `"depth"` — single-camera image tensors, shape `(num_envs, H, W, C)`.
- `"rgb_<camera_name>"` / `"depth_<camera_name>"` — for tasks with more than
  one camera (you choose the names; the adapter picks up any key starting
  with `rgb`/`depth`).

This is what lets the resulting env plug directly into rl-garden's existing
`ImageFrameStackWrapper` and `image_keys_from_env` (`rl_garden/common/cli_args.py`)
with zero extra code — no ManiSkill-specific translation layer is needed for
IsaacLab tasks.

**Image tensors must be raw pixel values** — e.g.
`self.camera.data.output["rgb"]` passed through untouched, `uint8` in
`[0, 255]`. Do **not** apply the mean-subtraction/`/255` normalization
IsaacLab's own `cartpole_camera_env.py` example does for "better training
results." rl-garden's image encoders (`rl_garden/encoders/base.py:
image_needs_normalization`) decide whether to apply `/255` normalization
based on the declared space's dtype/range, which is inferred from your
tensor's actual dtype. Pre-normalized input is **silently handled wrong**
(double-normalized), not rejected with an error — this is a training-quality
bug, not a crash, so test suites won't catch it. See
`rl_garden/envs/isaaclab/tasks/cartpole_direct_camera.py` for a correct
example.

`cfg.observation_space` can't express a `{"rgb": ..., "state": ...}` dict
shape — for image tasks just set it to a placeholder int (e.g. `1`).
IsaacLab's `DirectRLEnvCfg.validate()` only checks it's not `MISSING`;
nothing else in IsaacLab or rl-garden reads this field against your actual
returned obs. rl-garden's adapter (`_IsaacLabVecEnvAdapter`) builds its real
`gym.spaces.Dict`/`Box` from the runtime obs tensors instead.

## Adding a camera

Add a `camera_cfg` field to your cfg (type `TiledCameraCfg`); the scaffold's
`_setup_scene()` spawns it automatically if `cfg.camera_cfg is not None`:

```python
from isaaclab.sensors import TiledCameraCfg

@configclass
class MyCameraTaskEnvCfg(MyTaskEnvCfg):
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
    )
```

Then read `self.camera.data.output["rgb"]` in `_get_observations()`.

**Camera resolution: use 64×64 or 128×128.** rl-garden's default `PlainConv`
image encoder (`rl_garden/encoders/plain_conv.py`) hardcodes a conv/pool
stride schedule that only produces a valid feature map for those two sizes
(any other resolution silently builds a mis-sized `Linear` layer and crashes
at the first forward pass — `RuntimeError: mat1 and mat2 shapes cannot be
multiplied`). Pass matching `--camera_width`/`--camera_height` on the training
CLI (see below); it defaults to 64.

**Scene config for camera tasks must not set `clone_in_fabric=True`.** The
state-only default (`InteractiveSceneCfg(..., clone_in_fabric=True)`) mis-sizes
the camera sensor's per-env buffers and crashes with a CUDA `index out of
bounds` assertion during `env.reset()`. IsaacLab's own camera example
(`cartpole_camera_env.py`) omits this flag too — see
`cartpole_direct_camera.py`'s `scene` field for the working config.

**⚠️ Camera observations currently have a known, unresolved reliability
issue on the 6017-nofwd host — training can hang partway through. Read
`docs/ISAACLAB_CAMERA_TRAINING_STALL.md` before relying on this for real
work.** State-only tasks (no `camera_cfg`) are unaffected and fully reliable.

## Training your new task

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-v0 --obs_mode state \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

For a camera task:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-Camera-v0 --obs_mode rgb \
  --camera_width 64 --camera_height 64 --num_envs 32 --eval_freq 0 \
  --total_timesteps 200000 \
  --isaaclab.sim_device cuda:0
```

Relevant IsaacLab-specific CLI flags (`--isaaclab.<field>`, see
`IsaacLabConfig` in `rl_garden/common/env_args.py`):

- `--isaaclab.headless` (default `True`)
- `--isaaclab.sim_device` (default `cuda:0`)
- `--isaaclab.env_kwargs_json` — JSON dict forwarded verbatim as
  `setattr(env_cfg, key, value)` overrides on the parsed cfg (escape hatch for
  task-specific cfg fields you want to override from the CLI without adding a
  dedicated flag).

`--eval_freq 0` is required (or leave it out and it'll be rejected at eval-env
construction time) — the IsaacLab backend does not support a separate live
eval env; `AppLauncher` only supports one Isaac Sim instance per process.

## Training an existing task

Native `isaaclab_tasks` tasks (state-only) work through the same backend with
zero extra code:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id Isaac-Cartpole-Direct-v0 --obs_mode state \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

This works for both `ManagerBasedRLEnv` and `DirectRLEnv` native tasks
because both use the `"policy"` observation-group key by convention, which
`_IsaacLabVecEnvAdapter` falls back to when it doesn't find rl-garden's own
`"state"` key. Native image-observation tasks (e.g.
`Isaac-Cartpole-RGB-Camera-Direct-v0`) are **not** supported — those return a
single un-split `"policy"` image tensor with no separate state key, which
doesn't fit rl-garden's `Dict` obs convention. Write a scaffold-based task
instead if you need images.

## Overriding scene setup entirely

If your task doesn't fit "one articulation + optional camera" (multiple
objects, non-rigid bodies, a custom scene layout), override `_setup_scene`
and/or `_reset_idx` wholesale instead of relying on the scaffold's generic
versions — they're ordinary methods on a `DirectRLEnv` subclass, nothing
about the scaffold prevents this. Look at `RLGardenDirectRLEnv`'s own
implementation in `rl_garden/envs/isaaclab/direct_env.py` as your starting
point to copy from.
