# Writing Custom IsaacLab Tasks

This guide is for adding a new IsaacLab task to rl-garden (Isaac Lab 3.0). It
covers the `RLGardenDirectRLEnv` scaffold, which is the recommended way to
write new tasks: you declare the scene in a cfg, implement four methods, and
reset is generic.

If you only want to train PPO on a task that already exists in
`isaaclab_tasks` (official IsaacLab tasks, e.g. `Isaac-Cartpole-Direct`,
`Isaac-Ant-Direct` — 3.0 ids have no `-v0` suffix), you don't need any of this:
those run directly through `--env_backend isaaclab --env_id <task-id>`. See
[Training an existing task](#training-an-existing-task) below.

## The four methods you write

`RLGardenDirectRLEnv` (`rl_garden/envs/isaaclab/direct_env.py`) is a subclass
of IsaacLab's `DirectRLEnv` that provides generic implementations of
`_setup_scene`, `_reset_idx`, and `_pre_physics_step` for the common case: one
primary robot (an `Articulation`) plus an optional single camera. You only
implement the task-specific logic:

| Method | What it does |
|---|---|
| `_apply_action(self) -> None` | Send `self.actions` (already scaled by `cfg.action_scale`) to the robot, e.g. `self.robot.set_joint_effort_target_index(target=self.actions, joint_ids=...)`. |
| `_get_observations(self) -> dict` | Return the observation dict. See [Observation key convention](#observation-key-convention). |
| `_get_rewards(self) -> torch.Tensor` | Return a `(num_envs,)` reward tensor. |
| `_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]` | Return `(terminated, time_out)`, each `(num_envs,)` bool tensors. |

## The scene is declared, not built

In Isaac Lab 3.0, `InteractiveScene` builds the whole scene from the scene
cfg: every asset declared as an attribute of `cfg.scene` is spawned,
replicated across `num_envs`, and collision-filtered before `_setup_scene()`
runs. There is no `clone_environments()` step any more.

`RLGardenSceneCfg` (an `InteractiveSceneCfg`) declares the scaffold's shape:

| Attribute | Default |
|---|---|
| `ground` | `/World/ground` ground plane |
| `robot` | `MISSING` — your `ArticulationCfg` |
| `camera` | `None` — set a `CameraCfg` for camera tasks |
| `light` | `/World/Light` dome light |

Per-env prim paths use the `{ENV_REGEX_NS}` macro (e.g.
`"{ENV_REGEX_NS}/Robot"`). `RLGardenDirectRLEnv._setup_scene` only binds
handles: `self.robot = self.scene["robot"]` and `self.camera`
(`None` for state-only tasks). `_reset_idx` restores default joint pos/vel
and the default root pose offset by the env origin, through the
`write_*_to_sim_index` API.

## 3.0 data conventions

- Asset and sensor data (`self.robot.data.joint_pos`, `default_joint_pos`,
  `camera.data.output["rgb"]`, ...) are `ProxyArray` objects. Read them
  through `.torch`, which is a zero-copy torch view of the simulation
  buffer. Without `.torch`, the adapter's observation-space inference sees a
  warp dtype and builds the wrong space.
- Quaternions are `(x, y, z, w)` everywhere (`CameraCfg.OffsetCfg.rot`,
  `InitialStateCfg.rot`, root poses). The identity is
  `(0.0, 0.0, 0.0, 1.0)`. An old `(1, 0, 0, 0)` literal is a 180° rotation
  about X and fails silently (for example, an upside-down camera).

## Minimal example

```python
# rl_garden/envs/isaaclab/tasks/my_task.py
from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg

MY_ROBOT_CFG: ArticulationCfg = ...  # from isaaclab_assets, or your own


@configclass
class MyTaskSceneCfg(RLGardenSceneCfg):
    robot: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class MyTaskEnvCfg(RLGardenDirectEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1.0
    action_space = 1          # dim of the action your _apply_action expects
    observation_space = 4     # dim of your flat state vector (state-only tasks)
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: MyTaskSceneCfg = MyTaskSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)


class MyTaskEnv(RLGardenDirectRLEnv):
    cfg: MyTaskEnvCfg

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(target=self.actions)

    def _get_observations(self) -> dict:
        data = self.robot.data
        return {"state": torch.cat([data.joint_pos.torch, data.joint_vel.torch], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return -torch.abs(self.robot.data.joint_pos.torch).sum(dim=-1)

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

`SimulationCfg(physics=None)` (as above) means Isaac Sim PhysX. To make the
physics backend selectable from the CLI, declare a `PresetCfg`. See
[Physics and renderer presets](#physics-and-renderer-presets).

Then register the module so its `gym.register(...)` call actually runs:

```python
# rl_garden/envs/isaaclab/tasks/__init__.py
from rl_garden.envs.isaaclab.tasks.my_task import MyTaskEnv
__all__ = [..., "MyTaskEnv"]
```

`make_isaaclab_env()` (`rl_garden/envs/isaaclab/env.py`) imports
`rl_garden.envs.isaaclab.tasks` lazily, so adding your import there is
enough. Isaac Lab 3.0 allows importing `isaaclab.*` modules before the
simulation runtime starts, but keep task imports out of package-import time
so `rl_garden.envs` stays importable without Isaac Lab installed.
`make_isaaclab_env()` runs Isaac Lab 3.0's order:

1. `parse_env_cfg(env_id, ..., overrides=[...])` resolves presets.
2. `isaaclab.app.launch_simulation(env_cfg, ...)` boots Kit only if the
   resolved cfg needs it (Isaac Sim PhysX, or a Kit/RTX camera renderer).
   Newton physics with the Newton renderer runs Kit-less.
3. `gym.make(env_id, cfg=env_cfg)`.

Full worked reference: `rl_garden/envs/isaaclab/tasks/cartpole_direct.py`
reimplements IsaacLab 2.x's official Cartpole task end-to-end: action scaling,
reward terms, termination, and reset-time pole-angle randomization via the
`_sample_reset_state` hook (see below).

## Physics and renderer presets

Isaac Lab 3.0 selects backends through `PresetCfg` fields in the task cfg.
A `PresetCfg` lists named alternatives plus a `default`. rl-garden exposes the
two typed selectors:

- `--isaaclab.physics <name>` becomes `physics=<name>`.
- `--isaaclab.renderer <name>` becomes `renderer=<name>`.

Both are passed to `parse_env_cfg(..., overrides=...)`. A name must exist in
the task's presets, or `parse_env_cfg` raises. A `renderer=` selector on a
task without a camera also raises.

rl-garden's own tasks:

- `RlGarden-Cartpole-Direct-v0` uses `CartpoleDirectPhysicsCfg`, which is the
  official cartpole physics presets with `default` set to Isaac Sim PhysX.
  Selectable: `isaacsim_physx` (the default), `newton_mjwarp`,
  `newton_kamino`, `ovphysx`, and `physx` (auto). Only `isaacsim_physx` and
  `newton_mjwarp` have been trained with so far.
- `RlGarden-Cartpole-Direct-Camera-v0` uses the same physics presets. Its
  camera's `renderer_cfg=MultiBackendRendererCfg()` defaults to
  `newton_renderer` (Newton Warp, Kit-less). Selectable: `isaacsim_rtx` (Kit
  RTX), `ovrtx`, and `rtx` (auto). Only `newton_renderer` and `isaacsim_rtx`
  have been run so far.

```python
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.cartpole.cartpole_common import CartpolePhysicsCfg


@configclass
class MyPhysicsCfg(CartpolePhysicsCfg):  # or your own PresetCfg subclass
    default: PhysxCfg = PhysxCfg()


sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=MyPhysicsCfg())
```

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
`.torch` view of joint state (as `cartpole_direct.py` does) rather than
reading `self.robot.data.joint_pos.torch` fresh each time, update that cache
inside `_sample_reset_state` too. See the comment in
`cartpole_direct.py::_sample_reset_state` for why: `_get_observations` runs
after `_reset_idx` within the same `step()` call, so the cache must reflect
the just-written reset state, not the pre-reset one.

## Observation key convention

Unlike native IsaacLab tasks (which return `{"policy": obs}`),
`_get_observations()` here must return rl-garden's own strict cross-backend
key vocabulary (`rl_garden.observations.schema.validate_observation_space`;
no other key names are accepted anywhere in rl-garden):

- `"state"` — flat proprioceptive tensor, shape `(num_envs, D)`.
- `"rgb_<camera_name>"` / `"depth_<camera_name>"` — one key per camera, shape
  `(num_envs, H, W, C)` (you choose `camera_name`; IsaacLab bakes the actual
  camera set/resolution into the task registration rather than reading it
  from `ObservationConfig.rgb`/`.depth`/`.image_size` at runtime, so any
  non-empty name works). Bare `"rgb"`/`"depth"` keys are **not** valid.

This is what lets the resulting env plug directly into rl-garden's existing
`ImageFrameStackWrapper` (`rl_garden/envs/wrappers`) with zero extra code —
no ManiSkill-specific translation layer is needed for IsaacLab tasks.

**Image tensors must be raw pixel values**, e.g.
`self.camera.data.output["rgb"].torch.clone()`, `uint8` in `[0, 255]`. Do
**not** apply the mean-subtraction/`/255` normalization that IsaacLab's own
cartpole camera task does for "better training results." rl-garden's image
encoders (`rl_garden/encoders/base.py:
image_needs_normalization`) decide whether to apply `/255` normalization
based on the declared space's dtype/range, which is inferred from your
tensor's actual dtype. Pre-normalized input is **silently handled wrong**
(double-normalized), not rejected with an error. This is a training-quality
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

Set the `camera` attribute of your scene cfg to a `CameraCfg`.
(`TiledCameraCfg` is a deprecated alias in 3.0, because `Camera` already
renders all envs tiled.) Then read `self.camera.data.output["rgb"].torch`
in `_get_observations()`:

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class MyCameraTaskSceneCfg(MyTaskSceneCfg):
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
        renderer_cfg=MultiBackendRendererCfg(),  # newton_renderer by default
    )


@configclass
class MyCameraTaskEnvCfg(MyTaskEnvCfg):
    observation_space = 1
    scene: MyCameraTaskSceneCfg = MyCameraTaskSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)
```

The `.torch` view of `output["rgb"]` is non-contiguous: it strides into an
RGBA buffer, and the next render overwrites it. `.clone()` it before
returning it as an observation.

**Always set `renderer_cfg` explicitly.** Leaving it at the default
`RendererCfg()` resolves to Kit RTX.

**Camera resolution: use 64×64 or 128×128.** rl-garden's default `PlainConv`
image encoder (`rl_garden/encoders/plain_conv.py`) hardcodes a conv/pool
stride schedule that only produces a valid feature map for those two sizes.
Any other resolution silently builds a mis-sized `Linear` layer and crashes
at the first forward pass (`RuntimeError: mat1 and mat2 shapes cannot be
multiplied`). This backend doesn't consult `--obs.image_size`: the camera's
pixel resolution is baked into `CameraCfg.width`/`.height` in the task
registration, so keep it at one of the two supported sizes.

**⚠️ Kit RTX needs a driver Kit accepts.** On a rejected driver (e.g.
570.133.07, inside Kit's `[570.00, 570.158.01)` rejection range) Kit still
boots but creates no RTX scene renderer and the camera returns all-zero
frames. The adapter checks the first observation and raises `RuntimeError`
("produced only zero frames") instead of training on black images; the
default Newton renderer is unaffected. See
[`isaaclab-camera-stall.md`](isaaclab-camera-stall.md) for the history of
the 2.x stall and the 3.0 measurements. State-only tasks (no camera) are
unaffected.

## Training your new task

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-v0 \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

For a camera task, pass any non-empty camera name to `--obs.rgb`. IsaacLab
ignores the name itself (the real camera set/resolution is baked into the
task registration, see "Camera resolution" above); it only uses it to tell
"some vision requested" apart from state-only:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-Camera-v0 --obs.rgb camera \
  --num_envs 32 --eval_freq 0 \
  --total_timesteps 200000 \
  --isaaclab.sim_device cuda:0
```

Relevant IsaacLab-specific CLI flags (`--isaaclab.<field>`, see
`IsaacLabConfig` in `rl_garden/common/env_args.py`):

- `--isaaclab.headless` (default `True`)
- `--isaaclab.sim_device` (default `cuda:0`)
- `--isaaclab.physics` (default: the task's preset default) — see
  [Physics and renderer presets](#physics-and-renderer-presets)
- `--isaaclab.renderer` (default: the task's preset default, camera tasks only)
- `--isaaclab.env_kwargs_json` — a JSON dict applied as
  `setattr(env_cfg, key, value)` overrides on the parsed cfg, before launch.
  It's an escape hatch for task-specific cfg fields you want to override from
  the CLI without adding a dedicated flag.

`--eval_freq 0` is required. If you leave it out, the run is rejected at
eval-env construction time: the IsaacLab backend does not support a separate
live eval env, because `launch_simulation` hosts at most one Isaac Sim/Kit
app per process.

## Training an existing task

Native `isaaclab_tasks` tasks (state-only) work through the same backend with
zero extra code:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id Isaac-Cartpole-Direct \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

Native 3.0 tasks default to Newton physics (for cartpole, `default =
newton_mjwarp`), so the command above runs Kit-less. Add
`--isaaclab.physics isaacsim_physx` to run them on Isaac Sim PhysX, the 2.x
backend.

This works for both `ManagerBasedRLEnv` and `DirectRLEnv` native tasks
because both use the `"policy"` observation-group key by convention, which
`_IsaacLabVecEnvAdapter` falls back to when it doesn't find rl-garden's own
`"state"` key. Native image-observation tasks are **not** supported. For
example, `Isaac-Cartpole-Camera-Direct` returns a stacked channel-first
`"policy"` image plus a `"critic"` state, which doesn't fit rl-garden's
`Dict` obs convention. Write a scaffold-based task instead if you need
images.

## Overriding scene setup entirely

If your task doesn't fit "one articulation + optional camera" (multiple
objects, non-rigid bodies, a custom scene layout), subclass `RLGardenSceneCfg`
and add the extra assets as attributes. Then override `_setup_scene` (to bind
the extra handles via `self.scene["<name>"]`) and/or `_reset_idx`. These are
ordinary methods on a `DirectRLEnv` subclass, and nothing about the scaffold
prevents this. Start from `RLGardenDirectRLEnv`'s own implementation in
`rl_garden/envs/isaaclab/direct_env.py`.
