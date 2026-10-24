# Writing Custom IsaacLab Tasks

This guide is for adding a new IsaacLab task to rl-garden. It covers the
`RLGardenDirectRLEnv` scaffold, which is the recommended way to write new
tasks: you declare the scene in a cfg, implement four methods, and reset is
generic. One backend (`--env_backend isaaclab`) serves Isaac Lab 3.0 and
2.3.x; see [Supported Isaac Lab versions](#supported-isaac-lab-versions).

If you only want to train PPO on a task that already exists in
`isaaclab_tasks` (official IsaacLab tasks, e.g. `Isaac-Cartpole-Direct` on
3.0, `Isaac-Cartpole-Direct-v0` on 2.x), you don't need any of this: those
run directly through `--env_backend isaaclab --env_id <task-id>`. See
[Training an existing task](#training-an-existing-task) below.

## Supported Isaac Lab versions

| Isaac Lab | Status | Isaac Sim | Python | torch | numpy |
|---|---|---|---|---|---|
| 3.0.x | recommended | 6.2 (or none, Kit-less) | 3.12 | 2.11 | >= 2 |
| 2.3.x | supported | 5.x | 3.11 | >= 2.7 | < 2 |

The two generations cannot share a Python environment. At env construction
the backend detects the installed generation (`rl_garden.envs.isaaclab.detect`):
if the `isaaclab_physx` package is importable it is 3.0, otherwise if
`isaaclab` is importable it is 2.x, otherwise it raises `ImportError`. It then
builds the env with `rl_garden.envs.isaaclab.v3` or `rl_garden.envs.isaaclab.v2`.
The adapter, the zero-frame guard, frame stacking, `IsaacLabEnvConfig` and
the CLI are shared.

3.0-only features: Newton physics, the Newton Warp camera renderer, the
`--isaaclab.physics`/`--isaaclab.renderer` presets, terminal observations
(`--isaaclab.final_obs`, see [Terminal observations](#terminal-observations)),
and Kit-less runs. On 2.x the two preset flags raise `ValueError` ("presets
need Isaac Lab 3.0"), and every
run uses Kit + Isaac Sim PhysX, with cameras rendered by Kit RTX through
`TiledCamera`.

The `RlGarden-*` task ids are the same on both generations. Native ids are
each generation's own (2.x `Isaac-Cartpole-Direct-v0`, 3.0
`Isaac-Cartpole-Direct`).

## Package layout

```
rl_garden/envs/isaaclab/
  env.py, config.py, detect.py   # shared factory/adapter, config, generation detection
  tasks_common/                  # task logic mixins, written once (torch only)
  v3/{app,direct_env,runtime}.py, v3/tasks/   # Isaac Lab 3.0 scaffold + task cfgs
  v2/{app,direct_env,runtime}.py, v2/tasks/   # Isaac Lab 2.3.x scaffold + task cfgs
```

A task is a logic mixin in `tasks_common/` plus one cfg module per
generation (`v3/tasks/<task>.py`, `v2/tasks/<task>.py`). A task that only
needs one generation may skip the other cfg.

## The four methods you write

`RLGardenDirectRLEnv` (`v3/direct_env.py` and `v2/direct_env.py`, same class
name) is a subclass of IsaacLab's `DirectRLEnv` that provides generic
implementations of `_setup_scene`, `_reset_idx`, and `_pre_physics_step` for
the common case: one primary robot (an `Articulation`) plus an optional
single camera. You only implement the task-specific logic:

| Method | What it does |
|---|---|
| `_apply_action(self) -> None` | Send `self.actions` (already scaled by `cfg.action_scale`) to the robot, e.g. `self._apply_joint_effort(self.actions, joint_ids)`. |
| `_get_observations(self) -> dict` | Return the observation dict. See [Observation key convention](#observation-key-convention). |
| `_get_rewards(self) -> torch.Tensor` | Return a `(num_envs,)` reward tensor. |
| `_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]` | Return `(terminated, time_out)`, each `(num_envs,)` bool tensors. |

Write them in a mixin that uses only the scaffold accessors below, so the
same code runs on both generations:

| Accessor | 3.0 implementation | 2.x implementation |
|---|---|---|
| `_joint_pos() -> Tensor` / `_joint_vel() -> Tensor` | `robot.data.joint_pos.torch` / `joint_vel.torch` | `robot.data.joint_pos` / `joint_vel` |
| `_apply_joint_effort(target, joint_ids) -> None` | `robot.set_joint_effort_target_index(target=, joint_ids=)` | `robot.set_joint_effort_target(target, joint_ids=)` |
| `_camera_rgb() -> Tensor` | `camera.data.output["rgb"].torch.clone()` | `camera.data.output["rgb"]` |
| `_sample_reset_state(env_ids, joint_pos, joint_vel)` | reset-randomization hook, see below | same |

`self.robot.find_joints(...)`, `self.actions`, `self.reset_terminated`,
`self.episode_length_buf` and `self.max_episode_length` exist on both
generations and can be used directly. Mixin modules must not import
`isaaclab`.

## The scene is declared, not built

On both generations `InteractiveScene` builds the whole scene from the scene
cfg: every asset declared as an attribute of `cfg.scene` is spawned,
replicated across `num_envs`, and collision-filtered before `_setup_scene()`
runs.

`RLGardenSceneCfg` (an `InteractiveSceneCfg`) declares the scaffold's shape:

| Attribute | Default |
|---|---|
| `ground` | `/World/ground` ground plane |
| `robot` | `MISSING` — your `ArticulationCfg` |
| `camera` | `None` — set a `CameraCfg` (3.0) or `TiledCameraCfg` (2.x) for camera tasks |
| `light` | `/World/Light` dome light |

Per-env prim paths use the `{ENV_REGEX_NS}` macro (e.g.
`"{ENV_REGEX_NS}/Robot"`). `RLGardenDirectRLEnv._setup_scene` only binds
handles: `self.robot = self.scene["robot"]` and `self.camera`
(`None` for state-only tasks). `_reset_idx` restores default joint pos/vel
and the default root pose offset by the env origin, through each
generation's own writer API.

## Version differences you must know

| | Isaac Lab 3.0 | Isaac Lab 2.x |
|---|---|---|
| Asset/sensor data | `ProxyArray`; read through `.torch` (zero-copy view). Without `.torch` the adapter sees a warp dtype and builds the wrong space. | plain `torch.Tensor` |
| Quaternions | `(x, y, z, w)`, identity `(0.0, 0.0, 0.0, 1.0)` | `(w, x, y, z)`, identity `(1.0, 0.0, 0.0, 0.0)` |
| Camera cfg | `CameraCfg` (tiled) + `renderer_cfg` | `TiledCameraCfg` (`CameraCfg` is the slow per-camera path); no `renderer_cfg` |
| Physics cfg | `SimulationCfg(physics=...)`, presets | `SimulationCfg(dt, render_interval)`, always PhysX |
| Launch | `launch_simulation(env_cfg)` after parsing; Kit only if needed | `AppLauncher` before any `isaaclab_tasks` import |
| Camera device | any `--isaaclab.sim_device` (Newton); Kit RTX renders on physical GPU 0 regardless | `cuda:0` only (Isaac Sim 5.1 usdrt refuses other GPUs and hangs); the backend rejects other devices for camera tasks before Kit boots |
| Shutdown | `launch_simulation` context closes at process exit | the backend closes the Kit app explicitly after `env.close()` (camera runs otherwise hang in plugin unload); this ends the process, so nothing after `env.close()` runs |

A quaternion literal from the other generation is not an error: it is a
silent 180° rotation (for example, a camera facing away from the robot).
Always write the literal with a comment naming the order.

## Minimal example

Logic, written once:

```python
# rl_garden/envs/isaaclab/tasks_common/my_task.py
from __future__ import annotations

import torch


class MyTaskLogic:
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actuated_joint_ids, _ = self.robot.find_joints(".*")

    def _apply_action(self) -> None:
        self._apply_joint_effort(self.actions, self._actuated_joint_ids)

    def _get_observations(self) -> dict:
        return {"state": torch.cat([self._joint_pos(), self._joint_vel()], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return -torch.abs(self._joint_pos()).sum(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out
```

3.0 cfg:

```python
# rl_garden/envs/isaaclab/v3/tasks/my_task.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_garden.envs.isaaclab.tasks_common.my_task import MyTaskLogic
from rl_garden.envs.isaaclab.v3.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg

MY_ROBOT_CFG: ArticulationCfg = ...  # from isaaclab_assets, or your own


@configclass
class MyTaskSceneCfg(RLGardenSceneCfg):
    robot: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class MyTaskEnvCfg(RLGardenDirectEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1.0
    # Per-env action Box (bounded; see "Action space bounds" below).
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    # One per-env Box per key _get_observations() returns (see "Observation key convention").
    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)})
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: MyTaskSceneCfg = MyTaskSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)


class MyTaskEnv(MyTaskLogic, RLGardenDirectRLEnv):
    cfg: MyTaskEnvCfg


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

2.x cfg, the same module under `v2/tasks/` with the 2.x scaffold:

```python
# rl_garden/envs/isaaclab/v2/tasks/my_task.py
from rl_garden.envs.isaaclab.v2.direct_env import RLGardenDirectEnvCfg, RLGardenDirectRLEnv, RLGardenSceneCfg

# MyTaskSceneCfg / MyTaskEnvCfg / MyTaskEnv / gym.register(...): same bodies as above.
```

The cfg body is identical for this state-only task; it differs once you add
presets (3.0 only), a camera (`TiledCameraCfg`, wxyz) or any quaternion.
Keep the task id the same.

`SimulationCfg(physics=None)` (as above) means Isaac Sim PhysX on 3.0. To
make the physics backend selectable from the CLI on 3.0, declare a
`PresetCfg`. See [Physics and renderer presets](#physics-and-renderer-presets).

Then register each cfg module so its `gym.register(...)` call actually runs:

```python
# rl_garden/envs/isaaclab/v3/tasks/__init__.py (and v2/tasks/__init__.py)
from rl_garden.envs.isaaclab.v3.tasks.my_task import MyTaskEnv
__all__ = [..., "MyTaskEnv"]
```

`make_isaaclab_env()` (`rl_garden/envs/isaaclab/env.py`) detects the
generation and calls that package's `runtime.build`, which imports the task
package lazily, so `rl_garden.envs` stays importable without Isaac Lab
installed. The 3.0 order (`v3/runtime.py`):

1. `parse_env_cfg(env_id, ..., overrides=[...])` resolves presets.
2. `isaaclab.app.launch_simulation(env_cfg, ...)` boots Kit only if the
   resolved cfg needs it (Isaac Sim PhysX, or a Kit/RTX camera renderer).
   Newton physics with the Newton renderer runs Kit-less.
3. `gym.make(env_id, cfg=env_cfg)`.

The 2.x order (`v2/runtime.py`): `AppLauncher(headless, device,
enable_cameras=<vision requested>)` (after a `torch.compile` warm-up, followed
by removing Kit's bundled-torch `pip_prebundle` dirs from `sys.path`), then
import `isaaclab_tasks` and the task package, then `parse_env_cfg`, then
`gym.make`. 2.x cannot import `isaaclab_tasks` or the task modules before Kit
runs.

Full worked reference: `tasks_common/cartpole.py` (logic) with
`v3/tasks/cartpole_direct.py` and `v2/tasks/cartpole_direct.py` (cfgs)
reimplements IsaacLab 2.x's official Cartpole task end-to-end: action scaling,
reward terms, termination, and reset-time pole-angle randomization via the
`_sample_reset_state` hook (see below).

## Physics and renderer presets

Isaac Lab 3.0 only. Isaac Lab 3.0 selects backends through `PresetCfg`
fields in the task cfg. A `PresetCfg` lists named alternatives plus a
`default`. rl-garden exposes the two typed selectors:

- `--isaaclab.physics <name>` becomes `physics=<name>`.
- `--isaaclab.renderer <name>` becomes `renderer=<name>`.

Both are passed to `parse_env_cfg(..., overrides=...)`. A name must exist in
the task's presets, or `parse_env_cfg` raises. A `renderer=` selector on a
task without a camera also raises. On 2.x either flag raises `ValueError`
before Kit starts.

rl-garden's own tasks (3.0 cfgs):

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
    joint_pos[:, self._some_joint_idx] += torch.rand(..., device=joint_pos.device) * 0.2 - 0.1
    return joint_pos, joint_vel
```

If your `_get_observations()`/`_get_rewards()`/`_get_dones()` read a *cached*
`_joint_pos()`/`_joint_vel()` tensor (as `tasks_common/cartpole.py` does)
rather than calling the accessor fresh each time, update that cache inside
`_sample_reset_state` too. See the comment in
`CartpoleLogic._sample_reset_state` for why: `_get_observations` runs after
`_reset_idx` within the same `step()` call, so the cache must reflect the
just-written reset state, not the pre-reset one.

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

**Image tensors must be raw pixel values**, e.g. `self._camera_rgb()`,
`uint8` in `[0, 255]`. Do **not** apply the mean-subtraction/`/255`
normalization that IsaacLab's own cartpole camera task does for "better
training results." rl-garden's image encoders (`rl_garden/encoders/base.py:
image_needs_normalization`) decide whether to apply `/255` normalization
based on the declared space's dtype/range (your `cfg.observation_space`
entry, or inferred from your tensor's actual dtype). Pre-normalized input
is **silently handled wrong** (double-normalized), not rejected with an
error. This is a training-quality bug, not a crash, so test suites won't
catch it. See `CartpoleCameraLogic` in `tasks_common/cartpole.py` for a
correct example.

The `/255` rule is RGB-only. `"depth_<camera_name>"` keys are passed to the
network unscaled, in whatever unit your task emits (metres, or your own
`[0, 1]` normalization), so emit finite float32 values: clip Isaac Lab's
`inf` for no-hit to a max range.

Declare `cfg.observation_space` as a `gym.spaces.Dict` keyed by exactly the
keys `_get_observations()` returns, with one per-env `Box` each:

```python
observation_space = gym.spaces.Dict(
    {
        "rgb_front": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
        "depth_front": gym.spaces.Box(0.0, 1.0, (64, 64, 1), np.float32),  # your [0, 1] depth range
        "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
    }
)
```

IsaacLab never checks your returned obs against it. rl-garden's adapter
(`_IsaacLabVecEnvAdapter`) does, once, on the first observation: the key set
must match exactly (every returned key declared, nothing extra declared), and
each entry must be a `Box` with the tensor's per-env shape and dtype;
otherwise it raises `ObservationContractError`. It then uses your `Box`es,
bounds included, as the env's observation space (frame stacking broadcasts
them over the frame dim). An int placeholder (e.g. `1`) still passes
`DirectRLEnvCfg.validate()` and falls back to inferring the spaces from the
runtime tensors (`uint8` -> `[0, 255]`, anything else -> unbounded `float32`).

## Action space bounds

Declare `cfg.action_space` as a bounded per-env `gym.spaces.Box`. Isaac Lab
turns an int placeholder (e.g. `action_space = 1`) into
`Box(-inf, inf, (1,))`, and the adapter passes those bounds through as
`single_action_space`. Off-policy algorithms (SAC and the others with a
tanh-squashed actor) rescale actions to the bounds, so infinite bounds give
`nan` actions. Neither Isaac Lab nor the scaffold clips the action before
`_apply_action` (`self.actions = cfg.action_scale * action`), so the bounds
also set the largest command rl-garden sends: PPO clamps its sampled
Gaussian actions to them before stepping. Native `isaaclab_tasks` tasks
declare ints, so they get unbounded spaces. They train with PPO, but not with
off-policy algorithms.

## Adding a camera

Set the `camera` attribute of your scene cfg and read `self._camera_rgb()` in
`_get_observations()`. The cfg differs per generation.

3.0 (`CameraCfg`; `TiledCameraCfg` is a deprecated alias there, because
`Camera` already renders all envs tiled):

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class MyCameraTaskSceneCfg(MyTaskSceneCfg):
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # (x, y, z, w) identity
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
    observation_space = gym.spaces.Dict(
        {
            "rgb_front": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        }
    )
    scene: MyCameraTaskSceneCfg = MyCameraTaskSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)
```

2.x (`TiledCameraCfg`, wxyz, no `renderer_cfg`):

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


@configclass
class MyCameraTaskSceneCfg(MyTaskSceneCfg):
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # (w, x, y, z) identity
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
    )
```

On 2.x do not set `clone_in_fabric=True` on a camera task's scene cfg: it
mis-sizes the camera sensor's per-env buffers.

On 3.0 the `.torch` view of `output["rgb"]` is non-contiguous: it strides
into an RGBA buffer, and the next render overwrites it. The 3.0
`_camera_rgb()` therefore returns a `.clone()`. The 2.x accessor returns
Isaac Lab's tensor as-is.

**On 3.0, always set `renderer_cfg` explicitly.** Leaving it at the default
`RendererCfg()` resolves to Kit RTX.

**Camera resolution: use 64×64 or 128×128.** rl-garden's default `PlainConv`
image encoder (`rl_garden/encoders/plain_conv.py`) hardcodes a conv/pool
stride schedule that only produces a valid feature map for those two sizes.
Any other resolution silently builds a mis-sized `Linear` layer and crashes
at the first forward pass (`RuntimeError: mat1 and mat2 shapes cannot be
multiplied`). This backend doesn't consult `--obs.image_size`: the camera's
pixel resolution is baked into the camera cfg's `width`/`height` in the task
registration, so keep it at one of the two supported sizes.

**⚠️ Kit RTX needs a driver Kit accepts.** On a rejected driver (e.g.
570.133.07, inside Kit's `[570.00, 570.158.01)` rejection range) Kit still
boots but creates no RTX scene renderer and the camera returns all-zero
frames. The adapter checks the first observation and raises `RuntimeError`
("produced only zero frames") instead of training on black images; the
3.0 default Newton renderer is unaffected. 2.x has no non-RTX camera
renderer. See [`isaaclab-camera-stall.md`](isaaclab-camera-stall.md) for the
history of the 2.x stall and the 3.0 measurements. State-only tasks (no
camera) are unaffected.

## Episode metrics

The backend computes each env's episode `return` and `episode_len` itself and
reports them through `infos["final_info"]`, so they show up in the `[train]`
line and as `train/return` / `train/episode_len` without any task code.

Per-env episode metrics come from `self.extras["episode_metrics"]`, a dict of
`(num_envs,)` tensors. Assign it on every step where some env ends (e.g. in
`_get_dones`); only the rows of envs that ended on that step are read:

```python
def _get_dones(self):
    terminated, time_out = ...
    self.extras["episode_metrics"] = {"success": terminated.clone()}
    return terminated, time_out
```

Assign fresh tensors (`.clone()` Isaac Lab's persistent buffers such as
`self.reset_terminated`). `success` appears as `success_at_end` in `[train]`;
other keys are logged as `train/<key>`. A key named `return` or
`episode_len`, or a tensor of another shape, raises `ValueError`.

Scalar aggregates in `self.extras["log"]` (what `ManagerBasedRLEnv` and many
direct tasks already write on reset, e.g. `Episode_Reward/<term>`) are logged
as `env/<key>`. The backend forwards `log` only when the task assigns a new
dict object; Isaac Lab leaves the previous `log` in place on steps where no
env resets, and that unchanged object is treated as stale.

## Terminal observations

Isaac Lab resets a done env inside `step()`, so the observation it returns
for that env already belongs to the next episode. On 3.0 the backend turns on
the env cfg's `compute_final_obs` by default: Isaac Lab then computes the
observation once more before the reset, and the backend reports it as
`infos["final_observation"]` (masked by `infos["_final_observation"]`). PPO
and off-policy algorithms bootstrap truncated transitions from it
(`--bootstrap-at-done`, default `truncated`; terminations bootstrap 0).

- `--isaaclab.final_obs False` turns it off to save the cost: one extra
  observation computation on every step where some env resets. With
  thousands of envs that is nearly every step, so camera tasks can pay up to
  about twice the observation compute.
- 2.x has no terminal observation, so the default is off there and
  `--isaaclab.final_obs True` raises `ValueError` before Kit boots. Without
  it, PPO treats time-outs as terminal (bootstrap value 0) and off-policy
  algorithms bootstrap truncations from the reset observation.
- Set it only with `--isaaclab.final_obs`. A `compute_final_obs` key in
  `--isaaclab.env_kwargs_json` raises `ValueError`.

Two task-side requirements on 3.0:

- `_get_observations()` must return fresh tensors (e.g. `torch.cat(...)`,
  `.clone()`), not persistent buffers written in place. The terminal
  observation is not copied, so a reused buffer would be overwritten by the
  post-reset observation of the same step.
- A custom reset path must keep `compute_final_obs` working. Isaac Lab
  computes the terminal observation in `_reset_envs_from_buffer()` just
  before `_reset_idx()`. Overriding `_reset_envs_from_buffer()` with a
  mask-native reset that returns `None` raises `RuntimeError` while the flag
  is on, and an override that skips the base implementation drops the
  terminal observation.

## Training your new task

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-v0 \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

For a camera task, pass the camera name your task emits to `--obs.rgb`
(`--obs.rgb front` for an `"rgb_front"` key). The camera set/resolution is
baked into the task registration (see "Camera resolution" above), but the
observation contract still checks that the requested `rgb_<name>` key
exists:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-MyTask-Camera-v0 --obs.rgb front \
  --num_envs 32 --eval_freq 0 \
  --total_timesteps 200000 \
  --isaaclab.sim_device cuda:0
```

Relevant IsaacLab-specific CLI flags (`--isaaclab.<field>`, see
`IsaacLabConfig` in `rl_garden/common/env_args.py`):

- `--isaaclab.headless` (default `True`)
- `--isaaclab.sim_device` (default `cuda:0`)
- `--isaaclab.physics` (default: the task's preset default; 3.0 only) — see
  [Physics and renderer presets](#physics-and-renderer-presets)
- `--isaaclab.renderer` (default: the task's preset default, camera tasks
  only; 3.0 only)
- `--isaaclab.final_obs` (default: on for 3.0, off for 2.x; `True` is
  rejected on 2.x) — see [Terminal observations](#terminal-observations)
- `--isaaclab.env_kwargs_json` — a JSON dict applied as
  `setattr(env_cfg, key, value)` overrides on the parsed cfg, before
  `gym.make`. It's an escape hatch for task-specific cfg fields you want to
  override from the CLI without adding a dedicated flag.

`--eval_freq 0` is required. If you leave it out, the run is rejected at
eval-env construction time: the IsaacLab backend does not support a separate
live eval env, because a process hosts at most one Isaac Sim/Kit app.

## Training an existing task

Native `isaaclab_tasks` tasks (state-only) work through the same backend with
zero extra code:

```bash
python examples/train_online.py ppo --env_backend isaaclab \
  --env_id Isaac-Cartpole-Direct \
  --num_envs 4096 --eval_freq 0 --total_timesteps 200000
```

(`Isaac-Cartpole-Direct-v0` on 2.x.) Native 3.0 tasks default to Newton
physics (for cartpole, `default = newton_mjwarp`), so the command above runs
Kit-less on 3.0. Add `--isaaclab.physics isaacsim_physx` to run them on Isaac
Sim PhysX, the 2.x backend.

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
`rl_garden/envs/isaaclab/v3/direct_env.py` (or `v2/direct_env.py`); such an
override is generation-specific, so write it once per generation you
support.
