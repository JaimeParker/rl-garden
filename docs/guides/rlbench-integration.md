# RLBench Integration

This guide covers installing the `rlbench` env backend, the design choices
behind it, and how to train rl-garden's imitation-learning algorithms against
RLBench tasks. BC, FlowBC, TD3BC and BCQ train and evaluate on RLBench through
the shared offline path; DiffusionBC and A2ABC read H5 datasets only and build
no eval env; DAgger is online and offers mock experts only.

[RLBench](https://github.com/stepjam/RLBench) is a manipulation benchmark
built on CoppeliaSim + PyRep, with 107 tasks and both state and (5-camera)
vision observations. Unlike the `ogbench`, `minari` and `d4rl_legacy` backends, this one
**does not delegate to `gymnasium.make()`** (nor does `robomimic`) — RLBench's own `rlbench.gym.RLBenchEnv` doesn't match
rl-garden's conventions closely enough to reuse (see "Design" below), so
`rl_garden/envs/rlbench/env.py` builds `rlbench.environment.Environment`/
`TaskEnvironment` directly.

## Install

There is no pip extra for this backend (unlike `ogbench`): RLBench has no
PyPI wheel, and PyRep needs a separately-downloaded CoppeliaSim binary built
against native Qt/OpenGL libraries — this cannot be expressed as a `pip
install -e '.[rlbench]'` line. RLBench must already be importable in whatever
environment runs rl-garden.

```bash
# 1. CoppeliaSim (pinned to v4.1.0, what RLBench/PyRep are built against)
export COPPELIASIM_ROOT=/opt/CoppeliaSim   # pick a location that doesn't
                                            # collide with anything else
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p "$COPPELIASIM_ROOT"
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C "$COPPELIASIM_ROOT" --strip-components 1

# 2. PyRep, then RLBench. Neither is on PyPI (verified: `pip install rlbench`
#    fails with "No matching distribution found" -- there is no such
#    package), so both install straight from source, same as PyRep above.
#    Installing from a local clone keeps the installed package in sync with
#    the task list and dataset_generator.py. Use a plain `pip install .`,
#    never `-e`, so no .egg-info is written into the clone.
pip install git+https://github.com/stepjam/PyRep.git
git clone https://github.com/stepjam/RLBench.git 3rd_party/RLBench
pip install 3rd_party/RLBench   # or: pip install git+https://github.com/stepjam/RLBench.git
```

### Headless rendering

CoppeliaSim needs a display. Two options, in order of preference:

- **`Xvfb`** (software framebuffer, no `sudo`, no host-wide config change):
  `Xvfb :99 -screen 0 1280x1024x24 &`, then `DISPLAY=:99` for every command
  that touches RLBench. Slower / no GPU-accelerated rendering, but isolated
  to whatever process starts it.
- **A real Xorg server** (RLBench's own README "Running Headless" section):
  `sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024` plus
  a `/etc/X11/xorg.conf.d/99-maxclients.conf` drop-in. This is **host-wide**
  and `sudo`-gated — it changes the X config for every user of the machine,
  not just one container. Only do this after confirming `Xvfb` genuinely
  isn't sufficient, and only with explicit sign-off on a shared host.

## Design: why not `rlbench.gym.RLBenchEnv`

RLBench's own gym wrapper (`rlbench/gym.py`) registers
`rlbench/<task_name>-{state,vision}-v0` ids, but two of its choices don't fit
rl-garden's conventions:

- It keeps every low-dim field as its own Dict key (`joint_velocities`,
  `gripper_open`, `gripper_pose`, ...) instead of one flattened `"state"`
  key, which is the single state key rl-garden's observation schema expects.
- It names image keys `left_shoulder_rgb`/`front_rgb`/... (suffix
  `_rgb`/`_depth`) instead of rl-garden's strict `rgb_<cam>`/`depth_<cam>`
  key vocabulary (`rl_garden.observations.schema.validate_observation_space`)
  — keeping RLBench's native names would fail that validation.

So this integration builds the observation itself
(`rl_garden.buffers.rlbench_dataset.build_rlbench_observation`): the live env
and the dataset specs are always a `Dict`: `"state"` plus an
`rgb_<camera>`/`depth_<camera>` key for each camera named in
`--obs.rgb`/`--obs.depth` (`camera` ∈ `{left_shoulder, right_shoulder,
overhead, wrist, front}`). This is the producer mapping referenced
from [Configuration System](configuration.md#h5-dataset-observation-layout)
-- RLBench's own `left_shoulder_rgb`/`front_rgb`/... field names never reach
the H5 file or the live env's observation space; both are built from this
one shared helper instead.

This same helper module is the single source of truth for both the live env
(`rl_garden/envs/rlbench/env.py`) and the offline dataset loader below —
RLBench's live env and its stored demos are built from the same `Observation`
class, and both expose exactly the state/rgb/depth keys requested by
`--obs.*`, so the dataset and eval observation spaces match for the same
flags.

## Action mode

The only action mode this integration wires up: **`JointVelocity` arm +
`Discrete` gripper**, matching every RLBench README example verbatim. This
isn't yet CLI-selectable — a different action mode would also need a
different action-label derivation for the offline loader (see below), not
implemented here.

`MoveArmThenGripper` (the composable action mode class this combination
uses) does **not** implement `action_bounds()` — only RLBench's own preset
action modes (e.g. `JointPositionActionMode`) do. Found via a real install,
not assumed: `rlbench.gym.RLBenchEnv` calls `action_mode.action_bounds()`
unconditionally, which would raise `NotImplementedError` for this exact
combination too. This integration builds the `Box` bounds itself instead —
±1.0 rad/s for each `JointVelocity` arm dim (matching `Environment`'s own
`arm_max_velocity=1.0` default) and `[0, 1]` for the single `Discrete`
gripper dim (its own documented open/closed contract) — see
`rl_garden/envs/rlbench/env.py`.

## Usage

```bash
python examples/train_online.py rlpd \
  --env_backend rlbench --env_id reach_target \
  --dataset_backend rlbench --offline_dataset /data/rlbench_demos/reach_target \
  --num_envs 4 --num_eval_envs 2 \
  --total_timesteps 100000 --learning_starts 1000 --batch_size 256
```

Vision (`--obs.rgb <camera>`), `async` vectorization (each instance owns its
own CoppeliaSim renderer/GL context, same reasoning the `mujoco`/`ogbench`
backends already document for their own visual variants). `flow_bc` (like
`bc`) resolves its encoder generically from the `Dict` observation space's
schema (`ObservationEncoderMixin`, `rl_garden/algorithms/observation_encoders.py`),
so no RLBench-specific wiring is needed there either:

```bash
python examples/pretrain_offline.py flow_bc \
  --env_backend rlbench --env_id reach_target --obs.rgb left_shoulder \
  --rlbench.vectorization async \
  --dataset_backend rlbench --offline_dataset /data/rlbench_demos/reach_target
```

BC/FlowBC/TD3BC/BCQ (and every other algorithm in
`rl_garden/training/offline/` that calls `run_offline`) go through the
shared lifecycle (`rl_garden/training/offline/runner.py`), which already
builds an eval env from `--env_id`/`--env_backend` whenever one is requested
— RLBench eval needs no extra wiring beyond registering this backend.
`diffusion_bc` and `a2a_bc` are standalone scripts hardcoded to the H5
dataset format (never pluggable by `--dataset_backend`) and never build an
eval env for *any* backend, so RLBench demos aren't consumable by them
without a separate H5 conversion step.

DAgger (`rl_garden/training/online/entrypoints/dagger.py`) builds its training
env through the standard `make_training_envs`/`EnvBackend` path but creates no
eval env and offers mock experts only, so it does not use RLBench demos.

### Key config fields (`--rlbench.<field>`)

- `device`: device for the online vector env's torch tensors.
- `headless`: `True` by default.
- `env_kwargs_json`: JSON-encoded dict forwarded verbatim to
  `rlbench.environment.Environment` (`robot_setup`, `shaped_rewards`,
  `static_positions`, `arm_max_velocity`, ...). Rarely needed.
- `vectorization`: `"sync"` (default) or `"async"` (recommended once
  `--obs.rgb`/`--obs.depth` is set).

Camera selection and resolution are the shared `--obs.*` surface, not
RLBench-specific fields: `--obs.rgb`/`--obs.depth` name any of the 5 cameras
`{left_shoulder, right_shoulder, overhead, wrist, front}` (rgb+depth each;
never mask/point_cloud) — pass none to leave every camera off (RLBench's own
`ObservationConfig` default enables all 5; trimming here is a cost knob, not
a correctness one). `--obs.image_size` sets the shared per-camera render size
(default `(128, 128)`, RLBench's own default, when unset).

## Dataset format and the action-derivation convention

`--offline_dataset` is the **task's own demo directory** —
`<dataset_root>/<task_name>` in RLBench's own on-disk layout
(`<dataset_root>/<task_name>/variation0/episodes/episode<M>/...`; only
variation 0 is loaded), e.g.
`/data/rlbench_demos/reach_target`. The loader
(`rl_garden.buffers.rlbench_dataset.load_rlbench_dataset_to_replay_buffer`)
splits this back into `dataset_root`/`task_name` itself
(`path.parent`/`path.name`) — kept as one string, not two separate
arguments, so this loader's call signature matches every other loader's
(`load_offline_dataset`/`PriorDataReplayMixin.load_offline_replay_buffer`
always call with a single positional path).

RLBench demos (`rlbench.utils.get_stored_demos`) are pure file I/O — no
PyRep/CoppeliaSim launch needed to read them. **Importing `rlbench` at all
still requires a working `pyrep`/CoppeliaSim install**, though
(`rlbench/__init__.py` unconditionally imports `pyrep` transitively) — there
is no lighter "dataset-only" install path.

Demos carry no explicit reward/action/success field on disk — they're
`List[Observation]`. Two derivations follow RLBench's own README example
directly:

- **Action** for transition `i` (`obs=demo[i] -> next_obs=demo[i+1]`) comes
  from `demo[i]` itself:
  `concat(demo[i].joint_velocities, round(demo[i].gripper_open))` — RLBench's
  own imitation-learning example does the same single-observation derivation
  (`ground_truth_actions = [obs.joint_velocities for obs in batch]`).
- **Reward/done** are sparse and set only at each demo's last step
  (`reward=1.0`, `done=True`; `0.0`/`False` elsewhere) — every
  successfully-collected demo is assumed to end in task success (RLBench's
  own collection retries any failed attempt until it gets a clean run).

`--rlbench.live_demos`-equivalent: `load_rlbench_dataset_to_replay_buffer`'s
`live_demos=True` kwarg drives RLBench's own motion-planning oracle in real
time (`task.get_demos(..., live_demos=True)`) instead of reading a stored
dataset — no `dataset_root` needed in that case (only `task_name`, still
taken from the same `path` argument), but each demo is collected live and
this is slow. Use a small `num_traj` when exercising this path.

## Tests

```bash
pytest -q \
  tests/test_rlbench_dataset.py \
  tests/test_rlbench_env.py \
  tests/test_prior_data_replay.py
```

These use monkeypatched fakes for the entire `rlbench` package tree (no
network access, no `pyrep`/CoppeliaSim install required).

## Known issues and fixes

This backend was installed from scratch and exercised end to end (real
CoppeliaSim + PyRep + RLBench, `reach_target` task, both state-only and
`--obs.rgb`/`--obs.depth` vision, both stored and live-generated demos, and a
full `bc` training run with a live eval env) — the issues below are what
actually surfaced, with their fixes, not speculation:

- **`import rlbench` fails with `ModuleNotFoundError: No module named
  'gymnasium'`** even though you never asked for the `[gym]` extra:
  `rlbench/__init__.py` unconditionally imports `gymnasium` at module scope,
  a real upstream packaging gap (`setup.py` only declares it as an optional
  extra). Fix: install `gymnasium` unconditionally, it's required regardless.
- **`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`** under
  `Xvfb`: CoppeliaSim's bundled Qt needs several system libraries this
  guide's headless setup doesn't otherwise pull in —
  `libxkbcommon-x11-0` and a handful of `libxcb-*` runtime libraries, on top
  of `xvfb` itself. Fix: `apt-get install -y xvfb libxkbcommon-x11-0
  libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0
  libxcb-xkb1 libgl1 libglu1-mesa`.
- **`NotImplementedError: You must define your own action bounds.`** at env
  construction: see "Action mode" above — already fixed in this
  integration's own code (`rl_garden/envs/rlbench/env.py` builds the `Box`
  bounds itself rather than calling `action_mode.action_bounds()`), listed
  here only so the traceback is recognizable if it resurfaces from a
  different code path.

Also confirmed: the stored-demo path (`load_rlbench_dataset_to_replay_buffer`
without `live_demos=True`, and `infer_specs_from_rlbench`) needs no
`DISPLAY`/`Xvfb` at all, matching the "pure file I/O" claim above — only
`live_demos=True` and live env construction need a working display.

## Current limits

- Only one action mode is wired up (`JointVelocity` + `Discrete` gripper).
- No `seed()` forwarding beyond `np.random.seed()` — RLBench's own
  `rlbench.gym.RLBenchEnv` has the same limitation.
- `diffusion_bc` and `a2a_bc` build no eval env for any backend.
- Real verification so far covers one task (`reach_target`) with
  `SyncVectorEnv` only — an `AsyncVectorEnv`/`vectorization=async` smoke
  test and every other task family are architecturally identical (same
  `Environment`/`TaskEnvironment` construction regardless of task) but
  individually untested.
