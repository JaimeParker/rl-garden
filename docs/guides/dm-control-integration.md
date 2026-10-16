# dm_control Integration

This guide covers installing and using the `dm_control` env backend. It
complements [`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md).

[dm_control](https://github.com/google-deepmind/dm_control) is DeepMind's
MuJoCo continuous-control suite (cheetah, walker, quadruped, ...). The backend
wraps one `dm_control.suite` task per env, including NCRL's 11 custom tasks
(`rl_garden/envs/dm_control/custom_tasks/`, copied from NCRL under Apache-2.0),
and supports flat state observations, 64x64 pixel observations, or both.

## Install

```bash
pip install -e '.[dm_control]'   # dm_control>=1.0.15, mujoco>=3.0
```

`dm_control` is only imported when an env is built, so `rl_garden` imports,
backend discovery and `--print-config` work without it.

### Rendering (`MUJOCO_GL`)

Pixel observations need an OpenGL backend. On a headless server set it before
starting Python:

```bash
export MUJOCO_GL=egl      # GPU; use osmesa for CPU-only machines
```

The backend never sets `MUJOCO_GL` itself. Requesting a camera with neither
`MUJOCO_GL` nor `DISPLAY` set raises an error with this hint; a failing render
is re-raised with the same hint. State-only runs don't render (dm_control may
still print a harmless GLFW `DISPLAY` warning at import).

## Usage

```bash
# State observations
python examples/train_online.py sac \
  --env_backend dm_control --env_id walker-walk \
  --num_envs 2 --num_eval_envs 1 --num_eval_steps 500

# Pixels (+ state), NCRL custom task, hard variant
MUJOCO_GL=egl python examples/train_online.py sac \
  --env_backend dm_control --env_id cheetah-run-backwards --dm_control.hard \
  --obs.rgb default --obs.image_size 64 64
```

### Task names (`--env_id`)

`--env_id` is `<domain>-<task>`, NCRL's TASK_SET naming: the first dash
splits domain from task, remaining dashes become underscores
(`walker-run-backwards` -> `suite.load("walker", "run_backwards")`). An NCRL
`dmc-` prefix is accepted; `cup`/`point` alias `ball_in_cup`/`point_mass`.
Any task in `dm_control.suite` works. Only the listed NCRL tasks ending in
`-hard` (`walker-walk-hard`, `acrobot-swingup-hard`) select the hard reward
variant through the suffix; every other `*-hard` name is the real suite task
(`reacher-hard` -> `reacher/hard`, `finger-turn-hard` -> `finger/turn_hard`,
TD-MPC2 mt30) and builds without the wrapper. Add `--dm_control.hard` to wrap
any task. NCRL's 22-task list is
`rl_garden.envs.dm_control.tasks.NCRL_DMC_TASKS`:

| task_id | action_dim | state_dim | NCRL custom task |
|---|---|---|---|
| `cartpole-balance` | 1 | 5 | no |
| `acrobot-swingup`, `acrobot-swingup-sparse`, `acrobot-swingup-hard` | 1 | 6 | no |
| `cheetah-run` | 6 | 17 | no |
| `cheetah-run-backwards`, `cheetah-jump`, `cheetah-run-back`, `cheetah-run-front` | 6 | 17 | yes |
| `walker-stand`, `walker-walk`, `walker-walk-hard`, `walker-run` | 6 | 24 | no |
| `walker-walk-backwards`, `walker-run-backwards`, `walker-backflip` | 6 | 24 | yes |
| `quadruped-walk`, `quadruped-run` | 12 | 78 | no |
| `quadruped-stand`, `quadruped-jump`, `quadruped-roll`, `quadruped-roll-fast` | 12 | 78 | yes |

NCRL's custom hopper tasks (`hopper-hop-backwards`, `hopper-flip`,
`hopper-flip-backwards`) are registered too but are not in NCRL's task list.

### Episode and action semantics

- Actions are `Box(-1, 1)` and are rescaled linearly to the task's action spec.
- Each policy step repeats the action for `--dm_control.action_repeat` physics
  frames (default 2) and sums the rewards.
- Episodes are 1000 physics frames, i.e. `1000 / action_repeat` policy steps
  (500 by default), ended with `truncated=True`. The suite tasks never
  terminate, so `terminated` stays `False`. Set `--num_eval_steps` to at least
  one episode (500) so evaluation episodes finish.
- Hard variant (NCRL `-hard` task names or `--dm_control.hard`): NCRL's
  `HardTaskWrapper`, applied per physics frame inside the action repeat. Each
  frame's reward is zeroed below 0.2 (cheetah) / 0.6 (all other domains), then
  `0.1 * ||a||` is subtracted.

### Observations

| Request | Key | Space |
|---|---|---|
| `--obs.state` (default) | `state` | float32 `(state_dim,)`: dm_control's observation dict flattened in its fixed key order (`flat_observation=True`, same as NCRL) |
| `--obs.rgb default` | `rgb_default` | uint8 `(H, W, 3)`, HWC like every other rl-garden pixel backend (encoders permute to NCHW); `H, W` from `--obs.image_size`, default 64x64 |

`default` is the only camera name. It renders MuJoCo camera
`--dm_control.camera_id`, which defaults to 2 for quadruped and 0 for every
other domain, as in NCRL. Other camera names, `--obs.depth`,
`--obs.extra_state`, and `--obs.frame_stack > 1` without a camera raise
`ObservationContractError`. With a camera, `--obs.frame_stack` stacks frames via
`ImageFrameStackWrapper`.

### Key config fields (`--dm_control.<field>`)

- `device`: device for the vector env's torch tensors (default `cpu`).
- `vectorization`: `sync` (default) or `async` (one process per env, each with
  its own GL context).
- `action_repeat`: physics frames per policy step (default 2).
- `camera_id`: MuJoCo camera index for `rgb_default` (default per domain, see above).
- `hard`: enable the hard variant without the `-hard` suffix.

## Tests

```bash
MUJOCO_GL=egl pytest tests/test_dm_control_backend.py
```

The registry, config, and hard-reward tests run without dm_control. Tests that
build a real env skip when it is not installed.
