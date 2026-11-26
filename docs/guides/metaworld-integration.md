# Meta-World Integration

This guide covers installing the `metaworld` env backend and using both
dataset paths (official Minari expert demos, and live scripted-policy demo
generation) with rl-garden's training/eval stack.

[Meta-World](https://github.com/Farama-Foundation/Metaworld) is a
single-agent MuJoCo manipulation benchmark: 50 distinct tasks (`MT1`) plus
the `MT10`/`MT50` multi-task benchmarks. State observations are the
default; single-task env ids also support an added rgb+depth camera view
(see "Vision" below) — Meta-World has no first-class pixel-obs env id of
its own, so this integration renders one itself.

## Install

Unlike RLBench, Meta-World is a real, current PyPI package:

```bash
pip install -e '.[metaworld,minari]'
```

The `minari` extra is needed for the recommended dataset path below; it can be
left out if you only use online training or live demo generation.

## Usage

### Single task

```bash
python examples/pretrain_offline.py bc \
  --env_backend metaworld --env_id reach-v3 \
  --dataset_backend minari --offline_dataset metaworld/reach/expert-v0 \
  --num_eval_envs 2
```

`--env_id` is any of `metaworld.MT1.ENV_NAMES` (e.g. `reach-v3`,
`pick-place-v3`, `door-open-v3`, ...). Meta-World's own
`RandomTaskSelectWrapper` randomizes the goal/object pose on every reset
automatically — no extra wiring needed.

### MT10 / MT50 multi-task

```bash
python examples/train_online.py rlpd \
  --env_backend metaworld --env_id MT10 \
  --total_timesteps 1000000 --learning_starts 5000
```

`--num_envs`/`--num_eval_envs` are **ignored** for `MT10`/`MT50` — Meta-World
fixes the vectorized env count at 10/50 (one sub-env per task) and silently
drops any other value upstream; this backend doesn't try to override that.
`--metaworld.use_one_hot` (default `True`) appends a one-hot task id to each
sub-env's observation, the standard multi-task-RL convention for a single
shared policy to distinguish tasks.

### Key config fields (`--metaworld.<field>`)

- `device`: device for the online vector env's torch tensors.
- `vectorization`: `"sync"` (default) or `"async"` (one OS process per env).
- `use_one_hot`: only consulted when `--env_id` is `MT10`/`MT50`.

Observations are `Dict` spaces, `{"state": ...}` for state-only envs (float32),
with image keys added in vision mode.

Camera selection and resolution are the shared `--obs.*` surface, not a
Meta-World-specific field — see below.

### Vision (rgb + depth)

```bash
python examples/train_online.py rlpd \
  --env_backend metaworld --env_id reach-v3 \
  --obs.rgb corner2 --obs.depth corner2 --obs.image_size 84 84 \
  --num_envs 4 --num_eval_envs 2
```

Adds a single fixed camera's rgb+depth pair on top of the state vector —
the `Dict` observation gains `rgb_<camera>`/`depth_<camera>` keys next to
`state` (rl-garden's standard vision convention, same as RLBench's vision mode).
`--obs.rgb`/`--obs.depth` name any of the 6 cameras every Meta-World v3 task
scene defines: `corner`, `corner2` (the common choice in Meta-World vision
literature, e.g. DrM/DrQ-v2/TD-MPC2), `corner3`, `corner4`, `behindGripper`,
`gripperPOV`. There is no default camera: pass `--obs.rgb`/`--obs.depth`
explicitly. `--obs.image_size` defaults to `84 84`, the common Meta-World
vision size. `--obs.frame_stack N` stacks the last `N` image frames; it
requires at least one camera. The default `plain_conv` encoder's `flatten` pooling
sizes its projection layer from the image; `--encoder.plain_conv_pooling gap`
is the lighter choice at large images.

**Single-task env ids only** — `--env_id MT10`/`MT50` with `--obs.rgb`/
`--obs.depth` set raises a clear `ValueError`: those two env ids build their
10/50 sub-envs internally through `gym.make_vec`, with no per-sub-env
construction hook this backend can attach a camera renderer to.

**Depth is raw MuJoCo NDC, not linear/metric depth** — `mujoco
.mjr_readPixels`'s buffer, unmodified (same caveat
`rl_garden/envs/mujoco/custom_mujoco_env.py`'s own manual-renderer pattern
documents, and the one this integration's `_MetaWorldVisionWrapper` is
built the same way as). If a downstream use needs linear depth, apply the
znear/zfar conversion yourself.

Image observations are `rgb_<camera>` of shape `(N, H, W, 3)` uint8 and
`depth_<camera>` of shape `(N, H, W, 1)` float32. `reach-v3`'s default episode
length is 500 steps (`env.unwrapped.max_path_length`) with
`terminate_on_success` off by default for the live env, so `--num_eval_steps`
must exceed 500 for eval to complete even one episode (it counts vector steps,
so every eval env takes that many); otherwise `return`/`success_at_end` report
`nan`. This holds with or without `--obs.rgb`/`--obs.depth`.

**Online/live-eval only — no offline dataset path supports vision yet**:
both the official `metaworld/<task>/expert-v0` Minari datasets and this
integration's own `metaworld` live-demo backend are state-only
(`Dict({"state": Box(39,)})`). Use `--obs.rgb`/`--obs.depth` with an online
algorithm such as `rlpd`, or the live eval env only, not offline pretraining.

## Dataset: two independent paths

### Official expert demos (recommended, zero new code)

Farama publishes `metaworld/<task-name>/expert-v0` Minari datasets for
essentially all 50 tasks (verified live via `minari.list_remote_datasets()`):
`observation_space=Box(39,) float64`, `action_space=Box(4,) float32` — the
same shape as the live v3 env. rl-garden presents the dataset state as
`Dict({"state": Box(39,) float32})`, the same observation space the live env
declares. This already works today through the existing `minari`
dataset backend, no Meta-World-specific loader involved:

```bash
--dataset_backend minari --offline_dataset metaworld/reach/expert-v0
```

`<task-name>` drops the `-v3` suffix Meta-World's own env ids use (e.g.
`reach-v3` env id ↔ `metaworld/reach/expert-v0` dataset id).

### Live scripted-policy demo generation

Meta-World ships an expert scripted policy for every task
(`metaworld.policies.ENV_POLICY_MAP`). Use this when you need fresh
trajectories, or task coverage `expert-v0` doesn't have:

```bash
--dataset_backend metaworld --offline_dataset reach-v3 --offline_num_traj 50
```

`--offline_dataset` here is the **task name** (e.g. `reach-v3`), not a
filesystem path — Meta-World has no on-disk demo format of its own, so this
backend is always live (unlike RLBench, which supports both a stored-file
path and a live path).

Meta-World's own scripted-policy test suite
(`tests/metaworld/envs/mujoco/sawyer_xyz/test_scripted_policies.py`) only
expects an ~80% success rate across all 50 tasks, not 100%, so
`load_metaworld_dataset_to_replay_buffer` retries failed episodes and keeps
only successful ones — every collected demo is assumed to end in task
success, the same convention `rlbench_dataset.py` uses for RLBench's live
path: reward/done are sparse and set only at each demo's last step
(`reward=1.0`, `done=True`; `0.0`/`False` elsewhere).

## Tests

```bash
pytest -q \
  tests/test_metaworld_env.py \
  tests/test_metaworld_dataset.py
```

These monkeypatch `gymnasium.make`/`gymnasium.make_vec` and inject a fake
`metaworld` module into `sys.modules` — no `metaworld` package install
is required, but MuJoCo (with `gymnasium`'s MuJoCo renderer) must be importable.

## Known limitations

- `success_at_end` is unavailable for `MT10`/`MT50`: Meta-World builds their
  sub-envs internally with no per-sub-env hook to attach the episode-metrics
  wrapper to, so only `return`-based eval metrics are reported there. Single
  task env ids report `success_at_end`.
- Live scripted-policy demo generation keeps only demos that reach task
  success and retries failed ones. Meta-World's scripted policies do not
  succeed on every episode (its own test suite expects about 80% across the 50
  tasks), so harder tasks need more attempts.
