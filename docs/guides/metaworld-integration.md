# Meta-World Integration

This guide covers installing the `metaworld` env backend and using both
dataset paths (official Minari expert demos, and live scripted-policy demo
generation) with rl-garden's training/eval stack. It complements
[`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md).

[Meta-World](https://github.com/Farama-Foundation/Metaworld) is a
single-agent MuJoCo manipulation benchmark: 50 distinct tasks (`MT1`) plus
the `MT10`/`MT50` multi-task benchmarks. State observations are the
default; single-task env ids also support an added rgb+depth camera view
(see "Vision" below) — Meta-World has no first-class pixel-obs env id of
its own, so this integration renders one itself.

## Install

Unlike RLBench, Meta-World is a real, current PyPI package:

```bash
pip install -e '.[metaworld]'
```

## Usage

### Single task

```bash
python examples/pretrain_offline.py bc \
  --env_backend metaworld --env_id reach-v3 \
  --dataset_backend minari --offline_dataset metaworld/reach/expert-v0 \
  --num_envs 4 --num_eval_envs 2
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
- `camera`/`image_size`: only consulted when `--obs_mode rgb`, see below.

### Vision (rgb + depth)

```bash
python examples/train_online.py rlpd \
  --env_backend metaworld --env_id reach-v3 --obs_mode rgb \
  --metaworld.camera corner2 --metaworld.image_size 84 84 \
  --num_envs 4 --num_eval_envs 2
```

Adds a single fixed camera's rgb+depth pair on top of the state vector —
observation becomes a `Dict` with `state`/`rgb_<camera>`/`depth_<camera>`
keys (rl-garden's standard vision convention, same as RLBench's `rgb` mode).
`--metaworld.camera` is any of the 6 cameras every Meta-World v3 task scene
defines: `corner`, `corner2` (default — the common choice in Meta-World
vision literature, e.g. DrM/DrQ-v2/TD-MPC2), `corner3`, `corner4`,
`behindGripper`, `gripperPOV`. Default image size is `84x84` (the common
Meta-World-vision default, not rl-garden's other backends' `128x128`).

**Single-task env ids only** — `--env_id MT10`/`MT50` with `--obs_mode rgb`
raises a clear `ValueError`: those two env ids build their 10/50 sub-envs
internally through `gym.make_vec`, with no per-sub-env construction hook
this backend can attach a camera renderer to.

**Depth is raw MuJoCo NDC, not linear/metric depth** — `mujoco
.mjr_readPixels`'s buffer, unmodified (same caveat
`rl_garden/envs/mujoco/custom_mujoco_env.py`'s own manual-renderer pattern
documents, and the one this integration's `_MetaWorldVisionWrapper` is
built the same way as). If a downstream use needs linear depth, apply the
znear/zfar conversion yourself.

Verified for real on 6017 (`reach-v3`, `corner2`, 84×84): `rgb_corner2`
(`(N, 84, 84, 3)` uint8, real scene content — nonzero per-image pixel std,
not a blank/black frame) and `depth_corner2` (`(N, 84, 84, 1)` float32,
values in roughly `[0.98, 1.0]`, matching the documented nonlinear-NDC
range) both come back correctly, `--env_id MT10 --obs_mode rgb` raises the
`ValueError` above for real, and a short `rlpd` run trains + evaluates
end to end (image-key discovery, encoder construction, and
`success_at_end` all work under vision the same as state-only). Note when
sizing a quick smoke test: `reach-v3`'s default episode length is 500 steps
(`env.unwrapped.max_path_length`) with `terminate_on_success` off by
default for the live env — `--num_eval_steps` needs to comfortably exceed
that (across all eval envs) for eval to complete even one episode, or
`return`/`success_at_end` report `nan` (0 episodes completed) — this is a
pre-existing characteristic of Meta-World's default episode length, not a
vision-specific issue (reproduces identically in `--obs_mode state`).

**Online/live-eval only — no offline dataset path supports vision yet**:
both the official `metaworld/<task>/expert-v0` Minari datasets and this
integration's own `metaworld` live-demo backend are state-only (`Box(39,)`,
not `Dict`). Use `--obs_mode rgb` with an online algorithm (`rlpd`,
`dagger`) or the live eval env only, not offline pretraining.

## Dataset: two independent paths

### Official expert demos (recommended, zero new code)

Farama publishes `metaworld/<task-name>/expert-v0` Minari datasets for
essentially all 50 tasks (verified live via `minari.list_remote_datasets()`):
`observation_space=Box(39,) float64`, `action_space=Box(4,) float32` — the
exact same shape as the live v3 env (Meta-World's obs/action layout is
unchanged v2 → v3). This already works today through the existing `minari`
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
`metaworld` module into `sys.modules` — no real MuJoCo/metaworld install
required.

## Known issues and fixes

This backend was installed from scratch (plain `pip install metaworld`, no
CoppeliaSim-class system dependency needed) and exercised end to end against
the real package: single-task and `MT10` env construction, the Minari
expert-dataset path, live scripted-policy demo generation, and a full `bc`
offline-pretraining run with a live `reach-v3` eval env. What actually
surfaced:

- **`pip install minari` alone isn't enough to download a hosted dataset**:
  `minari.load_dataset(..., download=True)` raises `ImportError:
  huggingface_hub is not installed. Please install it using
  \`pip install "minari[hf]"\``. Fixed rl-garden's own `minari`
  optional-dependency group in `pyproject.toml` to install `minari[hf]`
  instead of plain `minari` (a real, pre-existing gap in that extra, not
  Meta-World-specific — it would have broken any Minari-hosted dataset
  download, not just Meta-World's).
- **`success_at_end` was always `nan` in eval output**, regardless of
  `--num_eval_steps`: Meta-World's own `RecordEpisodeStatistics` (part of
  the wrapper stack `gym.make("Meta-World/MT1", ...)` applies internally)
  only ever puts `{"r", "l", "t"}` into `info["episode"]`, never the
  `info["success"]` scalar `AutoTerminateOnSuccessWrapper` also sets every
  step — and rl-garden's shared eval-metric extraction
  (`rl_garden.algorithms.offline._append_episode_metrics`) only reads keys
  out of `info["episode"]`. Fixed the same way `robomimic`'s own backend
  solves the identical problem for its own success key
  (`_RobomimicEpisodeMetrics` in `rl_garden/envs/robomimic/env.py`): a small
  `_MetaWorldEpisodeMetrics` wrapper
  (`rl_garden/envs/metaworld/env.py`) attaches `episode["success_at_end"]`
  at episode end, mirroring that precedent exactly. **Only wired into the
  single-task branch** — Meta-World's own `make_mt_envs` builds `MT10`/
  `MT50`'s sub-envs internally with no per-sub-env construction hook this
  module can wrap, so `success_at_end` stays unavailable for that branch
  (`return`-based eval metrics still work there); not fixed, called out here
  as a known limitation rather than silently dropped.
- Confirmed obs/action space parity for real: the live `reach-v3` env's
  `observation_space`/`action_space` and `metaworld/reach/expert-v0`'s
  Minari-reported spaces are byte-for-byte the same bounds/shape (only
  dtype differs, `float64` vs. Minari's `float32` — already handled by the
  existing `minari` loader's float64→float32 cast).
- Live scripted-policy demo generation's retry-until-success loop was spot
  checked on two real tasks (`reach-v3`, `pick-place-v3`): both reached a
  100% keep rate (10/10 requested demos on the first attempt each) — plausible
  since these are two of Meta-World's easier tasks; the ~80% *aggregate*
  success rate the upstream test suite reports is averaged across all 50
  tasks, harder tasks pull that average down. The retry path itself (a demo
  that never reaches `info["success"] == 1` is discarded and retried) is
  separately covered by a forced-failure case in
  `tests/test_metaworld_dataset.py`.
