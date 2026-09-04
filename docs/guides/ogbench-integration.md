# OGBench Integration

This guide covers installing the `ogbench` env backend, the standard
(non-goal-conditioned) offline RL usage OGBench itself documents, the
task-family matrix, and regenerating OGBench's own datasets. It complements
[`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md),
which this backend follows structurally (mirrors the already-shipped
`mujoco` backend almost exactly).

OGBench (`3rd_party/ogbench`, ARISE — actually seohongpark/ogbench) provides
`mujoco`/`dm_control`-based locomotion and manipulation tasks under a
standard Gymnasium API, with HuggingFace/berkeley-hosted demonstration
datasets. Scope of this integration: **all 8 task families**
(locomotion: `pointmaze`/`antmaze`/`humanoidmaze`/`antsoccer`; manipulation:
`cube`/`scene`/`puzzle`), **both state and pixel observations**, consuming
the `*-singletask-*` variants OGBench itself provides for standard
(reward-based, non-goal-conditioned) offline RL — see [OGBench's own
README section on this](https://github.com/seohongpark/ogbench#usage-for-standard-non-goal-conditioned-offline-rl).
`powderworld` has no `singletask` registration at all and is out of scope
for standard RL, not a scoping choice made here.

## Install

```bash
python -m pip install -e '.[ogbench]'
```

Unlike `robomimic`, there is no version-pin fight here: plain `ogbench` from
PyPI declares `mujoco>=3.1.6`, `dm_control>=1.0.20`, `gymnasium[mujoco]`
with no upper bounds.

## The env-id vs dataset-name naming gotcha

OGBench's env id and its dataset name are **two different strings for the
same task** — the dataset name keeps the dataset-type word
(`navigate`/`stitch`/`play`/`noisy`) immediately before `singletask`; the env
id built from it drops that word. For example:

- Dataset name (pass to `--offline-dataset`): `cube-double-play-singletask-task2-v0`
- Env id (pass to `--env_id`): `cube-double-singletask-task2-v0`

There is a **third** string, relevant only if you regenerate or manually
place a dataset file: the on-disk `.npz` filename `ogbench.make_env_and_datasets`
actually looks for drops `singletask`/`task[n]` entirely (verified by
running this end to end) — for the dataset name above it looks for
`cube-double-play-v0.npz`, *not* `cube-double-play-singletask-task2-v0.npz`.
This is because the underlying trajectories are shared between the
goal-conditioned and every singletask variant of a family; only the
in-memory reward relabeling differs, which happens at load time regardless
of which `task[n]` you asked for. See "Dataset regeneration" below for the
concrete naming this implies for a regenerated file.

This is OGBench's own convention (see `ogbench.utils.make_env_and_datasets`),
not something this integration invents or auto-derives — the two flags will
look different for the same task, every time.

## Task-family matrix

State variant always exists; visual (pixel) variant only where noted:

| Family | Variants | Visual variant |
|---|---|---|
| `pointmaze` | `medium`/`large`/`giant`/`teleport` | no |
| `antmaze` | `medium`/`large`/`giant`/`teleport` | yes |
| `humanoidmaze` | `medium`/`large`/`giant`/`teleport` | yes |
| `antsoccer` | `arena`/`medium` | no |
| `cube` | `single`/`double`/`triple`/`quadruple` | yes |
| `cube-octuple` | (single variant) | no — and **no standard hosted singletask dataset**; only a manually-downloaded 100M-transition dataset (see OGBench README's "Additional Features") |
| `scene` | (single variant) | yes |
| `puzzle` | `3x3`/`4x4`/`4x5`/`4x6` | yes |

Each family's five task variants are `-singletask-task[1..5]-v0`, aliased by
`-singletask-v0` (no number) for the family's "default" task — see OGBench's
own default-task table.

## Pixel (visual-*) observations

Pixel envs return a single flat `Box(0, 255, (H, W, C), uint8)` per env, not
a per-camera Dict. rl-garden's `ImageFrameStackWrapper` only supports
Dict-keyed `rgb`/`depth` observations, so **frame stacking is out of scope
for this integration** — pixel observations pass through as a flat Box.
Set `--ogbench.vectorization async` for `visual-*` env ids: each env
instance owns its own MuJoCo renderer/GL context, and running each in its
own OS process sidesteps the same context-sharing risk the `mujoco`
backend's own module docstring already documents for camera observations.

## Usage

Manipulation task (arm manipulation, `cube-single`):

```bash
python examples/train_online.py rlpd \
  --env-backend ogbench --env-id cube-single-singletask-v0 \
  --obs-mode state \
  --dataset-backend ogbench --offline-dataset cube-single-play-singletask-v0 \
  --num-envs 4 --num-eval-envs 2 \
  --total-timesteps 100000 --learning-starts 1000 --batch-size 256
```

Locomotion task (`antmaze-large`):

```bash
python examples/train_online.py rlpd \
  --env-backend ogbench --env-id antmaze-large-singletask-task1-v0 \
  --obs-mode state \
  --dataset-backend ogbench --offline-dataset antmaze-large-navigate-singletask-task1-v0 \
  --num-envs 4 --num-eval-envs 2 \
  --total-timesteps 100000 --learning-starts 1000 --batch-size 256
```

Pixel task (`visual-antmaze-medium`), `async` vectorization:

```bash
python examples/train_online.py rlpd \
  --env-backend ogbench --env-id visual-antmaze-medium-singletask-task1-v0 \
  --ogbench.vectorization async \
  --obs-mode rgb \
  --dataset-backend ogbench --offline-dataset visual-antmaze-medium-navigate-singletask-task1-v0 \
  --num-envs 4 --num-eval-envs 2 \
  --total-timesteps 100000 --learning-starts 1000 --batch-size 256
```

### Key config fields (`--ogbench.<field>`)

- `device`: device for the online vector env's torch tensors.
- `env_kwargs_json`: JSON-encoded dict forwarded verbatim to `gym.make()`.
  Rarely needed — every task variant is already encoded in `env_id` itself
  (obs modality included).
- `vectorization`: `"sync"` (default, single process) or `"async"` (one OS
  process per env — recommended for `visual-*` env ids).

## Dataset regeneration

OGBench's own `data_gen_scripts/` (in `3rd_party/ogbench`, not part of the
installed PyPI package) can regenerate any dataset from scratch. Save the
regenerated file(s) at `<dataset_dir>/<family>-<variant>-<dataset_type>-v0.npz`
(**no** `singletask`/`task[n]` in the filename — see the naming gotcha
above) — `ogbench.make_env_and_datasets` only downloads a dataset when the
expected file is missing at `dataset_dir` (default `~/.ogbench/data`, same
default `load_ogbench_dataset_to_replay_buffer` uses), so a file already
sitting there is picked up transparently, no code changes needed. Verified
end to end this session: regenerated `cube-single-play-v0.npz` (+ its
`-val.npz`, both required — `ogbench`'s own loader reads both
unconditionally) with 15 episodes locally, dropped both at
`~/.ogbench/data/`, then loaded `cube-single-play-singletask-v0` through
`load_ogbench_dataset_to_replay_buffer` with **zero network access** — 15000
transitions loaded, correctly relabeled (544 task-success transitions).

### Manipulation/puzzle (zero extra dependencies)

Manipulation datasets are produced by a scripted oracle
(`ogbench.manipspace.oracles`), no jax involved — runs directly in the same
`ogbench` extra's venv. `generate_manipspace.py` reserves `num_episodes // 10`
episodes for the validation split, so request at least 10 episodes or the
`-val.npz` file will be empty and fail to load:

```bash
python 3rd_party/ogbench/data_gen_scripts/generate_manipspace.py \
  --env_name=cube-single-v0 --dataset_type=play \
  --num_episodes=1000 --save_path=data/cube-single-play-v0.npz
```

`--dataset_type`: `play` (non-Markovian oracle following a pre-computed
plan) or `noisy` (Markovian closed-loop oracle with Gaussian action noise).

### Locomotion (needs an isolated jax venv + expert checkpoints)

Locomotion regeneration needs OGBench's own `impls` codebase
(`from agents import SACAgent`) — `jax[cuda12]`/`flax`/`distrax`/
`ml_collections` — plus pretrained expert policy checkpoints. This must
**never** land in rl-garden's main (PyTorch-native) venv. Use a throwaway
venv, same spirit as
[`.agents/runbooks/baseline-install.md`](../../.agents/runbooks/baseline-install.md)'s
per-baseline JAX venv pattern — but this is dataset-generation tooling, not
an RL algorithm baseline, so it is **not** registered in
`baselines/baselines.yaml`:

```bash
python -m venv /tmp/ogbench-datagen && source /tmp/ogbench-datagen/bin/activate
pip install -r 3rd_party/ogbench/impls/requirements.txt

cd 3rd_party/ogbench/data_gen_scripts
wget https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz
tar xf experts.tar.gz && rm experts.tar.gz
export PYTHONPATH="../impls:${PYTHONPATH}"
python generate_locomaze.py --env_name=antmaze-large-v0 \
  --save_path=data/antmaze-large-navigate-v0.npz
```

To train an expert policy from scratch instead of using the hosted
checkpoints, run `main_sac.py --env_name=online-ant-xy-v0` in the same venv
(see OGBench's README "Reproducing expert policies" section).

## Tests

```bash
pytest -q \
  tests/test_ogbench_dataset.py \
  tests/test_ogbench_env.py \
  tests/test_prior_data_replay.py
```

These use monkeypatched fakes for `ogbench` (no network access or
mujoco/dm_control install required).

## Current limits

- No frame stacking for `visual-*` env ids (flat-Box passthrough only —
  see above).
- `cube-octuple` has no standard hosted singletask dataset through
  `ogbench.download_datasets`'s normal naming (only a manually-downloaded
  100M-transition dataset); the env itself is fully supported for online
  rollout.
- Only `cube-single`/`antmaze-large` have been exercised in this
  integration's own smoke tests so far; every other family/variant is
  architecturally identical (same env-id-driven construction, same
  `relabel_dataset()`-based reward labeling) but individually untested.
- No confirmed `seed()`-forwarding beyond `gymnasium.make(env_id)`'s own
  default `reset(seed=...)` handling.
