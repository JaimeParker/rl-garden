# Official FQL OGBench baseline

This directory launches the upstream JAX FQL repository's unmodified
`main.py`. It is not a PyTorch port, an installer, or a checkpoint/parity
evaluation suite. The complete task/phase configuration is in `presets.py`.

| Preset | Dataset | Budget | alpha / discount / Q aggregation |
| --- | --- | --- | --- |
| `cube-single-offline` | `cube-single-play-singletask-v0` | 1M offline | 300 / .99 / mean |
| `cube-double-off2on` | `cube-double-play-singletask-v0` | 1M offline + 1M online | 300 / .99 / mean |
| `scene-off2on` | `scene-play-singletask-v0` | 1M offline + 1M online | 300 / .99 / mean |
| `antmaze-large-task1-offline` | `antmaze-large-navigate-singletask-task1-v0` | 1M offline | 10 / .99 / min |
| `humanoidmaze-medium-task2-offline` | `humanoidmaze-medium-navigate-singletask-task2-v0` | 1M offline | 30 / .995 / mean |
| `antsoccer-arena-task4-off2on` | `antsoccer-arena-navigate-singletask-task4-v0` | 1M offline + 1M online | 30 / .995 / mean |
| `puzzle-3x3-task4-offline` | `puzzle-3x3-play-singletask-task4-v0` | 1M offline | 1000 / .99 / mean |
| `puzzle-4x4-task4-off2on` | `puzzle-4x4-play-singletask-task4-v0` | 1M offline + 1M online | 1000 / .99 / mean |

## Setup

Clone the official source deliberately. The reference-tested revision is
`e8cd16eb490332924dfa2492097219f181765933`; a different revision or dirty
tree is a recorded warning, not a launch blocker. An archive without `.git`
also runs if it contains a compatible `main.py`, with revision recorded as
unknown.

```bash
git clone https://github.com/seohongpark/fql.git /path/to/fql
git -C /path/to/fql checkout e8cd16eb490332924dfa2492097219f181765933
python3.11 -m venv FQL_VENV
FQL_VENV/bin/pip install --upgrade pip
FQL_VENV/bin/pip install -r baselines/fql/requirements-verified.txt
```

The requirements file is the verified OGBench-only Python package record.
Its `jax-cuda12-plugin[with_cuda]` requirement installs the plugin's CUDA 12
NVIDIA wheels for a driver-only host; the NVIDIA driver itself is still a host
requirement and must support that CUDA wheel set. It intentionally omits D4RL,
Gym 0.23, and mujoco-py/MuJoCo 2.1 because upstream `env_utils` imports them
only in its non-OGBench branches; all presets here use OGBench singletask
datasets.

The verified runtime used Python 3.11.14 with JAX 0.4.35, JAXLIB 0.4.34,
`jax-cuda12-plugin[with_cuda]==0.4.35`, and `jax-cuda12-pjrt==0.4.35`; its
resolved CUDA packages included nvcc 12.9.86, CUDA runtime 12.9.79, cuBLAS
12.9.2.10, and cuDNN 9.26.0.51. This is an evidence record, not a universal
driver recipe: keep the shown package versions together and select a compatible
CUDA/driver stack for the target host.

Check the isolated runtime before a formal run:

```bash
FQL_VENV/bin/python -c 'import jax; print(jax.devices())'
FQL_VENV/bin/python -c 'import importlib.metadata as m; print(m.version("ogbench"), m.version("gymnasium"))'
```

## Launch

`--dry-run` only prints the official command. It neither accesses the source
nor imports JAX nor creates `--output-dir`:

```bash
PYTHONPATH=. FQL_VENV/bin/python -m baselines.fql.run_ogbench \
  --fql-source /path/to/fql --preset antsoccer-arena-task4-off2on --seed 0 \
  --output-dir runs/fql/antsoccer-s0 --dry-run
```

For a formal GPU run, choose an available device explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  FQL_VENV/bin/python -m baselines.fql.run_ogbench \
  --fql-source /path/to/fql --preset antsoccer-arena-task4-off2on --seed 0 \
  --output-dir runs/fql/antsoccer-s0
```

Before starting upstream `main.py`, the wrapper requires OGBench 1.1.0,
Gymnasium 0.29.1, its FQL imports, and an actual CUDA JAX device. It writes
`command.json` (resolved recipe, source state, package fingerprint, GPU
preflight, command) and `result.json`. For a separate smoke directory only,
override a budget with `--offline-steps` and `--online-steps`; never treat a
smoke directory as a formal result.

Official FQL creates W&B project `fql` itself. `WANDB_MODE=disabled` is not a
launcher-supported guarantee that initialization will be bypassed; formal runs
need authorized upstream W&B credentials.

The launcher explicitly sends the official state recipe: 4×512 actor/critic,
critic-only LayerNorm, float32 defaults, batch 256, lr 3e-4, tau .005, flow
steps 10, 50 evaluation episodes, 5k logging, and 100k evaluation/save
cadence. It intentionally omits an `agent.encoder` CLI override so that the
upstream null encoder remains a real null rather than a string.
