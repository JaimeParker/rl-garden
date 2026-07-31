# WSRL Implementation Summary

## Overview

This repository includes a PyTorch implementation of the SAC-family backbone,
standalone **CQL (Conservative Q-Learning)** / **Cal-QL (Calibrated
Q-Learning)** entrypoints, and **WSRL (Warm-Start Reinforcement Learning)**.
The current design keeps SAC/REDQ update mechanics in a shared core, CQL and
Cal-QL as independent algorithm layers, and WSRL as the offline→online flow
layer.

WSRL enables efficient offline→online training:
- **Offline phase**: Pre-train with Cal-QL on ManiSkill trajectory H5 datasets
- **Online phase**: Fine-tune with SAC or CQL without retaining offline data

Standalone offline training is also available:
- **CQL**: Pure offline CQL pretraining from flat H5 datasets
- **CalQL**: Pure offline Cal-QL pretraining with MC return lower bounds

For the end-to-end PickCube reproduction workflow, including SAC checkpoint
training, WSRL dataset generation, and offline-to-online launch commands, see
[`wsrl-reproduction.md`](../guides/wsrl-reproduction.md).

## Key Features

### ✅ Algorithms Implemented
- **SAC**: Online SAC using the shared SACCore update path for Box and Dict observations
- **OfflineSAC**: Offline SAC scaffold over static replay buffers
- **CQL / CalQL**: Pure offline CQL and Cal-QL pretraining algorithms
- **WSRL**: Offline→online WSRL with CQL/Cal-QL for Box and Dict observations

### ✅ Core Components
- **SACCore**: Shared actor/critic/temperature update mechanics
- **Q-Ensemble (REDQ)**: 10 critics by default with subsampling (2 critics for target)
- **CQL Regularization**: Prevents Q-value overestimation with OOD action sampling
- **Cal-QL Lower Bounds**: Uses Monte Carlo returns to calibrate Q-values
- **Offline→Online Switching**: Seamless mode transition with configurable parameters
- **High-UTD Training**: Multiple critic updates per actor update

### ✅ Observation Support
- State-based observations (flat vectors)
- Vision-based observations (RGB/RGBD with dict spaces)
- Encoder detachment on actor path for efficient vision training

## Quick Start

### Installation

```bash
# Install rl-garden with dependencies
pip install -e .
```

### State-Based Training

```bash
# Online-only training (no offline pre-training)
python examples/train_off2on.py wsrl --env_id PickCube-v1 --num_offline_steps 0

# Offline→online training
python examples/train_off2on.py wsrl \
    --env_id PickCube-v1 \
    --offline_dataset_path demos/pickcube_state.h5 \
    --num_offline_steps 100000 \
    --num_online_steps 50000 \
    --n_critics 10 \
    --use_calql

# Use shell launcher
./scripts/train_wsrl.sh --env_id PickCube-v1
```

### Vision-Based Training

```bash
# RGB observations with plain_conv encoder
python examples/train_off2on.py wsrl \
    --env_id PickCube-v1 \
    --obs_mode rgb \
    --encoder plain_conv

# RGBD observations with ResNet encoder
python examples/train_off2on.py wsrl \
    --env_id PickCube-v1 \
    --obs_mode rgbd \
    --encoder resnet10

# Use shell launcher
./scripts/train_wsrl_rgbd.sh --env_id PickCube-v1 --encoder resnet10
```

### Offline-Only Pretraining (No Sim Env)

Use this when you have a static offline dataset (e.g., real-robot teleop H5)
and want a pretrained actor + critic checkpoint without spinning up a sim
env or running any eval.

Use the generic offline pretraining entrypoint and pass the algorithm as a
subcommand:

```bash
# Pure offline CQL
python examples/pretrain_offline.py cql \

    --offline_dataset_path /path/to/real_robot.h5 \
    --num_offline_steps 200000 \
    --checkpoint_dir runs/cql_pretrain \
    --buffer_device cuda

# Pure offline Cal-QL
python examples/pretrain_offline.py calql \

    --offline_dataset_path /path/to/real_robot.h5 \
    --num_offline_steps 200000 \
    --checkpoint_dir runs/calql_pretrain \
    --buffer_device cuda \
    --use_calql --cql_alpha 5.0

# Equivalent shell launchers
scripts/pretrain_offline.sh cql --offline_dataset_path /path/to/real_robot.h5
scripts/pretrain_offline.sh calql --offline_dataset_path /path/to/real_robot.h5
```

These write `cql_offline_pretrained.pt` or `calql_offline_pretrained.pt` by
default. The script infers obs/action specs from the H5, constructs an
`OfflineEnvSpec`, loads the dataset into the algorithm replay buffer, and runs
`run_offline_pretraining()`.

For WSRL-specific offline pretraining, use the `wsrl` subcommand. It builds a
`WSRL` agent directly (Cal-QL by definition) and is useful when the checkpoint
will be resumed by WSRL's offline→online flow:

```bash
python examples/pretrain_offline.py wsrl \

    --offline_dataset_path /path/to/real_robot.h5 \
    --num_offline_steps 200000 \
    --checkpoint_dir runs/robot_pretrain \
    --buffer_device cuda \
    --batch_size 256 \
    --use_calql --cql_alpha 5.0
```

The WSRL mode writes
`runs/robot_pretrain/checkpoints/wsrl_offline_pretrained.pt` by default, which
contains the policy, critic ensemble, target critic, optimizer state, and
Lagrange multipliers — everything needed to resume.

**Online fine-tune on a deployment machine** (which does have an env):

```bash
python examples/train_off2on.py wsrl \
    --env_id <your_env_id> \
    --load_checkpoint runs/robot_pretrain/checkpoints/wsrl_calql_offline_pretrained.pt \
    --num_offline_steps 0 \
    --num_online_steps 50000 \
    --online_replay_mode mixed \
    --offline_data_ratio 0.5
```

This cleanly separates the two WSRL phases across machines: the pretraining
host needs no sim, and the deployment host runs only online fine-tuning.

**Constraints:**
- State observations only (flat `Box`). For RGBD pretraining, write a vision
  variant — the standalone offline CQL/Cal-QL entrypoint and WSRL offline
  pretrain script raise with a clear error if they detect dict obs.
- Action bounds default to ±1; override with `--action_low` / `--action_high`
  if your dataset uses a different action space.
- `OfflineEnvSpec` has no `reset` / `step`; pure offline algorithms interpret
  `learn()` as gradient steps, while WSRL online fine-tuning still requires a
  real environment.

## Configuration Options

### Q-Ensemble (REDQ)
- `--n_critics 10`: Number of Q-networks in ensemble (default: 10)
- `--critic_subsample_size 2`: Number of critics to subsample for target (default: 2)

### Network Architecture (`net_arch`)
- `net_arch` is the primary network config interface for `SAC/WSRL`.
- Supported forms:
  - `list[int]`: shared architecture for actor and critic, e.g. `[256, 256, 256]`
  - `dict(pi=[...], qf=[...])`: separate actor/critic MLPs, e.g. `{"pi": [256, 256], "qf": [256, 256]}`
- `actor_hidden_dims` / `critic_hidden_dims` remain available for backward compatibility but are deprecated.

### CQL Parameters
- `--use_cql_loss`: Enable CQL regularization (default: True)
- `--cql_alpha 5.0`: CQL regularization weight (default: 5.0)
- `--cql_n_actions 10`: Number of OOD actions to sample (default: 10)
- `--cql_action_sample_method uniform`: Random OOD action source (`uniform` | `normal`)
- `--cql_autotune_alpha`: Auto-tune CQL alpha via Lagrange multiplier
- `--cql_importance_sample`: Use importance sampling (default: True)
- `--cql_max_target_backup`: Use max Q for target (default: True)
- `--cql_diff_clip_mode {skip_when_autotune,always}`: When to clamp the CQL
  OOD/data Q-diff to `[cql_clip_diff_min, cql_clip_diff_max]` (default:
  `skip_when_autotune`, matching WSRL — the clamp is skipped whenever
  `cql_autotune_alpha=True`). `always` matches the official Cal-QL JAX repo
  and CORL, which clamp unconditionally regardless of autotuning.
- `--cql_penalty_scale {lagrange_only,lagrange_times_alpha}`: How the
  autotuned Lagrange-weighted CQL penalty is scaled (default: `lagrange_only`,
  matching WSRL: `alpha_prime * (diff - target_gap)`). `lagrange_times_alpha`
  matches official Cal-QL JAX and CORL, which also multiply by the fixed
  `cql_alpha` scalar: `alpha_prime * cql_alpha * (diff - target_gap)`.
- `--cql_alpha_param {softplus,exp_clip}`: Parameterization of the CQL alpha
  Lagrange multiplier (default: `softplus`, unbounded, matching WSRL).
  `exp_clip` matches official Cal-QL JAX and CORL: `clip(exp(log_alpha), 0, 1e6)`.
- `--backup_entropy`: Include entropy in TD target backups (default: False, matching
  upstream WSRL/Cal-QL). This is a single global config — the same value applies to
  both offline and online phases. `switch_to_online_mode` does **not** flip it.
  The upstream `3rd_party/wsrl` config sets `backup_entropy=False` and the CQL
  agent asserts this invariant (`cql.py:240`); diverging would inject an entropy
  bonus into the TD target at the offline→online boundary.

### Cal-QL Parameters
- `--use_calql`: Enable Cal-QL lower bounds (default: True)
- `--calql_bound_random_actions`: Apply bounds to random actions (default: False)

### Offline→Online Control
- `--num_offline_steps 100000`: Number of offline training steps
- `--offline_dataset_path demos/foo.h5`: ManiSkill trajectory H5 path for offline pre-training
- `--offline_num_traj`: Optional number of trajectories to load from the H5
- `--num_online_steps 50000`: Number of online training steps
- `--warmup_steps 5000`: Frozen-policy rollout steps before online updates begin (paper default)
- `--online_replay_mode {empty,append,mixed}`: How to handle replay buffer at switch (default: `empty`, matches WSRL paper)
- `--online_use_cql_loss`: Whether CQL stays on during the online phase (default: `False`, paper-aligned)
- `--online_cql_alpha`: CQL alpha applied at the switch (default: `0.0`, paper-aligned)
- `--offline_data_ratio 0.5`: Only meaningful when `online_replay_mode=mixed`

#### Paper-aligned WSRL vs Cal-QL retention

The defaults above reproduce the WSRL recipe (drop CQL after warmup, empty
buffer, pure SAC online). If you instead want Cal-QL with offline-data
retention, override:

```bash
# WSRL paper recipe (default — no flags needed)
python examples/train_off2on.py wsrl --env_id <id> --offline_dataset_path <h5> \
    --num_offline_steps 200000 --num_online_steps 200000

# Cal-QL retention recipe
python examples/train_off2on.py wsrl --env_id <id> --offline_dataset_path <h5> \
    --num_offline_steps 200000 --num_online_steps 200000 \
    --online_use_cql_loss True --online_cql_alpha 5.0 \
    --online_replay_mode mixed --offline_data_ratio 0.5
```

**`switch_to_online_mode` emits a `UserWarning`** if `use_cql_loss=True` is
combined with `online_replay_mode='empty'`. That combination keeps CQL active
on a buffer with no offline support; CQL's conservatism floor is calibrated
against the offline data distribution, and on warmup-only data the OOD
LogSumExp estimates become high-variance, fighting the policy gradient. Pick
either pure WSRL (CQL off) or Cal-QL retention (mixed/append buffer).

#### WSRL switch-time diagnostics in wandb

`switch_to_online_mode` records these one-shot summaries (visible in the wandb
"Summary" panel, not the time-series charts):

- `wsrl/online_start_step`, `wsrl/warmup_end_step`
- `wsrl/online_use_cql_loss`, `wsrl/online_cql_alpha`, `wsrl/online_backup_entropy`
- `wsrl/online_replay_mode`, `wsrl/online_replay_cleared`,
  `wsrl/online_replay_size_before_clear` (empty mode), `wsrl/offline_data_ratio` (mixed mode)
- `wsrl/recompile_at_online_step` (only when `--use_compile`) — lets you separate
  recompile transients from real unlearning when reading post-switch curves

### Standalone Offline CQL/Cal-QL Control
- `cql`: Use `CQL` with a normal tensor replay buffer.
- `calql`: Use `CalQL` with an MC replay buffer and Cal-QL bounds.
- `wsrl`: Build `WSRL` for checkpoints intended for the WSRL offline→online flow.
- `--save_filename`: Override the default
  `<algorithm>_offline_pretrained.pt` checkpoint name.
- `--offline_sampling without_replace`: Sample offline batches without repeating
  until the static replay buffer is exhausted.

### Upstream-Parity Defaults
- `policy_lr=1e-4`, `q_lr=3e-4`, `alpha_lr=1e-4`
- `gamma=0.99`, `tau=0.005`
- WSRL actor/critic MLPs use layer norm by default
- Actor std parameterization supports `exp` and `uniform`

#### Aligning with official Cal-QL JAX / CORL

rl-garden's CQL/Cal-QL loss defaults follow the WSRL reference. Official
Cal-QL JAX and CORL's Cal-QL are **not interchangeable targets** — they
agree on the core CQL/Cal-QL math but diverge on the CQL diff-clip value,
critic/actor depth, and weight-init scheme, so each gets its own recipe
below rather than one shared flag block. Both recipes need the legacy D4RL
AntMaze environment, not the Minari-recovered one: install the
`d4rl-legacy` extra before running either.

##### Parameter reference

| Axis | Official Cal-QL JAX | CORL Cal-QL | rl-garden flag |
|---|---|---|---|
| Reward transform | scale=10, bias=-5 | scale=10, bias=-5 | `--reward_scale 10 --reward_bias -5` |
| Discount / target update | gamma=0.99, tau=0.005 | gamma=0.99, tau=0.005 | `--gamma 0.99 --tau 0.005` |
| CQL alpha (static weight) | 5.0 | 5.0 (offline and online) | `--cql_alpha 5.0` |
| Lagrange target gap | 0.8 | 0.8 | `--cql_target_action_gap 0.8` |
| Lagrange penalty formula | `alpha_prime * cql_min_q_weight * (diff - gap)` | same | `--cql_penalty_scale lagrange_times_alpha` |
| Lagrange multiplier parameterization | `clip(exp(log_alpha), 0, 1e6)`, raw param initialized to `1.0` → effective α′₀ = `e` | same clip form | `--cql_alpha_param exp_clip --cql_alpha_lagrange_init 2.718281828459045` |
| CQL diff clip | applied unconditionally, but left at `[-inf, inf]` (a no-op) | applied unconditionally, **`cql_clip_diff_min=-200`** | `--cql_diff_clip_mode always` (add `--cql_clip_diff_min -200.0` for CORL only) |
| Critic depth | 4 hidden layers × 256 | **5** hidden layers × 256 | see "Architecture cannot be set for off2on" below |
| Actor depth | 2 hidden layers × 256 | **3** hidden layers × 256 (hardcoded in CORL, not config-driven) | see below |
| Critic count | twin-Q | twin-Q | `--n_critics 2 --critic_subsample_size 2` (self-disables ensembling whenever `>= n_critics`) |
| Weight init | orthogonal, hidden gain=√2, **final layer gain=1e-2**, bias=0 | orthogonal, gain=√2 **uniformly on every layer including outputs**, bias=0 | `--kernel_init orthogonal_near_zero_output` matches JAX exactly; **no option currently matches CORL** (Gap 2 below) |
| Actor log_std affine | multiplier=1.0, offset=-1.0 | multiplier=1.0, offset=-1.0 | `--policy_log_std_multiplier 1.0 --policy_log_std_offset -1.0` |
| Offline / online steps | 1M / 1M | 1M / 1M | `--num_offline_steps 1000000` / `--num_online_steps 1000000` |
| Offline/online mix ratio | 0.5 | 0.5 | `--online_replay_mode mixed --offline_data_ratio 0.5` — set explicitly; rl-garden's off2on default is `"auto"`, an adaptive scheme unique to rl-garden that matches neither reference |
| Online CQL | stays active, same `cql_alpha=5.0` | stays active, same `cql_alpha=5.0` | `--online_use_cql_loss True --online_cql_alpha 5.0` |
| Online Cal-QL calibration (`max(Q, mc_return)` bound) | **stays active** — `enable_calql` is one static flag never toggled at the online switch, and online MC returns are computed per-trajectory, not placeholders | **disabled** at the online switch via `switch_calibration()` | rl-garden's `use_calql` is one static value for the whole run — matches JAX as-is; **cannot reproduce CORL's online-off behavior** (Gap 3 below) |
| Eval episodes | script uses 20; use 100 to reduce sparse-success-rate noise | 100 | `--num_eval_episodes 100` for both |
| Env / dataset | legacy D4RL | legacy D4RL | `--env_backend d4rl_legacy --dataset_source d4rl_legacy` |

##### Official Cal-QL JAX — offline pretrain

`scripts/pretrain_calql_d4rl_legacy.sh` already encodes this exact recipe —
run it directly rather than reassembling the flags by hand. (It is currently
an untracked file in this checkout; if it's gone, rebuild it from the table
above plus `--actor_hidden_layers 2 --critic_hidden_layers 4` for the
offline entrypoint's architecture flags.)

##### CORL Cal-QL — offline pretrain

Same script, with the CORL-specific deltas from the table applied:

```bash
scripts/pretrain_calql_d4rl_legacy.sh \
    --actor_hidden_layers 3 \
    --critic_hidden_layers 5 \
    --cql_clip_diff_min -200.0
```

`--kernel_init` has no exact CORL match (Gap 2) — `orthogonal_near_zero_output`
is the closest available option, but it does not reproduce CORL's
uniform-gain scheme.

##### Online fine-tuning — currently blocked for both

The reference recipe for continuing either checkpoint into online
fine-tuning would be:

```bash
python examples/train_off2on.py calql \
    --load_checkpoint <offline_checkpoint.pt> \
    --env_backend d4rl_legacy --dataset_source d4rl_legacy \
    --online_replay_mode mixed --offline_data_ratio 0.5 \
    --online_use_cql_loss True --online_cql_alpha 5.0 \
    --num_online_steps 1000000
```

**This currently fails for both styles** — see Gap 1 below. Continuing a
JAX/CORL-aligned offline checkpoint through the standard off2on entrypoint
is not possible today.

##### Known gaps

- **Gap 1 (blocking — hard error).** Two independent structural mismatches
  between the offline-pretrained checkpoint and the off2on-constructed
  model:
  - `rl_garden/training/off2on/_args.py`'s `CQLOff2OnArgs` has no
    `policy_log_std_multiplier` / `policy_log_std_offset` fields, and
    `off2on/calql.py` / `off2on/wsrl.py`'s `build_calql` / `build_wsrl`
    never pass them through to `Off2OnCalQL(...)` / `WSRL(...)`. A
    JAX/CORL-aligned offline actor has trainable `log_std_multiplier` /
    `log_std_offset` parameters in its `state_dict`; the off2on-constructed
    actor does not.
  - The off2on entrypoint exposes no `net_arch` / hidden-layer flags at
    all — `CQLCore._resolve_net_arch` (`rl_garden/algorithms/cql.py:411-441`)
    falls back to `{"pi": [256, 256], "qf": [256, 256]}` for every off2on
    run, regardless of what architecture the offline checkpoint used (4 or
    5 critic hidden layers, 2 or 3 actor hidden layers per the table above).

  Either mismatch alone produces a `state_dict` key/shape mismatch.
  `agent.load()` defaults to `strict=True`
  (`rl_garden/algorithms/base_algorithm.py:257`), so loading a JAX/CORL-aligned
  offline checkpoint into `train_off2on.py` raises a `RuntimeError`. There
  is currently no way to continue such a checkpoint into online
  fine-tuning through the standard off2on entrypoint.
- **Gap 2 (non-blocking — silent divergence, offline phase only).** No
  `kernel_init` value reproduces CORL's `orthogonal_init=True` scheme
  (uniform gain=√2 on every `nn.Linear`, including output layers) —
  rl-garden's `"orthogonal"` uses gain=1.0
  (`rl_garden/networks/mlp.py:35-36`), and `"orthogonal_near_zero_output"`
  uses JAX's mixed-gain scheme, not CORL's. This only affects the offline
  pretrain phase (initialization is overwritten once a checkpoint loads on
  top), so CORL-style offline pretraining currently cannot match CORL's
  exact initial weights.
- **Gap 3 (non-blocking — silent divergence, CORL online phase only).**
  rl-garden's `use_calql` is one static value for an entire run; there is
  no `online_use_calql` toggle analogous to CORL's `switch_calibration()`.
  Official JAX doesn't need this — it keeps calibration active throughout
  online fine-tuning, so JAX-style reproduction is unaffected. CORL-style
  online fine-tuning currently cannot disable the calibration bound at the
  online switch the way CORL does.

### Vision-Specific
- `--obs_mode rgb`: Observation mode (rgb | rgbd)
- `--encoder plain_conv`: Image encoder (plain_conv | resnet10 | resnet18)
- `--camera_width 128`: Camera width (default: 128)
- `--camera_height 128`: Camera height (default: 128)

### Acceleration

Two single-GPU speedups are wired in. Both are orthogonal and stack.

**1. `EnsembleQCritic` is vmap-fused (always on).** N critics share one
prototype and stacked parameters; `torch.func.vmap` runs them in a single
fused forward pass instead of N independent kernel launches. Replaces the
old `nn.ModuleList` layout. Legacy checkpoints (with `q_nets.<i>.*` keys)
are migrated transparently on load.

**2. `--use_compile` (off by default).** Wraps `_critic_loss`, `_actor_loss`,
and `_target_q` with `torch.compile(mode="default")`. First step pays a
30–60 s warm-up; subsequent steps run a fused inductor graph.

```bash
# Enable compile-based acceleration
python examples/pretrain_offline.py wsrl \

    --offline_dataset_path real_robot.h5 \
    --num_offline_steps 100000 \
    --batch_size 1024 \
    --use_compile
```

**Measured speedup** (RTX 5060, PyTorch 2.11, state-only, batch=1024,
n_critics=10, cql_n_actions=10):

| Configuration              | ms / grad step | step/s | speedup |
|----------------------------|---------------:|-------:|--------:|
| vmap critic only (default) |         86.0   |  11.6  |    1.0× |
| vmap + `--use_compile`     |         48.8   |  20.5  |    1.76× |

Notes:
- `compile_mode="reduce-overhead"` uses CUDA graphs and currently conflicts
  with the separately-compiled critic/actor methods (tensor lifetimes cross
  callable boundaries). Stick with the default `"default"` mode unless you
  benchmark a specific environment.
- Inside `switch_to_online_mode`, the compiled methods are re-wrapped because
  Python-side flags (`use_cql_loss`, `cql_alpha`) may have flipped and would
  otherwise leave a stale specialization in the graph. `backup_entropy` is
  **not** among these — it stays at its constructor value across phases. The
  recompile step is recorded as the `wsrl/recompile_at_online_step` wandb
  summary so you can tell post-switch recompile transients apart from real
  unlearning when inspecting Q-value curves.
- For larger speedups, also raise `--batch_size` until VRAM is exhausted —
  the small networks here leave most of the GPU idle.

## Python API

### State-Based WSRL

```python
from rl_garden.algorithms import WSRL
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.envs import make_maniskill_env, ManiSkillEnvConfig

# Create environment
env_cfg = ManiSkillEnvConfig(env_id="PickCube-v1", num_envs=16, obs_mode="state")
env = make_maniskill_env(env_cfg)

# Create WSRL agent
agent = WSRL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256]},
    n_critics=10,  # REDQ ensemble
    critic_subsample_size=2,
    use_cql_loss=True,
    use_calql=True,
    cql_alpha=5.0,
    gamma=0.99,
)

# Offline training
load_maniskill_h5_to_replay_buffer(agent.replay_buffer, "demos/pickcube_state.h5")
for _ in range(100_000):
    agent.train(gradient_steps=1)

# Switch to online mode
agent.switch_to_online_mode()

# Online fine-tuning
agent.learn(total_timesteps=50_000)
```

### Offline CQL/Cal-QL Pretraining (Python)

```python
import numpy as np
from gymnasium import spaces

from rl_garden.algorithms import CalQL, OfflineEnvSpec
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer


obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=1)

agent = CalQL(
    env=env_spec,
    n_critics=10,
    critic_subsample_size=2,
    use_calql=True,
    cql_alpha=5.0,
    checkpoint_dir="runs/calql_pretrain/checkpoints",
)

load_maniskill_h5_to_replay_buffer(agent.replay_buffer, "real_robot.h5")
agent.learn_offline(num_steps=200_000, save_filename="calql_offline_pretrained.pt")
```

Use `CQL` instead of `CalQL` for pure CQL without MC-return
lower bounds. To resume on a deployment host with a real env, construct a
compatible `WSRL`, `CalQL`, or `SAC`-family agent against the live env and call
`agent.load(checkpoint_path)`.

### WSRL Offline→Online Resume (Python)

```python
from rl_garden.algorithms import WSRL

agent = WSRL(env=live_env, n_critics=10, critic_subsample_size=2, use_calql=True)
agent.load("runs/robot_pretrain/checkpoints/wsrl_calql_offline_pretrained.pt")
agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
agent.learn(total_timesteps=50_000)
```

### Vision-Based WSRL

```python
from rl_garden.algorithms import WSRL
from rl_garden.encoders import default_image_encoder_factory

# Create environment with RGB observations
env_cfg = ManiSkillEnvConfig(
    env_id="PickCube-v1",
    num_envs=16,
    obs_mode="rgb",
    include_state=True,
)
env = make_maniskill_env(env_cfg)

# Create WSRL agent
agent = WSRL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256]},
    n_critics=10,
    use_calql=True,
    image_keys=("rgb",),
    state_key="state",
    image_encoder_factory=default_image_encoder_factory(features_dim=256),
)

# Train
agent.learn(total_timesteps=1_000_000)
```

## Architecture

### Class Hierarchy

```
OffPolicyAlgorithm
├── SAC(SACCore)
└── _CQLRolloutTrainingShell(CQLCore)
    └── _CalQLRolloutTrainingShell
        └── WSRL

OfflineRLAlgorithm
├── OfflineSAC(SACCore)
└── CQL(CQLCore)
    └── CalQL
```

### Key Components

1. **SACCore** (`rl_garden/algorithms/sac_core.py`)
   - Shared SAC actor/critic/alpha update loop
   - REDQ target critic subsampling
   - High-UTD splitting, scheduler stepping, grad clipping, target updates

2. **SACPolicy / WSRLPolicy** (`rl_garden/policies/`)
   - `SACPolicy` owns Q-ensembles, critic subsampling helpers, modern MLP
     options, and actor std parameterization.
   - `WSRLPolicy` is a compatibility shim with WSRL-style defaults.
   - CQL alpha Lagrange state is owned by `CQLCore`, not the policy.

3. **CQL / CalQL** (`rl_garden/algorithms/cql.py`, `calql.py`)
   - `CQLCore` implements conservative regularization, CQL alpha, and max
     target backup.
   - `CalQLCore` adds MC replay buffers and MC return lower bounds.
   - Online and offline shells share the same loss implementation.

4. **MCReplayBuffer** (`rl_garden/buffers/mc_buffer.py`)
   - Cached vectorized Monte Carlo return computation
   - Episode boundary tracking
   - Circular-buffer wraparound handling
   - Support for both Tensor and Dict observations

5. **ManiSkill H5 Loader** (`rl_garden/buffers/maniskill_h5.py`)
   - Loads `traj_*` H5 groups into existing MC replay buffers
   - Supports flat state observations and dict/RGBD observation groups

6. **WSRL Algorithm** (`rl_garden/algorithms/wsrl.py`)
   - Inherits the Cal-QL rollout shell
   - Auto-selects `FlattenExtractor`/`MCTensorReplayBuffer` for Box observations
   - Auto-selects `CombinedExtractor`/`MCDictReplayBuffer` for Dict observations
   - Offline→online mode switching
   - Empty/append/mixed replay modes
   - Offline probe and WSRL phase logging

## Test Coverage

The SAC/CQL/Cal-QL/WSRL stack has focused tests covering policy options,
CQL/Cal-QL loss semantics, standalone offline CQL/Cal-QL pretraining,
high-UTD dispatch, MC replay returns, RGBD support, checkpoint roundtrips, and
ManiSkill H5 loading.

Run tests:
```bash
pytest tests/test_cql_calql.py tests/test_sac_core.py tests/test_wsrl*.py -v
```

## References

- **WSRL Paper**: [Warm-Start Reinforcement Learning](https://arxiv.org/abs/2412.07762)
- **Cal-QL Paper**: [Calibrated Q-Learning](https://arxiv.org/abs/2303.05479)
- **CQL Paper**: [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)
- **REDQ Paper**: [Randomized Ensembled Double Q-learning](https://arxiv.org/abs/2101.05982)

## Implementation Notes

- Follows rl-garden's SB3-style architecture
- GPU-native operations (no numpy in hot path)
- Compatible with ManiSkill's GPU-parallel environments
- Supports both state and vision observations
- Minimal, focused implementation (no unnecessary abstractions)

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `--batch_size` or `--n_critics`
2. **Slow training**: Increase `--utd` for more gradient steps per env step
3. **Unstable training**: Reduce `--cql_alpha` or disable with `--use_cql_loss False`

### Performance Tips

- Use `--n_critics 10` with `--critic_subsample_size 2` for best offline performance (REDQ)
- CLI defaults already match WSRL paper recipe (`--utd 4.0`,
  `--online_use_cql_loss False`, `--online_cql_alpha 0.0`,
  `--online_replay_mode empty`); override only if you want Cal-QL retention
  behavior or a different update-to-data ratio
- For state-based WSRL, the default `--utd 4.0` follows paper Table 1
  (Adroit/Kitchen/AntMaze); vision-based training keeps `--utd 0.25`
- Enable `--use_calql` for better offline pre-training with Cal-QL bounds

## License

See LICENSE file in the repository root.
