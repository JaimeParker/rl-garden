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
[`WSRL_REPRODUCTION.md`](WSRL_REPRODUCTION.md).

## Key Features

### ✅ Algorithms Implemented
- **SAC / RGBDSAC**: Online SAC using the shared SACCore update path
- **OfflineSAC**: Offline SAC scaffold over static replay buffers
- **CQL / CalQL**: Pure offline CQL and Cal-QL pretraining algorithms
- **WSRL**: State-based WSRL with CQL/Cal-QL
- **WSRLRGBD**: Vision-based WSRL for RGB/RGBD observations

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
python examples/train_wsrl.py --env_id PickCube-v1 --num_offline_steps 0

# Offline→online training
python examples/train_wsrl.py \
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
python examples/train_wsrl_rgbd.py \
    --env_id PickCube-v1 \
    --obs_mode rgb \
    --encoder plain_conv

# RGBD observations with ResNet encoder
python examples/train_wsrl_rgbd.py \
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

Use the generic offline pretraining entrypoint and select the algorithm with
`--algorithm`:

```bash
# Pure offline CQL
python examples/pretrain_offline.py \
    --algorithm cql \
    --offline_dataset_path /path/to/real_robot.h5 \
    --num_offline_steps 200000 \
    --checkpoint_dir runs/cql_pretrain \
    --buffer_device cuda

# Pure offline Cal-QL
python examples/pretrain_offline.py \
    --algorithm calql \
    --offline_dataset_path /path/to/real_robot.h5 \
    --num_offline_steps 200000 \
    --checkpoint_dir runs/calql_pretrain \
    --buffer_device cuda \
    --use_calql --cql_alpha 5.0

# Equivalent launchers
scripts/pretrain_cql_offline.sh --offline_dataset_path /path/to/real_robot.h5
scripts/pretrain_calql_offline.sh --offline_dataset_path /path/to/real_robot.h5
```

These write `cql_offline_pretrained.pt` or `calql_offline_pretrained.pt` by
default. The script infers obs/action specs from the H5, constructs an
`OfflineEnvSpec`, loads the dataset into the algorithm replay buffer, and runs
`run_offline_pretraining()`.

For WSRL-specific offline pretraining, use `--algorithm wsrl`. It builds a
`WSRL` agent directly (Cal-QL by definition) and is useful when the checkpoint
will be resumed by WSRL's offline→online flow:

```bash
python examples/pretrain_offline.py \
    --algorithm wsrl \
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

`--algorithm wsrl-calql` is a deprecated alias for `--algorithm wsrl`; it still
works but emits a `DeprecationWarning`. `examples/pretrain_cql_offline.py` and
`examples/pretrain_wsrl_offline.py` remain as thin compatibility wrappers for
older commands.

**Online fine-tune on a deployment machine** (which does have an env):

```bash
python examples/train_wsrl.py \
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
- `net_arch` is the primary network config interface for `SAC/RGBDSAC/WSRL/WSRLRGBD`.
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
- `--backup_entropy`: Include entropy in TD target backups (default: False, matching upstream WSRL/Cal-QL)

### Cal-QL Parameters
- `--use_calql`: Enable Cal-QL lower bounds (default: True)
- `--calql_bound_random_actions`: Apply bounds to random actions (default: False)

### Offline→Online Control
- `--num_offline_steps 100000`: Number of offline training steps
- `--offline_dataset_path demos/foo.h5`: ManiSkill trajectory H5 path for offline pre-training
- `--offline_num_traj`: Optional number of trajectories to load from the H5
- `--num_online_steps 50000`: Number of online training steps
- `--online_cql_alpha 0.5`: CQL alpha for online phase (optional)
- `--online_use_cql_loss False`: Disable CQL loss for online phase (optional)

### Standalone Offline CQL/Cal-QL Control
- `--algorithm cql`: Use `CQL` with a normal tensor replay buffer.
- `--algorithm calql`: Use `CalQL` with an MC replay buffer and Cal-QL bounds.
- `--algorithm wsrl`: Build `WSRL` for checkpoints intended for the WSRL
  offline→online flow. `wsrl-calql` is accepted as a deprecated alias.
- `--agent cql|calql`: Backward-compatible alias accepted by the CQL wrapper.
- `--save_filename`: Override the default
  `<algorithm>_offline_pretrained.pt` checkpoint name.
- `--offline_sampling without_replace`: Sample offline batches without repeating
  until the static replay buffer is exhausted.

### Upstream-Parity Defaults
- `policy_lr=1e-4`, `q_lr=3e-4`, `alpha_lr=1e-4`
- `gamma=0.99`, `tau=0.005`
- WSRL actor/critic MLPs use layer norm by default
- Actor std parameterization supports `exp` and `uniform`

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
python examples/pretrain_offline.py \
    --algorithm wsrl \
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
  otherwise leave a stale specialization in the graph.
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

### Vision-Based WSRLRGBD

```python
from rl_garden.algorithms import WSRLRGBD
from rl_garden.encoders import default_image_encoder_factory

# Create environment with RGB observations
env_cfg = ManiSkillEnvConfig(
    env_id="PickCube-v1",
    num_envs=16,
    obs_mode="rgb",
    include_state=True,
)
env = make_maniskill_env(env_cfg)

# Create WSRLRGBD agent
agent = WSRLRGBD(
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
│   └── RGBDSAC
└── _CQLRolloutTrainingShell(CQLCore)
    └── _CalQLRolloutTrainingShell
        └── WSRL
            └── WSRLRGBD

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
   - Inherits `CalQL`
   - Offline→online mode switching
   - Empty/append/mixed replay modes
   - Offline probe and WSRL phase logging

7. **WSRLRGBD** (`rl_garden/algorithms/wsrl_rgbd.py`)
   - Vision-based variant
   - Encoder detachment on actor path
   - Dict observation support

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
- Set `--online_cql_alpha 0.5` or `--online_use_cql_loss False` for online phase
- Use `--utd 1.0` for state-based, `--utd 0.25` for vision-based training
- Enable `--use_calql` for better offline pre-training with Cal-QL bounds

## License

See LICENSE file in the repository root.
