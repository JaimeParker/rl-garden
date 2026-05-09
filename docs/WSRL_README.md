# WSRL Implementation Summary

## Overview

This repository now includes a complete PyTorch implementation of **WSRL (Warm-Start Reinforcement Learning)** with **Cal-QL (Calibrated Conservative Q-Learning)** and **REDQ (Randomized Ensembled Double Q-learning)**.

WSRL enables efficient offline→online training:
- **Offline phase**: Pre-train with Cal-QL on ManiSkill trajectory H5 datasets
- **Online phase**: Fine-tune with SAC or CQL without retaining offline data

For the end-to-end PickCube reproduction workflow, including SAC checkpoint
training, WSRL dataset generation, and offline-to-online launch commands, see
[`WSRL_REPRODUCTION.md`](WSRL_REPRODUCTION.md).

## Key Features

### ✅ Algorithms Implemented
- **WSRL**: State-based WSRL with CQL/Cal-QL
- **WSRLRGBD**: Vision-based WSRL for RGB/RGBD observations

### ✅ Core Components
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
OffPolicyAlgorithm (base class)
├── SAC (state-based SAC)
│   └── RGBDSAC (vision-based SAC)
└── WSRL (state-based WSRL with CQL/Cal-QL)
    └── WSRLRGBD (vision-based WSRL)
```

### Key Components

1. **WSRLPolicy** (`rl_garden/policies/wsrl_policy.py`)
   - Q-ensemble with configurable size
   - Critic subsampling for REDQ
   - Optional CQL alpha Lagrange multiplier
   - Upstream-style layer norm and actor std parameterization options
   - Uses shared network modules from `rl_garden/networks/`

2. **MCReplayBuffer** (`rl_garden/buffers/mc_buffer.py`)
   - Cached vectorized Monte Carlo return computation
   - Episode boundary tracking
   - Circular-buffer wraparound handling
   - Support for both Tensor and Dict observations

3. **ManiSkill H5 Loader** (`rl_garden/buffers/maniskill_h5.py`)
   - Loads `traj_*` H5 groups into existing MC replay buffers
   - Supports flat state observations and dict/RGBD observation groups

4. **WSRL Algorithm** (`rl_garden/algorithms/wsrl.py`)
   - CQL regularization with OOD action sampling
   - Cal-QL lower bounds using MC returns
   - Offline→online mode switching
   - High-UTD training support

5. **WSRLRGBD** (`rl_garden/algorithms/wsrl_rgbd.py`)
   - Vision-based variant
   - Encoder detachment on actor path
   - Dict observation support

## Test Coverage

All WSRL components have focused tests covering policy options, CQL/Cal-QL loss
semantics, high-UTD dispatch, MC replay returns, RGBD support, and ManiSkill H5
loading.

Run tests:
```bash
pytest tests/test_wsrl*.py tests/test_mc*.py tests/test_maniskill_h5_loader.py -v
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
