# rl-garden

`rl-garden` is a PyTorch-native robot-learning framework for online RL, offline RL,
imitation learning, and offline-to-online training. It provides reusable algorithm,
policy, encoder, replay-buffer, and environment-backend components for simulation,
offline datasets, and real-robot systems.

The framework keeps rollout, replay, inference, and update paths on torch tensors,
with GPU-vectorized execution as the preferred training path. Environment backends
are registered independently from algorithms so additional simulators and robot
platforms can be integrated without creating platform-specific training entrypoints.

## Capabilities

- **Online RL:** SAC, PPO, DrQ-v2, FlashSAC, and ResidualSAC.
- **Offline RL and imitation:** BC, IQL, CQL, Cal-QL, and offline SAC-family
  pretraining.
- **Offline-to-online:** WSRL pretraining, warm start, and online fine-tuning.
- **Observations:** flat state tensors and dict observations containing RGB, depth,
  proprioception, or mixed vector inputs.
- **Visual encoders:** PlainConv, ResNet, and ViT backbones with configurable image-key
  fusion, pooling, augmentation, and proprioception fusion.
- **Replay:** tensor, dict, Monte-Carlo return, residual-action, and PPO rollout
  buffers with explicit storage and sample devices.
- **Environment backends:** a registry-based interface with current ManiSkill and
  RoboTwin implementations and support for adding further platforms.
- **Robot integration:** EE twist/impedance control, teleoperation, demonstration
  recording, ACT base policies, and learned reward classifiers.

## Project Layout

```text
rl_garden/
├── algorithms/    # Online, offline, off-to-online, and residual algorithms
├── buffers/       # Replay and rollout buffers
├── common/        # Logging, CLI/env args, checkpoints, optimizers, utilities
├── datasets/      # Offline and WSRL dataset workflows
├── encoders/      # State, CNN, ResNet, RGBD/proprio, pooling, augmentation
├── envs/          # Environment backend registry, implementations, wrappers
├── models/        # ACT and reward models
├── networks/      # Actor, critic, value, and backbone modules
├── policies/      # Algorithm policy composition
└── training/      # Registered online, offline, and off2on training packages
robot_infra/       # Controllers, teleoperation, and real-robot utilities
examples/          # Thin dispatchers and specialized experiment entrypoints
scripts/           # Launchers with experiment defaults
tests/             # Unit and backend/accelerator integration tests
docs/              # Workflow and subsystem documentation
3rd_party/         # Read-only research references and external projects
```

## Installation

Clone the repository and initialize its submodules:

```bash
git clone <your-repo-url>
cd rl-garden
git submodule update --init --recursive
```

Install the package and the extras needed for your workflow:

```bash
pip install -e .
pip install -e ".[dev]"          # pytest and development tools
pip install -e ".[maniskill]"    # ManiSkill backend dependencies
pip install -e ".[wandb]"        # Weights & Biases logging
```

Other environment backends may require their own runtime and assets. See the
backend-specific documentation before launching a run.

## Training Entrypoints

Training is organized around three registry dispatchers. The first positional
argument selects the algorithm:

| Stage | Entrypoint | Registered algorithms |
|---|---|---|
| Online | `examples/train_online.py` | `sac`, `ppo`, `drqv2`, `flash_sac`, `residual_sac` |
| Offline | `examples/pretrain_offline.py` | `bc`, `iql`, `cql`, `calql`, `wsrl` |
| Offline-to-online | `examples/train_off2on.py` | `wsrl` |

All registry-managed entrypoints accept `--print-config`. It prints the fully
resolved recursive JSON configuration and exits before creating environments,
loggers, or agents. Normal runs save the same configuration under
`{log_dir}/{run_name}/config.json`.

### Online Training

State SAC:

```bash
python examples/train_online.py sac \
  --env-id PickCube-v1 --obs-mode state --num-envs 16

# Launcher with experiment defaults
scripts/train_sac_state.sh
```

Visual SAC and PPO:

```bash
python examples/train_online.py sac \
  --env-id PickCube-v1 --obs-mode rgb --encoder plain_conv

python examples/train_online.py ppo \
  --env-id PickCube-v1 --obs-mode rgb --encoder plain_conv
```

ResidualSAC:

```bash
python examples/train_online.py residual_sac \
  --env-id PickCube-v1 \
  --obs-mode rgb \
  --control-mode pd_ee_twist \
  --base-policy act \
  --base-ckpt-path act-peg-only \
  --residual-action-scale 0.1
```

ResidualSAC-specific options include `--base-policy {act,sac,zero}`,
`--base-ckpt-path`, `--residual-action-scale`, ACT temporal aggregation controls,
base-SAC reconstruction options, and residual demo loading via
`--offline-dataset-path` / `--offline-data-ratio`. Demo loading is currently
supported only by `residual_sac`.

Additional launchers include:

```bash
scripts/train_sac_rgbd.sh --encoder resnet10
scripts/train_ppo_state.sh
scripts/train_ppo_rgbd.sh --encoder plain_conv
scripts/train_drqv2_rgb.sh
scripts/train_residual_rgbd.sh
scripts/train_residual_state.sh
```

### Offline Pretraining

Offline training reads a flat or dict-observation H5 dataset without creating a
simulator unless `--env_id` is supplied for evaluation:

```bash
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 \
  --num_offline_steps 700000

scripts/pretrain_offline.sh iql \
  --offline_dataset_path demos/pickcube.h5
```

BC and IQL support dict observations containing image and state inputs. CQL,
Cal-QL, and WSRL currently use flat state datasets for their standard offline
workflow.

### Offline-to-Online Training

```bash
python examples/train_off2on.py wsrl \
  --env_id PickCube-v1 \
  --offline_dataset_path demos/pickcube.h5

scripts/train_wsrl.sh
scripts/train_wsrl_rgbd.sh
```

See [Reproducing WSRL](docs/WSRL_REPRODUCTION.md) for the complete checkpoint,
dataset-generation, offline-pretraining, and online-fine-tuning workflow.

### Environment Backends

Training algorithms select an environment through `--env-backend`; backend-specific
arguments use a nested namespace. For example, PPO on RoboTwin:

```bash
python examples/train_online.py ppo \
  --env-backend robotwin \
  --env-id place_empty_cup \
  --obs-mode rgb \
  --robotwin.robotwin-root /path/to/RoboTwin
```

See [RoboTwin Integration](docs/ROBOTWIN.md) for installation, assets, observation
mapping, rewards, and performance controls.

The peg-insertion environment has dedicated camera, controller, and robot defaults
that can be passed through the unified training entrypoint:

```bash
python examples/train_online.py sac \
  --env-id PegInsertionSidePegOnly-v1 \
  --obs-mode rgb \
  --control-mode pd_ee_delta_pose \
  --maniskill.reward-mode normalized_dense \
  --maniskill.robot-uids panda_wristcam_gripper_closed_wo_norm \
  --per-camera-rgbd \
  --image-fusion-mode per_key
```

Residual peg experiments use the same env flags with `residual_sac` through
`scripts/train_residual_rgbd.sh` or `scripts/train_residual_state.sh`.

## Visual Training

Use `--encoder plain_conv` for the lightweight CNN path, a ResNet name such as
`--encoder resnet10`, or `--encoder vit` for the ViT path. Image keys can be fused
in two ways:

- `stack_channels`: concatenate visual keys before a single encoder. This is the
  default and the simplest path for a single RGB stream.
- `per_key`: encode each visual key independently and concatenate features. Use it
  for multi-camera observations and pretrained three-channel backbones.

Example with a pretrained ResNet backbone:

```bash
python examples/train_online.py sac \
  --env-id PickCube-v1 \
  --obs-mode rgb \
  --encoder resnet10 \
  --image-fusion-mode per_key \
  --pretrained-weights resnet10-imagenet \
  --freeze-resnet-backbone
```

ViT example:

```bash
python examples/train_online.py residual_sac \
  --env-id PegInsertionSidePegOnly-v1 \
  --obs-mode rgb \
  --include-state \
  --per-camera-rgbd \
  --image-fusion-mode per_key \
  --encoder vit \
  --base-policy act \
  --base-ckpt-path act-peg-only
```

`--freeze-resnet-backbone` keeps the stem and residual blocks fixed while leaving
the pooling/bottleneck head trainable. `--freeze-resnet-encoder` freezes the full
visual extractor. For dict observations, SAC shares the encoder between actor and
critic; actor updates detach encoder features while critic updates train it.

Torchvision-style ResNet checkpoints must be converted to rl-garden parameter names:

```bash
python scripts/convert_resnet_checkpoint.py \
  --input pretrained_models/resnet10_pretrained.pt \
  --output pretrained_models/resnet10_pretrained_converted.pt \
  --arch resnet10
```

## Checkpoints

Checkpoints are torch-native `.pt` dictionaries containing model, optimizer, and
training state. Replay snapshots are optional separate files; save them when exact
off-policy continuation requires preserving replay distribution.

See [Checkpoint Save & Load](docs/CHECKPOINT.md) for default paths, resume commands,
replay-buffer tradeoffs, and algorithm compatibility.

## Library Composition

Algorithms accept either a shared MLP layout or separate policy/value layouts:

```python
from rl_garden.algorithms import BC, IQL, SAC, WSRL

sac = SAC(env=env, net_arch=[256, 256, 256])
wsrl = WSRL(env=env, net_arch={"pi": [256, 256], "qf": [256, 256]})
iql = IQL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256], "vf": [256, 256]},
)
bc = BC(env=env, net_arch=[256, 256])
```

Policies accept custom extractors through `policy_kwargs`:

```python
from rl_garden.algorithms import SAC
from rl_garden.encoders import CombinedExtractor, resnet_encoder_factory

agent = SAC(
    env=env,
    policy_kwargs={
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {
            "image_keys": ("rgb",),
            "image_encoder_factory": resnet_encoder_factory("resnet10"),
            "fusion_mode": "per_key",
        },
    },
)
```

Box observations select flatten/tensor components; dict observations select the
combined extractor and dict replay path.

## Robot Infrastructure and Reward Models

`robot_infra/` contains EE twist and impedance controllers, teleoperation, and
demonstration-recording utilities. See:

- [Controller setup](robot_infra/controller/README.md)
- [Teleoperation and recording](docs/TELEOP_README.md)

Learned reward utilities live under `rl_garden/models/reward/`. Typical entrypoints
include:

```bash
python rl_garden/models/reward/classifiers/hsv/generate_labels.py \
  --data_dir data/epi0-19_trimmed --tune_hsv --camera high
python rl_garden/models/reward/classifiers/color/train.py
python rl_garden/models/reward/classifiers/alignment/train.py
```

## Testing

Run the available test suite from the repository root:

```bash
pytest tests -q
```

During development, start with the smallest relevant tests. Examples:

```bash
pytest -q tests/test_training_registry.py tests/test_cli_args.py
pytest -q tests/test_checkpoint.py
pytest -q tests/test_replay_buffer.py tests/test_mc_buffer.py
```

Backend and accelerator smoke tests require their corresponding optional runtime and
hardware. If those dependencies are unavailable, report the skipped or failed check
rather than changing the framework's preferred device path.

## Documentation

- [Checkpoint Save & Load](docs/CHECKPOINT.md)
- [Offline Training Acceleration](docs/OFFLINE_ACCELERATION.md)
- [Residual SAC](docs/RESIDUAL_SAC.md)
- [RNG and Numerical Determinism](docs/RNG_AND_NUMERICAL_DETERMINISM.md)
- [RoboTwin Integration](docs/ROBOTWIN.md)
- [Teleoperation and Recording](docs/TELEOP_README.md)
- [WSRL Overview](docs/WSRL_README.md)
- [WSRL Reproduction](docs/WSRL_REPRODUCTION.md)

## Research Influences

The implementation combines ideas and engineering patterns from multiple projects
rather than treating any single framework as its template. Reference implementations
are kept under `3rd_party/` and include ManiSkill, stable-baselines3, hil-serl, WSRL,
Cal-QL, RLPD/RLinf, BPPO, Uni-O4, TDMPC2, and robot-controller projects. Treat these
directories as read-only unless a change is explicitly requested.
