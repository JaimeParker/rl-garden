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

- **Online RL:** SAC, PPO, DrQ-v2, and FlashSAC.
- **Offline RL and imitation:** BC, IQL, CQL, Cal-QL, and offline SAC-family
  pretraining.
- **Offline-to-online:** WSRL pretraining, warm start, and online fine-tuning.
- **Observations:** flat state tensors and dict observations containing RGB, depth,
  proprioception, or mixed vector inputs.
- **Visual encoders:** PlainConv, ResNet, and ViT backbones with configurable image-key
  fusion, pooling, augmentation, and proprioception fusion.
- **Replay:** tensor, dict, Monte-Carlo return, and PPO rollout buffers with
  explicit storage and sample devices.
- **Environment backends:** a registry-based interface with current ManiSkill and
  RoboTwin implementations and support for adding further platforms.
- **Robot integration:** EE twist/impedance control, teleoperation, demonstration
  recording, and learned reward classifiers.

## Project Layout

```text
rl_garden/
├── algorithms/    # Online, offline, and off-to-online algorithms
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
docs/              # Guides, design docs, and roadmaps (see docs/README.md)
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
| Online | `examples/train_online.py` | `sac`, `ppo`, `drqv2`, `flash_sac`, `td3`, `rlpd`, `rlpd_hybrid`, `tdmpc2`, recurrent and transformer SAC/PPO |
| Offline | `examples/pretrain_offline.py` | `bc`, `iql`, `cql`, `calql`, `wsrl`, `awac`, `td3_bc`, `tdmpc2_multitask` |
| Offline-to-online | `examples/train_off2on.py` | `wsrl`, `calql`, `iql`, `awac` |

All registry-managed entrypoints accept `--config PRESET.yaml`, `--print-config`,
`--dry-run`, and `--explain-param FIELD`. Static printing does not load a
simulator; dry-run materializes the selected environment and agent but never
trains. Normal runs atomically save the same versioned effective configuration
under `{log_dir}/{run_name}/config.json`. See the
[configuration guide](docs/guides/configuration.md).

### Online Training

State SAC with a reusable preset:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml
```

Visual SAC and PPO:

```bash
python examples/train_online.py sac \
  --env-id PickCube-v1 --obs-mode rgb --encoder plain_conv

python examples/train_online.py ppo \
  --env-id PickCube-v1 --obs-mode rgb --encoder plain_conv
```

Additional preset-backed entrypoints include:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_rgb_resnet.yaml
python examples/train_online.py ppo \
  --config configs/online/ppo_state.yaml
python examples/train_online.py ppo \
  --config configs/online/ppo_rgb.yaml
python examples/train_online.py drqv2 \
  --config configs/online/drqv2_rgb.yaml
```

### Offline Pretraining

Offline training reads a flat or dict-observation H5 dataset without creating a
simulator unless `--env_id` is supplied for evaluation:

```bash
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 \
  --num_offline_steps 700000

python examples/pretrain_offline.py iql \
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

python examples/train_off2on.py wsrl \
  --config configs/off2on/wsrl.yaml
python examples/train_off2on.py wsrl \
  --config configs/off2on/wsrl_rgb.yaml
```

See [Reproducing WSRL](docs/guides/wsrl-reproduction.md) for the complete checkpoint,
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

See [RoboTwin Integration](docs/guides/robotwin.md) for installation, assets, observation
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
python examples/train_online.py sac \
  --env-id PegInsertionSidePegOnly-v1 \
  --obs-mode rgb \
  --include-state \
  --per-camera-rgbd \
  --image-fusion-mode per_key \
  --encoder vit
```

`--freeze-resnet-backbone` keeps the stem and residual blocks fixed while leaving
the pooling/bottleneck head trainable. `--freeze-resnet-encoder` freezes the full
visual extractor. For dict observations, SAC shares the encoder between actor and
critic; actor updates detach encoder features while critic updates train it.

Torchvision-style ResNet checkpoints must be converted to rl-garden parameter names:

```bash
python tools/conversion/convert_resnet_checkpoint.py \
  --input pretrained/resnet/resnet10_pretrained.pt \
  --output pretrained/resnet/resnet10_pretrained_converted.pt \
  --arch resnet10
```

## Checkpoints

Checkpoints are torch-native `.pt` dictionaries containing model, optimizer, and
training state. Replay snapshots are optional separate files; save them when exact
off-policy continuation requires preserving replay distribution.

See [Checkpoint Save & Load](docs/guides/checkpoint.md) for default paths, resume commands,
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
- [Teleoperation and recording](docs/guides/teleop.md)

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

See [docs/README.md](docs/README.md) for the full index. Highlights:

- [Checkpoint Save & Load](docs/guides/checkpoint.md)
- [Configuration System](docs/guides/configuration.md)
- [Offline Training Acceleration](docs/guides/offline-acceleration.md)
- [RoboTwin Integration](docs/guides/robotwin.md)
- [Teleoperation and Recording](docs/guides/teleop.md)
- [WSRL Reproduction](docs/guides/wsrl-reproduction.md)
- [WSRL Overview](docs/design/wsrl-overview.md)
- [RNG and Numerical Determinism](docs/design/rng-numerical-determinism.md)

## Research Influences

The implementation combines ideas and engineering patterns from multiple projects
rather than treating any single framework as its template. Reference implementations
are kept under `3rd_party/` and include ManiSkill, stable-baselines3, hil-serl, WSRL,
Cal-QL, RLPD/RLinf, BPPO, Uni-O4, TDMPC2, and robot-controller projects. Treat these
directories as read-only unless a change is explicitly requested.
