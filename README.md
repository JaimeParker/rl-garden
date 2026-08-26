<p align="center">
  <img src="docs/assets/logo.png" alt="rl-garden logo" width="360">
</p>

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

- **Online RL:** SAC, PPO, RLPD, RLPD-Hybrid, TD3, DrQ-v2, FlashSAC, TDMPC2,
  DPPO, SACFlow, ACRLPD, plus recurrent and transformer SAC/PPO variants. See
  the [Training Entrypoints](#training-entrypoints) table for the full list.
- **Offline RL and imitation:** BC, Diffusion BC, IQL, CQL, Cal-QL, BCQ, PLAS,
  EDAC, SPOT, ReBRAC, FQL, QGF, QAM, TD3+BC, AWAC, and multitask TDMPC2.
- **Offline-to-online:** WSRL, Cal-QL, IQL, AWAC, SPOT, and ACFQL pretraining,
  warm start, and online fine-tuning.
- **Observations:** flat state tensors and dict observations containing RGB, depth,
  proprioception, or mixed vector inputs.
- **Visual encoders:** PlainConv, ResNet, DrQ-v2 conv, 3D CNN, and ViT backbones
  with configurable image-key fusion, pooling, augmentation, and proprioception
  fusion. Actor and critic share one encoder by default; SAC-family and PPO-family
  policies can opt into independent actor/critic encoder architectures.
- **Replay:** tensor, dict, Monte-Carlo return, and PPO rollout buffers with
  explicit storage and sample devices.
- **Environment backends:** a registry-based interface with ManiSkill, RoboTwin,
  IsaacLab, MuJoCo, MuJoCo Warp (GPU), Minari, legacy D4RL/Adroit/Kitchen, a
  real-robot Franka backend, and a template for adding further platforms.
- **Robot integration:** EE twist/impedance control, teleoperation, demonstration
  recording, and learned reward classifiers.
- **Distributed training:** optional RLinf/Ray adapters for offline, async
  off-policy (SAC/RLPD), and FSDP on-policy (PPO) training.

## Project Layout

```text
rl_garden/
├── algorithms/    # Online, offline, and off-to-online algorithms
├── buffers/       # Replay and rollout buffers
├── common/        # Logging, CLI/env args, checkpoints, optimizers, utilities
├── datasets/      # Offline and WSRL dataset workflows
├── encoders/      # State, CNN, ResNet, RGBD/proprio, pooling, augmentation
├── envs/          # Environment backend registry, implementations, wrappers
├── integrations/  # Optional RLinf/Ray adapters (offline, SAC/RLPD, FSDP PPO)
├── models/        # ACT and reward models
├── networks/      # Actor, critic, value, and backbone modules
├── policies/      # Algorithm policy composition
└── training/      # Registered online, offline, and off2on training packages
robot_infra/       # Optional submodule (rlgarden-robot-infra): controllers, teleoperation
real_world/        # Optional submodule (rlgarden-real-world): ActorLoop/LearnerLoop, franka_real backend
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
| Online | `examples/train_online.py` | `sac`, `ppo`, `drqv2`, `flash_sac`, `td3`, `rlpd`, `rlpd_hybrid`, `tdmpc2`, `dppo`, `sac_flow`, `acrlpd`, `recurrent_sac`, `recurrent_ppo`, `transformer_sac`, `transformer_ppo`, `dagger`, `policy_distillation` |
| Offline | `examples/pretrain_offline.py` | `bc`, `diffusion_bc`, `vision_diffusion_bc`, `flow_bc`, `iql`, `cql`, `calql`, `bcq`, `plas`, `edac`, `spot`, `rebrac`, `fql`, `qgf`, `qam`, `wsrl`, `awac`, `td3_bc`, `tdmpc2_multitask` |
| Offline-to-online | `examples/train_off2on.py` | `wsrl`, `calql`, `iql`, `awac`, `spot`, `acfql` |

Every registered algorithm's exact args and defaults can be listed with
`--print-config`; use `python examples/train_online.py --help` (or
`pretrain_offline.py` / `train_off2on.py`) to see the current algorithm list
directly from the registry instead of relying on this table staying in sync.

**"Online" means "needs live env rollout during training," not "is
reward-driven RL."** Most entries under `train_online.py` optimize a reward
signal (`sac`, `ppo`, `rlpd`, ...), but `dagger` and `policy_distillation` are
imitation learning: their loss is supervised regression against an expert/
teacher action, not reward. They live under `train_online.py` (reusing its
env-construction, logging, and checkpoint machinery) purely because, unlike
`bc`/`diffusion_bc`/`flow_bc` under `pretrain_offline.py`, they need to
collect on-policy rollouts to see the state distribution the trained policy
will actually encounter -- that's an infrastructure requirement, not a claim
that they're RL. `policy_distillation` in particular trains no critic and
never reads a reward from the env at all; see its module docstring
(`rl_garden/algorithms/policy_distillation.py`) for the teacher/student split.

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
  --offline_dataset demos/pickcube.h5 \
  --num_offline_steps 700000

python examples/pretrain_offline.py iql \
  --offline_dataset demos/pickcube.h5
```

BC, IQL, CQL, and FQL support dict observations containing image and state
inputs. Cal-QL, WSRL, and the remaining offline algorithms currently use flat
state datasets for their standard offline workflow.

### Offline-to-Online Training

```bash
python examples/train_off2on.py wsrl \
  --env_id PickCube-v1 \
  --offline_dataset demos/pickcube.h5

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

Other registered `--env-backend` values: `maniskill` (default for most examples),
`isaaclab` (see [IsaacLab Custom Tasks](docs/guides/isaaclab-custom-tasks.md) and the
[known camera-stall issue](docs/guides/isaaclab-camera-stall.md)), `mujoco`,
`mujoco_warp` (GPU-vectorized MuJoCo), `minari`, and `d4rl_legacy` (legacy
Gym/D4RL Adroit and Kitchen tasks, see
[D4RL Legacy Manipulation Baselines](docs/guides/d4rl-legacy-expansion.md)).
`franka_real` is added by the optional `rlgarden-real-world` submodule for
real-robot training.

`rl_garden/envs/custom/` is a runnable template for authoring a brand-new
environment that isn't wrapping an existing simulator (`--env-backend custom
--env-id PointReach-v0`). Copy it to start your own backend; see
`.agents/rules/adding-env-backend.md` for the full contract.

## Visual Training

Use `--encoder plain_conv` for the lightweight CNN path, a ResNet name such as
`--encoder resnet10`/`resnet18`, `--encoder drqv2_conv` for DrQ-v2's conv stack,
`--encoder cnn3d` for volumetric/stacked-frame input, or `--encoder vit` for the
ViT path. Image keys can be fused in two ways:

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
  --env-id PickCube-v1 \
  --obs-mode rgb \
  --image-fusion-mode per_key \
  --encoder vit
```

`--freeze-resnet-backbone` keeps the stem and residual blocks fixed while leaving
the pooling/bottleneck head trainable. `--freeze-resnet-encoder` freezes the full
visual extractor. By default, actor and critic (SAC-family and PPO-family) share
one encoder instance; actor updates detach encoder features while critic/value
updates train it. `--critic-encoder <name>` opts a policy into an independently
architected critic encoder (e.g. an MLP critic over state paired with a ResNet
actor over images) — when set, the actor's own encoder trains through the actor
loss instead of relying on the critic's gradient. This is an advanced, opt-in
path; leaving it unset keeps today's shared-encoder behavior unchanged.

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

Robot hardware/communication/control and real-world RL orchestration live in
two separate repos, added back as optional git submodules:

- [rlgarden-robot-infra](https://github.com/JaimeParker/rlgarden-robot-infra) --
  EE twist and impedance controllers, the Franka bridge, teleop devices.
- [rlgarden-real-world](https://github.com/JaimeParker/rlgarden-real-world) --
  `ActorLoop`/`LearnerLoop`, the `franka_real` env backend, SERL/HIL-SERL
  training loops, teleoperation demo/recording scripts, and the [teleop
  guide](https://github.com/JaimeParker/rlgarden-real-world/blob/main/docs/teleop.md).

Neither is required for simulation-only use; `git submodule update --init` to
pull them in.

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
- [D4RL Legacy Manipulation Baselines](docs/guides/d4rl-legacy-expansion.md)
- [IsaacLab Camera Training Stall (known issue)](docs/guides/isaaclab-camera-stall.md)
- [IsaacLab Custom Tasks](docs/guides/isaaclab-custom-tasks.md)
- [Offline Training Acceleration](docs/guides/offline-acceleration.md)
- [RoboTwin Integration](docs/guides/robotwin.md)
- [Teleoperation and Recording](https://github.com/JaimeParker/rlgarden-real-world/blob/main/docs/teleop.md) (in `rlgarden-real-world`)
- [WSRL Reproduction](docs/guides/wsrl-reproduction.md)
- [IQL Overview](docs/design/iql-overview.md)
- [RLinf Integration](docs/design/rlinf-integration.md)
- [WSRL Overview](docs/design/wsrl-overview.md)
- [RNG and Numerical Determinism](docs/design/rng-numerical-determinism.md)

## Research Influences

The implementation combines ideas and engineering patterns from multiple projects
rather than treating any single framework as its template. Reference implementations
are kept under `3rd_party/`: `Cal-QL`, `wsrl`, and `implicit_q_learning` are real git
submodules (see [Baseline Install](.agents/runbooks/baseline-install.md)); `CORL`,
`NexRL`, `RL-100`, `RLinf`, `dppo`, `fql`, `hil-serl`, `qam`, `qc`, `qgf`, and
`stable-baselines3` are untracked read-only reference clones. Treat all of these
directories as read-only unless a change is explicitly requested.
