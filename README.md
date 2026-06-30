# rl-garden

SB3-style, PyTorch-native, GPU-parallel robot learning framework supporting online RL, offline RL, and imitation learning on ManiSkill and RoboTwin.

`rl-garden` organizes a broad set of robot learning algorithms — SAC, PPO, CQL, Cal-QL, IQL, BC, WSRL, ResidualSAC — into reusable `Algorithm / Policy / FeaturesExtractor` modules, keeping the full rollout, replay, and update path on torch tensors with no numpy hot-path conversion. It targets GPU-vectorized simulators (ManiSkill, RoboTwin) and real-robot deployment via an integrated pd_ee_twist controller and teleoperation stack.

## Goals

- Support the full offline→online robot learning pipeline: offline pretraining (BC/IQL/CQL/Cal-QL), warm-start fine-tuning (WSRL), and online RL (SAC/PPO/ResidualSAC).
- Keep ManiSkill GPU vectorization (`ManiSkillVectorEnv`) first-class; expose the same algorithm/encoder interface to RoboTwin and future simulators.
- Follow SB3 design patterns (`Algorithm` / `Policy` / `FeaturesExtractor`) without importing `stable_baselines3`.
- Make visual pipelines pluggable: `PlainConv` and hil-serl-inspired `ResNetV1` + spatial pooling + proprio fusion.
- Make new algorithms and environments incremental via subclassing and composition, with no changes to unrelated base classes.

## What Is Already Implemented

**Online algorithms:**
- `SAC` for both flat Box state observations and Dict image/image+state observations.
- `SACCore`, shared by online SAC, offline SAC, CQL, Cal-QL, and WSRL.
- `PPO` for both flat Box state observations and Dict image observations.
- `ResidualSAC`: adds a learned residual on top of a frozen base policy (e.g. ACT); supports state and RGBD obs.

**Offline algorithms:**
- `IQL` (Implicit Q-Learning): offline actor training via expectile regression + AWR.
- `CQL` / `CalQL` as standalone offline pretraining algorithms.
- `BC` (Behavioral Cloning): actor-only supervised imitation baseline.
- `OfflineSAC`: offline-only SAC variant that mirrors CQL update defaults without conservative penalty.
- `WSRL` as the Cal-QL-based offline→online warm-start flow layer.

**GPU-native replay buffers:**
- `TensorReplayBuffer` for Box observations.
- `DictReplayBuffer` for `{rgb, depth, state, ...}` observations.
- `MCTensorReplayBuffer` / `MCDictReplayBuffer`: Monte-Carlo return variants (used by CalQL/WSRL).
- `ResidualTensorReplayBuffer` / `ResidualDictReplayBuffer`: store base + residual actions for ResidualSAC.
- `RolloutBuffer` / `DictRolloutBuffer`: on-policy rollout storage for PPO.

**Encoder stack:**
- `FlattenExtractor`
- `PlainConvEncoder`
- `CombinedExtractor` (image + proprio fusion)
- `ResNetEncoder` (`resnet10`/`resnet18`/`resnet34`) + pooling heads (`AvgPool`, `SpatialSoftmax`, `SpatialLearnedEmbeddings`)
- `RandomCrop` data augmentation layer

**Networks:**
- `SquashedGaussianActor` (tanh-squashed, used by SAC/CQL/IQL/BC/WSRL families)
- `DiagGaussianActor` (unsquashed, used by PPO)
- `EnsembleQCritic` (REDQ-style Q-ensemble)
- `ValueNetwork` (V-function for IQL/PPO)
- `MLPResNet` (ResNet-style MLP backbone)

**Simulator env factories:**
- **ManiSkill** (`ManiSkillEnvConfig` + `make_maniskill_env()`): GPU-vectorized ManiSkill envs with state/rgb/rgbd wrappers, custom Panda agents, and peg-insertion tasks.
- **RoboTwin** (`RoboTwinEnvConfig` + `make_robotwin_env()`): RoboTwin simulator adapter supporting SAC and PPO training on manipulation tasks (e.g. `place_shoe`, `place_empty_cup`).

**Python examples** for state SAC, RGB SAC, PPO (ManiSkill + RoboTwin), ResidualSAC, WSRL, and offline pretraining (CQL/CalQL/IQL/BC).

**Unit tests + CUDA smoke tests** (26 test files covering all major components).

**ACT base action provider** for ResidualSAC (full DETR-based implementation under `rl_garden/models/act/`).

**Robot infrastructure**: GPU-side pd_ee_twist + impedance controller and teleoperation recording for real-robot WSRL workflows.

## Project Layout

```text
rl-garden/
├── 3rd_party/                 # reference submodules (read-only)
│   ├── ManiSkill
│   ├── stable-baselines3
│   ├── hil-serl
│   ├── wsrl
│   ├── BPPO
│   ├── Uni-O4
│   ├── tdmpc2
│   ├── Cal-QL
│   ├── rlpd / RLinf
│   ├── Cartesian-Impedance-Controller
│   └── serl_franka_controllers
├── rl_garden/
│   ├── algorithms/            # SAC, PPO, CQL, CalQL, IQL, BC, WSRL, ResidualSAC, OfflineSAC
│   ├── buffers/               # Tensor/Dict/MC/Residual replay buffers, RolloutBuffer, H5 loader
│   ├── common/                # Logger, CLI args, checkpoint I/O, optimizer factories, types
│   ├── datasets/              # WSRL dataset generation from SAC checkpoints
│   ├── encoders/              # FlattenExtractor, PlainConv, CombinedExtractor, ResNet, augmentation
│   ├── envs/                  # ManiSkill + RobotWin env factories, custom tasks/agents/wrappers
│   ├── models/
│   │   ├── act/               # ACT policy model (DETR backbone), provider interface
│   │   └── reward/            # HSV / color / alignment reward classifiers
│   ├── networks/              # SquashedGaussianActor, DiagGaussianActor, EnsembleQCritic, MLPResNet
│   └── policies/              # SACPolicy, PPOPolicy, IQLPolicy, BCPolicy, WSRLPolicy, ResidualSACPolicy
├── robot_infra/
│   ├── controller/            # GPU-side pd_ee_twist + impedance controller, ManiSkill simulator backend
│   └── teleop/                # EE twist teleoperation interface + WSRL recording scripts
├── examples/                  # Python training entrypoints (SAC, PPO, ResidualSAC, WSRL, offline)
├── scripts/                   # Shell launchers for common training runs
├── tests/                     # CPU unit tests and CUDA smoke tests (26 files)
├── docs/                      # WSRL reproduction guide, ResidualSAC design notes
├── pretrained_models/         # Expected location for ResNet / ACT checkpoints
└── pyproject.toml
```

## Install

1. Clone and initialize submodules:

```bash
git clone <your-repo-url>
cd rl-garden
git submodule update --init --recursive
```

2. Install package:

```bash
pip install -e .
```

3. Optional extras:

```bash
pip install -e ".[dev]"          # pytest
pip install -e ".[maniskill]"    # mani_skill dependency helper
pip install -e ".[wandb]"        # wandb logger backend
```

## Quick Start

### State SAC

```bash
python examples/train_sac_state.py --env_id PickCube-v1 --num_envs 16
```

### State PPO

```bash
python examples/train_ppo_state.py --env_id PickCube-v1 --num_envs 512
scripts/train_ppo_state.sh
```

### RGB PPO

```bash
python examples/train_ppo_rgbd.py --env_id PickCube-v1 --obs_mode rgb --encoder plain_conv
scripts/train_ppo_rgbd.sh
```

### RoboTwin

RoboTwin must be installed or its repository passed via `--robotwin-root`:

```bash
# SAC on RoboTwin RGB (default task: place_shoe)
python examples/train_sac_robotwin_rgbd.py \
  --env-id place_shoe --robotwin-root /path/to/RoboTwin

# PPO on RoboTwin RGB (default task: place_empty_cup)
python examples/train_ppo_robotwin_rgbd.py \
  --env-id place_empty_cup --robotwin-root /path/to/RoboTwin

# Shell launcher for the place_empty_cup PPO run
scripts/train_ppo_robotwin_place_empty_cup_rgbd.sh
```

The RoboTwin adapter uses the same `CombinedExtractor` + `PlainConv`/`ResNet` visual pipeline as ManiSkill
and shares the same `VisionSACTrainingArgs` / `VisionPPOTrainingArgs` CLI surface, so encoder and
replay-buffer flags work identically across both simulators.

Use `--print-config` with any registry-managed training entrypoint to print the
fully resolved recursive JSON configuration and exit. Normal runs save that
configuration to `{log_dir}/{run_name}/config.json`. Explicit CLI flags override
`RLG_*` logging environment defaults.

### Offline Pretraining

Choose the offline algorithm with a required subcommand. Flat Box and Dict (image+state) H5 datasets are supported:

```bash
# BC baseline (actor-only imitation learning)
python examples/pretrain_offline.py bc \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 100000

# IQL (Implicit Q-Learning)
python examples/pretrain_offline.py iql \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 300000

# Cal-QL (default, best for offline→online transfer)
python examples/pretrain_offline.py calql \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 700000

# CQL (pure conservative Q-learning)
python examples/pretrain_offline.py cql \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 700000

# WSRL agent (Cal-QL backbone, for later WSRL offline→online flow)
python examples/pretrain_offline.py wsrl \
  --offline_dataset_path demos/pickcube.h5 --num_offline_steps 100000
```

Dict observations (image+state H5) are supported by the `iql` and `bc` subcommands. Use a flat state dataset for `cql`/`calql`/`wsrl`.

Shell launchers wrap the same entrypoint:

```bash
scripts/pretrain_offline.sh calql --offline_dataset_path demos/pickcube.h5
scripts/pretrain_offline.sh cql --offline_dataset_path demos/pickcube.h5
```

The primary entrypoint provides `cql`, `calql`, `wsrl`, `iql`, and `bc`
subcommands; `scripts/pretrain_offline.sh` wraps it for shell launches. The final checkpoint defaults to
`<algorithm>_offline_pretrained.pt` and can be loaded into compatible
SAC-family agents for later evaluation or fine-tuning.

#### Optional Online Evaluation During Offline Training

Pass `--env_id` to spin up a ManiSkill eval env during offline training for periodic success/return metrics:

```bash
python examples/pretrain_offline.py iql \
  --offline_dataset_path demos/pickcube.h5 \
  --env_id PickCube-v1 --eval_freq 10000 --num_eval_steps 50
```

When `--env_id` is omitted only loss curves are logged; no simulator is started.

End-to-end WSRL reproduction, from SAC checkpoints to dataset generation and
offline-to-online training, is documented in
[`docs/WSRL_REPRODUCTION.md`](docs/WSRL_REPRODUCTION.md).

State SAC with EE twist control:

```bash
python examples/train_sac_state.py --env_id PickCube-v1 --num_envs 16 --control_mode pd_ee_twist
```

RGB SAC (`PlainConv`):

```bash
python examples/train_sac_rgbd.py --env_id PickCube-v1 --obs_mode rgb --encoder plain_conv
```

RGB SAC (`ResNet10`):

```bash
python examples/train_sac_rgbd.py --env_id PickCube-v1 --obs_mode rgb --encoder resnet10
```

Image fusion modes:

- `stack_channels` is the default. It concatenates visual keys along the channel
  dimension before one image encoder, matching the current `PlainConv` path and
  existing CNN experiments.
- `per_key` encodes each image key independently and concatenates encoded
  features, matching hil-serl's `EncodingWrapper` style. Prefer this mode for
  pretrained ResNet and multi-view RGB inputs.
- For single-key `rgb` observations (one camera, no `depth` key), `per_key` and
  `stack_channels` are effectively equivalent at the tensor level: both feed the
  same `B x 3 x H x W` image into a ResNet/CNN. Differences matter when there
  are multiple visual keys (for example `rgb+depth` or multi-camera RGB).
- Multi-camera envs (e.g. `PegInsertionSidePegOnly-v1` with `base_camera` +
  `hand_camera`): ManiSkill's default `FlattenRGBDObservationWrapper` channel-
  stacks all cameras into one `rgb` key, which collapses `per_key` back into a
  single 6-channel encoder and breaks 3-channel ImageNet pretrained weights.
  Pass `--per_camera_rgbd` (or set `per_camera_rgbd=True` on
  `ManiSkillEnvConfig`) to keep each camera as its own `rgb_<camera>` key. The
  `train_sac_rgbd_peg.py` entrypoint enables this by default and pairs it with
  `--image_fusion_mode per_key`.

Recommended visual training settings:

- `PlainConv` baseline: use the defaults, i.e. `--encoder plain_conv` and
  `--image_fusion_mode stack_channels`. This is the simplest and fastest path
  for current PickCube/Peg CNN experiments.
- ResNet from scratch: use `--encoder resnet10`; keep
  `--image_fusion_mode stack_channels` for single-key `rgb`, or switch to
  `per_key` when using multiple camera/image keys.
- Pretrained ResNet: use `--encoder resnet10 --image_fusion_mode per_key
  --pretrained_weights <name> --freeze_resnet_backbone`. This keeps the
  pretrained stem/residual blocks fixed while training the pooling/bottleneck
  head through critic loss.
- Full frozen visual encoder: use `--freeze_resnet_encoder` only for ablations
  or when the pretrained visual representation should remain completely fixed.

Freeze a pretrained ResNet encoder:

```bash
python examples/train_sac_rgbd.py \
  --env_id PickCube-v1 \
  --obs_mode rgb \
  --encoder resnet10 \
  --image_fusion_mode per_key \
  --pretrained_weights resnet10-imagenet \
  --freeze_resnet_encoder
```

Freeze only the pretrained ResNet backbone:

```bash
python examples/train_sac_rgbd.py \
  --env_id PickCube-v1 \
  --obs_mode rgb \
  --encoder resnet10 \
  --image_fusion_mode per_key \
  --pretrained_weights resnet10-imagenet \
  --freeze_resnet_backbone
```

`--freeze_resnet_backbone` freezes the ResNet stem and residual blocks while
leaving the pooling/bottleneck head trainable. For Dict observations, SAC calls
the shared extractor with `stop_gradient=True` on actor updates, so image
encodings do not receive actor-loss gradients. Critic updates call the extractor
with `stop_gradient=False`, so the image encoder/head is updated by critic loss.

`ResNetEncoder` uses its own parameter naming (`stem_conv` / `stem_norm` /
`blocks.<i>.{conv,norm,proj}*`), so torchvision-style checkpoints
(`conv1`, `bn1`, `layer<N>.<block>.bn1/2`, `downsample.0/1`, ...) need to be
re-keyed before `--pretrained_weights` can load them. Run the conversion once:

```bash
python scripts/convert_resnet_checkpoint.py \
  --input pretrained_models/resnet10_pretrained.pt \
  --output pretrained_models/resnet10_pretrained_converted.pt \
  --arch resnet10
```

Then pass the converted name (without the `.pt` suffix) via
`--pretrained_weights resnet10_pretrained_converted`. The script drops the
ImageNet `fc.*` classification head, which is intentional — only backbone
weights are loaded.

Shell launchers:

```bash
scripts/train_sac_state.sh                       # state SAC, PickCube-v1
scripts/train_sac_rgbd.sh --encoder resnet10     # RGB SAC, PickCube-v1
scripts/train_sac_rgbd_peg.sh                    # RGB SAC, PegInsertionSidePegOnly-v1
scripts/train_wsrl.sh                            # WSRL state, PickCube-v1
scripts/train_wsrl_rgbd.sh                       # WSRL RGB, PickCube-v1
```

### Residual SAC

Residual SAC learns a residual action on top of a frozen base policy. The RGBD
entrypoints use the unified base-policy provider:

- `--base_policy act` loads an ACT checkpoint.
- `--base_policy sac` loads an rl-garden SAC checkpoint.
- `--base_policy zero` uses a zero-action base policy.
- `--debug` is a shortcut for a zero base policy and does not load a checkpoint.

Debug training:

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_rgbd.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --debug \
  --log_type tensorboard
```

Peg-only training uses the peg-specific launcher. In non-debug mode the
RGBD base policy defaults to ACT and loads `pretrained_models/act-peg-only.pt`
by name:

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_rgbd_peg.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --base_policy act \
  --base_ckpt_path act-peg-only \
  --log_type tensorboard
```

Use an existing SAC checkpoint as the RGBD base policy with `--base_policy sac`.
For visual SAC checkpoints, pass the base-policy encoder settings that match
the saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_rgbd_peg.sh \
  --control_mode pd_ee_delta_pose \
  --residual-action-scale 1 \
  --base_policy sac \
  --base_ckpt_path runs/<run_name>/checkpoints/final.pt \
  --base_sac_encoder resnet10 \
  --base_sac_image_fusion_mode per_key \
  --log_type tensorboard
```

State-observation peg training is also available and uses the same base-policy
flags:

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_state_peg.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --base_policy act \
  --base_ckpt_path act-peg-only \
  --log_type tensorboard
```

`ResidualSAC` follows the resfit action convention: replay/critic actions are
normalized to `[-1, 1]`, while the env receives raw actions via `ActionScaler`.
In `--debug` mode the base-action provider returns all-zero raw actions, which
is useful for testing the residual rollout/update path. ACT checkpoints are
selected with
`--base_policy act --base_ckpt_path <name-or-path>`; names resolve under
`pretrained_models/`, so `--base_ckpt_path act-peg-only` loads
`pretrained_models/act-peg-only.pt`. SAC base policies require a checkpoint path
and can use `--base_sac_encoder`, `--base_sac_encoder_features_dim`,
`--base_sac_image_fusion_mode`, and `--base_sac_deterministic` to reconstruct
visual saved policies. The generic RGBD launcher mirrors `train_sac_rgbd.py`;
peg-only kwargs such as `--fix_box`, `--fix_peg_pose`, `--robot_uids`, and
`--reward_mode` live only in the peg residual entrypoints.
`--residual-action-scale` multiplies the actor's unit residual output before
adding it to the normalized base action.
Design details are in
[`docs/RESIDUAL_SAC.md`](docs/RESIDUAL_SAC.md).

Evaluate a saved RGBD peg ResidualSAC checkpoint and write a side-by-side
`base_camera | hand_camera` video:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_residual_sac_rgbd_peg.sh \
  --checkpoint_path runs/<run_name>/checkpoints/final.pt \
  --base_policy act \
  --base_ckpt_path act-peg-only \
  --num_eval_envs 16 \
  --num_eval_steps 100 \
  --output_dir runs/<run_name>/eval_residual_videos
```

## Checkpoint Semantics

Checkpoint save paths are resolved by `resolve_checkpoint_dir()` in
`rl_garden/common/cli_args.py`. Unless `--checkpoint_dir` is explicitly passed,
the default is:

```
{log_dir}/{run_name}/checkpoints/
```

- `log_dir` defaults to `"runs"` (from `LoggingArgs.log_dir`).
- `run_name` is `--exp_name` if provided, otherwise `f"{algorithm}_offline_pretrain__{seed}__{int(time.time())}"`.

If **both** `--save_final_checkpoint=False` and `--checkpoint_freq=0`,
`resolve_checkpoint_dir()` returns `None` and no checkpoints are saved at all.

### What Gets Saved

Intermediate checkpoints (when `--checkpoint_freq > 0`):

```
runs/<run_name>/checkpoints/checkpoint_10000.pt
runs/<run_name>/checkpoints/checkpoint_20000.pt
...
```

Final checkpoint (when `--save_final_checkpoint=True`, the default):

```
runs/<run_name>/checkpoints/<algorithm>_offline_pretrained.pt
```

`--save_filename` overrides the final checkpoint filename.

### Checkpoint Format

Checkpoints are torch-native `.pt` dictionaries (format version 1). They
contain model state, optimizer state, global step counters, and metadata.
Algorithm class name aliases exist in `common/checkpoint.py` so legacy
checkpoints (`OfflineCQL`, `OfflineCalQL`) remain loadable by renamed classes
(`CQL`, `CalQL`).

Replay buffer snapshots (when `--save_replay_buffer`) are written to a
separate `_replay_buffer.pt` file next to the checkpoint.

Load a checkpoint:

```bash
python examples/pretrain_offline.py calql \
  --load_checkpoint runs/<run_name>/checkpoints/calql_offline_pretrained.pt
```

## Network Architecture

`net_arch` configuration follows SB3 semantics:

```python
from rl_garden.algorithms import SAC, WSRL, IQL, BC

sac = SAC(env=env, net_arch=[256, 256, 256])   # same actor/critic MLP sizes

wsrl = WSRL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256]},  # separate actor/critic
)

iql = IQL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256], "vf": [256, 256]},
)

bc = BC(env=env, net_arch=[256, 256])  # actor-only; no "qf"/"vf" key
```

`actor_hidden_dims` / `critic_hidden_dims` are still accepted for backward
compatibility, but they are deprecated in favor of `net_arch`.

## Feature Extractor Injection

```python
from rl_garden.algorithms import SAC, IQL, BC
from rl_garden.encoders import CombinedExtractor, resnet_encoder_factory

# RGB SAC with ResNet10 encoder
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

# BC with a custom extractor
bc = BC(
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

SAC/IQL/BC choose observation handling automatically:

- `spaces.Box` observations use `FlattenExtractor` and `TensorReplayBuffer`.
- `spaces.Dict` observations use `CombinedExtractor` and `DictReplayBuffer`.
- Dict observations may be image-only, image+state, or mixed vector inputs.

## Reward Classifiers

Generate HSV labels from compressed HDF5 episodes:

```bash
python rl_garden/reward_models/classifiers/hsv/generate_labels.py --data_dir data/epi0-19_trimmed --tune_hsv --camera high
python rl_garden/reward_models/classifiers/hsv/generate_labels.py --camera high --output data/labels.npz
```

Train the color reward classifier:

```bash
python rl_garden/reward_models/classifiers/color/train.py
```

Train the alignment reward classifier:

```bash
python rl_garden/reward_models/classifiers/alignment/train.py
```

## Robot Infrastructure

`robot_infra/` provides the real-robot integration layer:

- **Controller** (`robot_infra/controller/`): GPU-side pd_ee_twist and impedance
  controller implementations, with a ManiSkill simulator backend for co-development.
  EE kinematics run on torch tensors (Jacobian/IK on CUDA for GPU paths,
  Pinocchio-based CPU path as fallback).
- **Teleoperation** (`robot_infra/teleop/`): EE twist teleoperation interface
  (Pico/spacemouse) and recording scripts for collecting WSRL demonstration
  datasets on real hardware. See `robot_infra/controller/README.md` for setup.

## Core Design Constraints

- Rollout path is GPU-native: no `action.cpu().numpy()` or numpy replay handoff in training hot path.
- Replay tensors use `(T, N, ...)` storage layout for vectorized environments.
- Dict image observations use a shared encoder for actor/critic, and actor updates stop gradients through image encodings.
- SB3-like structure is borrowed, but `stable_baselines3` is not imported by framework code.
- `pd_ee_twist` is integrated as a ManiSkill control mode through the custom Panda agent registration path.
- In ManiSkill GPU simulation, EE controllers run on torch tensors (including Jacobian/IK math on CUDA for the GPU kinematics path).
- CPU simulation remains a compatibility fallback, where IK uses the Pinocchio-based CPU path.

## Testing

Run all CPU unit tests:

```bash
pytest tests/
```

Targeted test groups:

```bash
# Replay buffers and H5 loading
pytest tests/test_replay_buffer.py tests/test_mc_buffer.py tests/test_maniskill_h5_loader.py

# Offline algorithms
pytest tests/test_offline_algorithm.py tests/test_iql.py tests/test_cql_calql.py tests/test_wsrl.py

# Networks and encoders
pytest tests/test_networks.py tests/test_resnet_encoder.py

# PPO
pytest tests/test_ppo.py

# Checkpoint and CLI
pytest tests/test_checkpoint.py tests/test_cli_args.py tests/test_policy_kwargs.py

# ResidualSAC
pytest tests/test_residual_sac.py tests/test_act_provider.py

# WSRL-specific
pytest tests/test_wsrl.py tests/test_wsrl_rgbd.py tests/test_wsrl_policy.py tests/test_wsrl_dataset_generation.py
```

CUDA smoke test (requires CUDA + ManiSkill runtime):

```bash
pytest tests/test_sac_smoke.py
```

Offline evaluation harness (runs a saved checkpoint against a live env):

```bash
python tests/eval_offline.py --checkpoint runs/<run>/checkpoints/iql_offline_pretrained.pt \
  --env_id PickCube-v1
```

## References

- ManiSkill SAC baselines:
  - `3rd_party/ManiSkill/examples/baselines/sac/sac.py`
  - `3rd_party/ManiSkill/examples/baselines/sac/sac_rgbd.py`
- SB3 architecture references:
  - `3rd_party/stable-baselines3/stable_baselines3/common/off_policy_algorithm.py`
  - `3rd_party/stable-baselines3/stable_baselines3/common/on_policy_algorithm.py`
  - `3rd_party/stable-baselines3/stable_baselines3/ppo/ppo.py`
  - `3rd_party/stable-baselines3/stable_baselines3/sac/sac.py`
  - `3rd_party/stable-baselines3/stable_baselines3/sac/policies.py`
  - `3rd_party/stable-baselines3/stable_baselines3/common/torch_layers.py`
- ManiSkill PPO baselines:
  - `3rd_party/ManiSkill/examples/baselines/ppo/ppo.py`
  - `3rd_party/ManiSkill/examples/baselines/ppo/ppo_rgb.py`
- PPO-family offline-to-online references:
  - `3rd_party/BPPO/`
  - `3rd_party/Uni-O4/`
- hil-serl visual encoder references:
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/vision/resnet_v1.py`
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/common/encoding.py`
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/vision/data_augmentations.py`

## Roadmap

- [ ] Add stricter policy tests (optimizer parameter disjointness, numerical checks).
- [ ] Add Flax-vs-Torch parity validation for ResNet as a stretch goal.
- [ ] Expand augmentation pipeline (`RandomCrop`) integration into the training loop.
- [ ] Add a lightweight benchmark script for comparing offline algorithms.
- [ ] Add tanh-squashed PPO distributions while preserving SB3/ManiSkill-style
      unsquashed Gaussian PPO as the default baseline.
- [ ] Add separate actor/value feature extractors for PPO, including support for
      different encoder architectures on each path.
- [ ] Extend PPO toward BPPO/Uni-O4-style offline-to-online policy improvement,
      including behavior-policy snapshots, external Q/V advantage providers,
      ensemble policies, and KL-regularized behavior updates.
