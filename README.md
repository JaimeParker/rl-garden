# rl-garden

SB3-style, PyTorch-native, GPU-parallel SAC framework for ManiSkill.

`rl-garden` re-organizes ManiSkill SAC baselines into reusable algorithm/policy/encoder modules while keeping the full rollout + replay + update path on torch tensors (no numpy hot-path conversion).

## Goals

- Keep ManiSkill GPU vectorization (`ManiSkillVectorEnv`) first-class.
- Follow SB3 design patterns (`Algorithm` / `Policy` / `FeaturesExtractor`) without importing `stable_baselines3`.
- Make visual pipelines pluggable: `PlainConv` and hil-serl-inspired `ResNetV1` + spatial pooling + proprio fusion.
- Make new algorithms/extensions incremental via subclassing/composition.

## What Is Already Implemented

- `SAC` for both flat Box state observations and Dict image/image+state observations.
- `SACCore`, shared by online SAC, offline SAC, CQL, Cal-QL, and WSRL.
- `CQL` / `CalQL` as standalone offline pretraining algorithms.
- `WSRL` as the Cal-QL-based offline→online warm-start flow layer.
- GPU-native replay buffers:
  - `TensorReplayBuffer` for Box observations.
  - `DictReplayBuffer` for `{rgb, depth, state, ...}` observations.
- Encoder stack:
  - `FlattenExtractor`
  - `PlainConv`
  - `CombinedExtractor` (image + proprio fusion)
  - `ResNetEncoder` (`resnet10`/`resnet18`/`resnet34`) + pooling heads
- ManiSkill env factory with wrappers for state/rgb/rgbd.
- Python examples for state SAC and RGB SAC.
- Unit tests + CUDA smoke tests.

## Project Layout

```text
rl-garden/
├── 3rd_party/                      # read-only references (git submodules)
│   ├── ManiSkill
│   ├── stable-baselines3
│   └── hil-serl
├── rl_garden/
│   ├── algorithms/
│   ├── buffers/
│   ├── common/
│   ├── encoders/
│   ├── envs/
│   ├── networks/
│   ├── policies/
│   └── reward_models/
├── examples/
├── scripts/
├── tests/
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

State SAC:

```bash
python examples/train_sac_state.py --env_id PickCube-v1 --num_envs 16
```

End-to-end WSRL reproduction, from SAC checkpoints to dataset generation and
offline-to-online training, is documented in
[`docs/WSRL_REPRODUCTION.md`](docs/WSRL_REPRODUCTION.md).

Offline pretraining from a flat ManiSkill H5 dataset:

```bash
scripts/pretrain_offline.sh --algorithm calql --offline_dataset_path demos/pickcube.h5
scripts/pretrain_offline.sh --algorithm cql --offline_dataset_path demos/pickcube.h5
scripts/pretrain_cql_offline.sh --offline_dataset_path demos/pickcube.h5
scripts/pretrain_calql_offline.sh --offline_dataset_path demos/pickcube.h5
```

The primary entrypoint is `examples/pretrain_offline.py --algorithm
cql|calql|wsrl-calql`; the algorithm-specific launchers are compatibility
wrappers. The final checkpoint defaults to
`<algorithm>_offline_pretrained.pt` and can be loaded into compatible
SAC-family agents for later evaluation or fine-tuning.

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

Shell launchers:

```bash
scripts/train_sac_state.sh                       # state SAC, PickCube-v1
scripts/train_sac_rgbd.sh --encoder resnet10     # RGB SAC, PickCube-v1
scripts/train_sac_rgbd_peg.sh                    # RGB SAC, PegInsertionSidePegOnly-v1
```

Shell launchers wrap `python examples/train_sac_*.py` with env-specific
flag presets (`--env_id`, `--num_envs`, `--total_timesteps`, etc.) and
logging environment variables (`RLG_LOG_TYPE`, `RLG_LOG_KEYWORDS`).
Algorithm hyperparameters (`batch_size`, `utd`, learning rates, ...) live
in `SACTrainingArgs` / `VisionSACTrainingArgs` and are not duplicated in
the shell.

Logging backend selection:

```bash
# Default: tensorboard
scripts/train_sac_state.sh --log_type tensorboard

# Weights & Biases (RGBD)
scripts/train_sac_rgbd.sh \
  --log_type wandb \
  --wandb_project rl-garden \
  --encoder resnet10 \
  --log_keywords debug,pickcube

# Stdout only (no tensorboard/wandb artifacts)
scripts/train_sac_state.sh --log_type none
```

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

Python-side extractor injection:

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

SAC chooses observation handling automatically:

- `spaces.Box` observations use `FlattenExtractor` and `TensorReplayBuffer`.
- `spaces.Dict` observations use `CombinedExtractor` and `DictReplayBuffer`.
- Dict observations may be image-only, image+state, or mixed vector inputs.
- Dict-obs training defaults (smaller `batch_size`, lower `utd`) live in
  `VisionSACTrainingArgs` and the `examples/train_sac_rgbd*.py` entrypoints.
  When constructing `SAC(env=dict_env, ...)` directly from Python, consider
  overriding `batch_size=512, utd=0.25` to match the vision-tuned defaults.

Network architecture configuration (`net_arch`) for SAC/CQL/Cal-QL/WSRL families:

```python
from rl_garden.algorithms import SAC, WSRL

sac = SAC(env=env, net_arch=[256, 256, 256])  # same actor/critic MLP sizes

wsrl = WSRL(
    env=env,
    net_arch={"pi": [256, 256], "qf": [256, 256]},  # separate actor/critic
)
```

`actor_hidden_dims` / `critic_hidden_dims` are still accepted for backward
compatibility, but they are deprecated in favor of `net_arch`.

## Core Design Constraints

- Rollout path is GPU-native: no `action.cpu().numpy()` or numpy replay handoff in training hot path.
- Replay tensors use `(T, N, ...)` storage layout for vectorized environments.
- Dict image observations use a shared encoder for actor/critic, and actor updates stop gradients through image encodings.
- SB3-like structure is borrowed, but `stable_baselines3` is not imported by framework code.
- `pd_ee_twist` is integrated as a ManiSkill control mode through the custom Panda agent registration path.
- In ManiSkill GPU simulation, EE controllers run on torch tensors (including Jacobian/IK math on CUDA for the GPU kinematics path).
- CPU simulation remains a compatibility fallback, where IK uses the Pinocchio-based CPU path.

## Testing

CPU unit tests:

```bash
pytest tests/test_replay_buffer.py tests/test_resnet_encoder.py
```

CUDA smoke tests (requires CUDA + ManiSkill runtime):

```bash
pytest tests/test_sac_smoke.py
```

## References

- ManiSkill SAC baselines:
  - `3rd_party/ManiSkill/examples/baselines/sac/sac.py`
  - `3rd_party/ManiSkill/examples/baselines/sac/sac_rgbd.py`
- SB3 architecture references:
  - `3rd_party/stable-baselines3/stable_baselines3/common/off_policy_algorithm.py`
  - `3rd_party/stable-baselines3/stable_baselines3/sac/sac.py`
  - `3rd_party/stable-baselines3/stable_baselines3/sac/policies.py`
  - `3rd_party/stable-baselines3/stable_baselines3/common/torch_layers.py`
- hil-serl visual encoder references:
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/vision/resnet_v1.py`
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/common/encoding.py`
  - `3rd_party/hil-serl/serl_launcher/serl_launcher/vision/data_augmentations.py`

## Roadmap

- [ ] Formalize `policy_kwargs` / `features_extractor_class`-style injection API.
- [ ] Add stricter policy tests (optimizer parameter disjointness, numerical checks).
- [ ] Expand `net_arch` docs/examples across all training entrypoints.
- [ ] Add Flax-vs-Torch parity validation for ResNet as a stretch goal.
- [ ] Expand augmentation pipeline integration in training loop.
- [ ] Add checkpoint/load examples and a lightweight benchmark script.
