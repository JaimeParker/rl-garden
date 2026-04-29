# rl-garden

SB3-style, PyTorch-native, GPU-parallel SAC framework for ManiSkill.

`rl-garden` re-organizes ManiSkill SAC baselines into reusable algorithm/policy/encoder modules while keeping the full rollout + replay + update path on torch tensors (no numpy hot-path conversion).

## Goals

- Keep ManiSkill GPU vectorization (`ManiSkillVectorEnv`) first-class.
- Follow SB3 design patterns (`Algorithm` / `Policy` / `FeaturesExtractor`) without importing `stable_baselines3`.
- Make visual pipelines pluggable: `PlainConv` and hil-serl-inspired `ResNetV1` + spatial pooling + proprio fusion.
- Make new algorithms/extensions incremental via subclassing/composition.

## What Is Already Implemented

- `SAC` (state-based) as the base off-policy algorithm.
- `RGBDSAC(SAC)` subclass for dict observations with image encoders.
- GPU-native replay buffers:
  - `TensorReplayBuffer` for Box observations.
  - `DictReplayBuffer` for `{rgb, depth, state, ...}` observations.
- Encoder stack:
  - `FlattenExtractor`
  - `PlainConv`
  - `CombinedExtractor` (image + proprio fusion)
  - `ResNetEncoder` (`resnet10`/`resnet18`/`resnet34`) + pooling heads
- ManiSkill env factory with wrappers for state/rgb/rgbd.
- Examples for state SAC and RGB SAC.
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
│   └── policies/
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
leaving the pooling/bottleneck head trainable. In RGBD SAC actor updates, the
policy calls the shared extractor with `stop_gradient=True`, so image encodings
do not receive actor-loss gradients. Critic updates call the extractor with
`stop_gradient=False`, so the image encoder/head is updated by critic loss.

Shell launchers:

```bash
scripts/train_sac_state.sh
scripts/train_sac_rgbd.sh --encoder resnet10
```

Logging backend selection:

```bash
# Default: tensorboard
scripts/train_sac_state.sh --log_type tensorboard

# Weights & Biases
scripts/train_sac_state.sh \
  --log_type wandb \
  --wandb_project rl-garden \
  --log_keywords debug,pickcube

# Stdout only (no tensorboard/wandb artifacts)
scripts/train_sac_state.sh --log_type none
```

Python-side extractor injection:

```python
from rl_garden.algorithms import RGBDSAC
from rl_garden.encoders import CombinedExtractor, resnet_encoder_factory

agent = RGBDSAC(
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

Network architecture configuration (`net_arch`) for SAC/WSRL families:

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
- RGBD path uses shared encoder for actor/critic, and actor updates stop gradients through image encodings.
- SB3-like structure is borrowed, but `stable_baselines3` is not imported by framework code.

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
