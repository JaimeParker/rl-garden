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

Freeze a pretrained ResNet encoder:

```bash
python examples/train_sac_rgbd.py \
  --env_id PickCube-v1 \
  --obs_mode rgb \
  --encoder resnet10 \
  --pretrained_weights resnet10-imagenet \
  --freeze_resnet_encoder
```

Freeze only the pretrained ResNet backbone:

```bash
python examples/train_sac_rgbd.py \
  --env_id PickCube-v1 \
  --obs_mode rgb \
  --encoder resnet10 \
  --pretrained_weights resnet10-imagenet \
  --freeze_resnet_backbone
```

Shell launchers:

```bash
scripts/train_sac_state.sh
scripts/train_sac_rgbd.sh --encoder resnet10
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
        },
    },
)
```

## Core Design Constraints

- Rollout path is GPU-native: no `action.cpu().numpy()` or numpy replay handoff in training hot path.
- Replay tensors use `(T, N, ...)` storage layout for vectorized environments.
- RGBD path uses shared encoder for actor/critic, and supports detaching encoder on actor updates.
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
- [ ] Add Flax-vs-Torch parity validation for ResNet as a stretch goal.
- [ ] Expand augmentation pipeline integration in training loop.
- [ ] Add checkpoint/load examples and a lightweight benchmark script.
