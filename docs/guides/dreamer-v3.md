# DreamerV3 Model-Based RL

DreamerV3 is a world-model-based RL algorithm that learns policies through imagination—
training a latent-space world model (RSSM: Recurrent State-Space Model) to predict
environment dynamics, then planning with a learned actor-critic within the model's imagination.
It works for both state-based and image-based observations.

This implementation is ported from the official JAX codebase and
[r2dreamer](https://github.com/NM512/r2dreamer) (PyTorch reference, `rep_loss="dreamer"`
branch).

## Quick Start

### State Observations (MuJoCo HalfCheetah)

```bash
python examples/train_online.py dreamer_v3 \
  --config configs/online/dreamer_v3_halfcheetah_state.yaml
```

### Visual Observations (ManiSkill PushCube, 64×64 RGB)

```bash
python examples/train_online.py dreamer_v3 \
  --config configs/online/dreamer_v3_maniskill_pushcube_rgb.yaml
```

This config keeps the replay buffer on CUDA (`buffer_device: cuda`) and shrinks
`buffer_size` to 100,000: a uint8 image buffer at the default `buffer_size`
(1,000,000) can exceed a shared 24 GB GPU's memory on its own -- see the config
file's header comment for the math.

## Configuration and Hyperparameters

Key hyperparameters (see `rl_garden/training/online/args.py::DreamerV3TrainingArgs`):

| Parameter | Default | Description |
|---|---|---|
| `size` | `12M` | Model size preset (`12M`, `25M`, `50M`, `100M`, `200M`, `400M`, `ncrl_m`) |
| `batch_size` | `16` | Batch size for training |
| `batch_length` | `64` | Sequence length for training windows |
| `train_ratio` | `512.0` | Gradient steps per environment step (train_ratio ÷ (batch_size × batch_length)) |
| `imag_horizon` | `15` | Imagination rollout horizon for actor-critic training |
| `contdisc` | `False` | Continue-discount mode (see note below) |
| `compute_dtype` | `None` | Precision for computation (see note below) |

### Model Size Presets

The `size` parameter selects a row of `RSSM_SIZES` (`rl_garden/world_models/rssm.py`).
The stochastic state is always `stoch=32` variables with `discrete` classes each:

| Preset | `deter` | `hidden` | `discrete` | `units` | `cnn_depth` |
|---|---|---|---|---|---|
| `12M` | 2048 | 256 | 16 | 256 | 16 |
| `25M` | 3072 | 384 | 24 | 384 | 24 |
| `50M` | 4096 | 512 | 32 | 512 | 32 |
| `100M` | 6144 | 768 | 48 | 768 | 48 |
| `200M` | 8192 | 1024 | 64 | 1024 | 64 |
| `400M` | 12288 | 1536 | 96 | 1536 | 96 |

For rapid prototyping, use `12M`; scale up for complex visual tasks.

## Important Notes

### `contdisc` (Continue-Discount)

The `contdisc` flag controls how the model handles episode termination:

- **Default (False)**: Matches r2dreamer behavior. The continue head's target is
  `1 − terminated`, and imagination applies `disc = 1 − 1/horizon`.
- **Official JAX default (True)**: The continue head's target is
  multiplied by `1 − 1/horizon`, and imagination uses `disc = 1`.

In both modes imagined rollouts weight steps by the continue head's prediction. Pass
`--contdisc` to match official JAX semantics; the default here (`False`) aligns with the
r2dreamer port.

### Precision (`compute_dtype`)

- **On CUDA** (default): Uses `bfloat16` via `torch.autocast` (no loss scaling).
- **On CPU** (or explicit `--compute_dtype float32`): Falls back to `float32`.

Official JAX uses jit-compilation for performance; this port does not use
`torch.compile`, and `compute_dtype` only selects the autocast dtype.

## Limitations

The following are not currently supported:

- **Unequal image sizes**: All RGB images must have the same width and height.
- **Depth observations**: Only RGB images are processed by the decoder.
- **Frame stacking**: DreamerV3 is recurrent and maintains latent history; frame
  stacking is rejected when the agent is constructed (keep `--obs.frame_stack` at 1).
- **Mixed Dict actions**: Action spaces must be homogeneous (all Box, no mixed
  continuous + discrete). Purely discrete action spaces are not yet ported.

## Architecture Overview

DreamerV3's components live in `rl_garden`:

- **`rl_garden/world_models/rssm.py`**: Recurrent state-space model
  (RSSMSize presets table; observe, step, reward, continue_, model_loss, decode).
- **`rl_garden/encoders/dreamer_conv.py`**: Convolutional image encoder for
  visual observations (state vectors go through a symlog MLP in the same encoder).
- **`rl_garden/policies/dreamer_policy.py`**: Actor-critic policy (predict, actor head,
  critic head). The imagination loop is `DreamerV3._imagination_losses`.
- **`rl_garden/algorithms/dreamer_v3.py`**: Main algorithm class (rollout hooks,
  model/actor-critic training, replay buffer carry-state).
- **`rl_garden/world_models/imagine.py`**: Imagination rollout for planning within
  the learned model.
- **`rl_garden/planners/`**: Planner interface (DreamerV3 uses amortized policy,
  not decision-time planning like TD-MPC2).

## Related Classes

- `ModelBasedAlgorithm`: Base class for world-model-based algorithms
  (`rl_garden/algorithms/model_based.py`).
- `SequenceReplayBuffer`: Cross-episode sequence buffer with learned
  state carry (DreamerV3 uses its uniform sampling mode) (`rl_garden/buffers/sequence_replay_buffer.py`).
  As in the reference implementations, an ended episode's final observation is a
  training row with the last reward and the termination flag, followed by the
  reset row with zero reward. A backend that reports no `final_observation`
  (Isaac Lab 2.x) has no such row: the last reward and the termination flag of
  each episode are then not trained, and the replay value target before the end
  bootstraps from the reset observation. The choice is per step: a backend that
  reports `final_observation` must fill it for every env that ended, as the
  environment contract requires.

## Further Reading

- Ported from: [r2dreamer](https://github.com/NM512/r2dreamer) (Naoki et al.).
- Official JAX: [dreamerv3](https://github.com/danijar/dreamerv3) (Hafner et al.,
  arXiv:2301.04104, Nature 2025).
