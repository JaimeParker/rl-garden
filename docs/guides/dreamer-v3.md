# DreamerV3 Model-Based RL

DreamerV3 is a world-model-based RL algorithm that learns policies through imagination—
training a latent-space world model (RSSM: Recurrent State-Space Model) to predict
environment dynamics, then planning with a learned actor-critic within the model's imagination.
It works for both state-based and image-based observations.

This implementation is ported from the official JAX codebase and
[r2dreamer](https://github.com/NM512/r2dreamer) (PyTorch reference, `rep_loss="dreamer"`
branch) and follows the decisions in `.agents/rules/adding-algorithm.md`.

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

This config sets `buffer_device: cpu` and a smaller `buffer_size` (100,000):
a GPU-resident (`buffer_device: cuda`, the default) uint8 image replay
buffer at the default `buffer_size` (1,000,000) can exceed a shared 24 GB
GPU's memory on its own -- see the config file's header comment for the math.

## Configuration and Hyperparameters

Key hyperparameters (see `rl_garden/training/online/_args.py::DreamerV3TrainingArgs`):

| Parameter | Default | Description |
|---|---|---|
| `size` | `12M` | Model size preset (`12M`, `25M`, `50M`, `100M`, `200M`, `400M`) |
| `batch_size` | `16` | Batch size for training |
| `batch_length` | `64` | Sequence length for training windows |
| `train_ratio` | `512.0` | Gradient steps per environment step (train_ratio ÷ (batch_size × batch_length)) |
| `imag_horizon` | `15` | Imagination rollout horizon for actor-critic training |
| `contdisc` | `False` | Continue-discount mode (see note below) |
| `compute_dtype` | `None` | Precision for computation (see note below) |

### Model Size Presets

The `size` parameter determines the RSSM's capacity:

```python
# size12M: deter_size=200, hidden_size=600, stoch_size=32×32, units=640
# size25M: deter_size=300, hidden_size=1000, stoch_size=32×32, units=1024
# ... up to size400M with larger dimensions
```

For rapid prototyping, use `12M`; scale up for complex visual tasks.

## Important Notes

### `contdisc` (Continue-Discount)

The `contdisc` flag controls how the model handles episode termination:

- **Default (False)**: Matches r2dreamer behavior. Terminal states set discount to 0;
  imagined rollouts ignore the continue head's prediction.
- **Official JAX default (True)**: Continue head output is multiplied by `1 − 1/horizon`;
  imagined rollouts use `disc = 1`.

Set `--contdisc true` to match official JAX semantics; the default here (`False`)
aligns with the r2dreamer port.

### Precision (`compute_dtype`)

- **On CUDA** (default): Uses `bfloat16` via `torch.autocast` (no loss scaling).
- **On CPU** (or explicit `--compute_dtype float32`): Falls back to `float32`.

Official JAX uses jit-compilation for performance; this port does not yet use
`torch.compile` (tracked for follow-up). Enable it experimentally by setting
`compute_dtype` and consulting `rl_garden/algorithms/dreamer_v3.py`'s docstring.

## Limitations

The following are not currently supported:

- **Unequal image sizes**: All RGB images must have the same width and height.
- **Depth observations**: Only RGB images are processed by the decoder.
- **Frame stacking**: DreamerV3 is recurrent and maintains latent history; frame
  stacking is rejected at preflight (set `frame_stack=1` or omit the flag).
- **Mixed Dict actions**: Action spaces must be homogeneous (all Box, no mixed
  continuous + discrete). Purely discrete action spaces are not yet ported.
- **dm_control backend**: Only `mujoco`, `maniskill`, and other standard backends
  are available for online training.

## Architecture Overview

DreamerV3's components live in `rl_garden`:

- **`rl_garden/world_models/rssm.py`**: Recurrent state-space model
  (RSSMSize presets table; observe, step, reward, continue, model_loss, decode).
- **`rl_garden/encoders/dreamer_conv.py`**: Convolutional image encoder for
  visual observations (state vectors bypassed).
- **`rl_garden/policies/dreamer_policy.py`**: Actor-critic policy that wraps the
  RSSM's imagination loop (predict, actor head, critic head).
- **`rl_garden/algorithms/dreamer_v3.py`**: Main algorithm class (rollout hooks,
  model/actor-critic training, replay buffer carry-state).
- **`rl_garden/world_models/imagine.py`**: Imagination rollout for planning within
  the learned model.
- **`rl_garden/planners/`**: Planner interface (DreamerV3 uses amortized policy,
  not decision-time planning like TD-MPC2).

## Related Classes

- `ModelBasedAlgorithm`: Base class for world-model-based algorithms
  (`rl_garden/algorithms/model_based.py`).
- `SequenceReplayBuffer`: Tolerant, cross-episode sequence buffer with learned
  state carry (`rl_garden/buffers/sequence_replay_buffer.py`).

## Further Reading

- Plan: `.agents/rules/dreamer-v3.md` (decisions and design for this port).
- Ported from: [r2dreamer](https://github.com/NM512/r2dreamer) (Naoki et al.).
- Official JAX: [dreamerv3](https://github.com/danijar/dreamerv3) (Hafner et al.,
  arXiv:2301.04104, Nature 2025).
