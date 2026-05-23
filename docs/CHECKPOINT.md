# Checkpoint Save & Load

## What gets saved

`save_final_checkpoint=True` (default) writes a `.pt` file containing:

- Model weights (actor, critic, encoder)
- Optimizer state (Adam momentum/variance buffers)
- LR scheduler state
- Global step counters (`_global_step`, `_global_update`)
- Observation/action space metadata (for compatibility validation)
- Hyperparameters

## Replay buffer

Replay buffer snapshots are **off by default** (`--save_replay_buffer False`)
because buffers can be large (GB-scale). When enabled, the buffer is written to
a separate `_replay_buffer.pt` next to the checkpoint.

For off-policy algorithms (SAC, CQL, Cal-QL, WSRL), the replay buffer is part
of the training state. Loading a checkpoint without its replay buffer starts the
resumed run with an empty buffer. The first several thousand gradient steps
train on highly correlated, low-diversity online data — a distribution shift
that often degrades the policy before the buffer refills. This is inherent to
off-policy learning, not a bug in the checkpoint mechanism.

| Scenario | `--save_replay_buffer` |
|---|---|
| One-off artifact (e.g. final model for deployment) | `False` (default) — saves disk space |
| Checkpoint will be used to **resume training** later | `True` — preserves full training state |
| Offline-only checkpoint (Cal-QL/CQL pretraining) | Usually not needed — the offline dataset can be reloaded |

## Loading

```bash
# Resume SAC training from checkpoint
python examples/train_sac_state.py \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --total_timesteps 500000

# Load with replay buffer
python examples/train_sac_state.py \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --load_replay_buffer True \
  --total_timesteps 500000
```

`--load_replay_buffer True` (default) attempts to load the buffer; it is a
no-op when the checkpoint has no saved buffer.

## Compatibility

Algorithm class aliases keep legacy checkpoints loadable:

| Saved as | Loads into |
|---|---|
| `OfflineCQL` | `CQL` |
| `OfflineCalQL` | `CalQL` |
| `WSRL` | `WSRL` (also accepts `CalQL`, `CQL`) |

Each algorithm class declares which checkpoints it accepts via
`_compatible_checkpoint_algorithms`. Mismatched checkpoints raise a `ValueError`
describing the incompatibility.
