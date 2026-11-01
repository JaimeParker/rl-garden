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
python examples/train_online.py sac \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --total_timesteps 500000

# Load with replay buffer
python examples/train_online.py sac \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --load_replay_buffer True \
  --total_timesteps 500000
```

`--load_replay_buffer True` (default) attempts to load the buffer; it is a
no-op when the checkpoint has no saved buffer.

## Evaluating a checkpoint

`rl-garden-eval` (from a source checkout: `python examples/eval_checkpoint.py`;
the code is `rl_garden/training/evaluation.py`) rebuilds the agent, loads the
weights without replay or optimizer state, and runs the policy until
`--num-eval-episodes` episodes finish:

```bash
rl-garden-eval --checkpoint-path runs/<run>/checkpoints/final.pt \
  --num-eval-envs 16 --num-eval-episodes 50 --output-json /tmp/eval.json
```

- `--phase auto --algorithm auto` (the defaults) pick the registry entry whose
  algorithm class is the checkpoint's `algorithm_class`; the agent is built by
  that entry's builder, so an algorithm registered by another package is
  evaluated the same way as an in-tree one.
- The training args come from the run's `config.json` (default:
  `<checkpoint dir>/../config.json`, or `--config-path`), including the env
  backend's sub-config, then from the checkpoint's hyperparameters.
- `--env-backend`, `--env-id` and `--control-mode` replace the run's values only
  when passed; `--obs.*` replaces the run's observation config only when it
  differs from the default one. Left out, the run's environment is used, so
  evaluating a `gymnasium`-backend run needs no backend flags. A backend
  sub-config field (e.g. `--gymnasium.device cuda:1`) replaces the run's value
  only when set to a non-default value.
- `strict` (on by default) rejects a checkpoint whose class, spaces or policy
  graph do not match the rebuilt agent.
- A checkpoint whose algorithm class sets `checkpoint_eval_unsupported_reason`
  is refused with that reason before any environment is built. The dataset-only
  offline stages set it: `a2a_bc`, `diffusion_bc` and `consistency_distill_bc`
  (their policies predict whole action chunks), `hilp` and `opal` (skill
  pretraining), `rssm_pretrain` (world-model pretraining), `tdmpc2_multitask`
  (no multitask evaluation) and `pwm` (evaluated by its own training run in a
  padded multitask environment). Their checkpoints still rebuild and load
  through the registered builders.

## Compatibility

Algorithm class aliases keep legacy checkpoints loadable:

| Saved as | Loads into |
|---|---|
| `OfflineCQL` | `CQL` |
| `OfflineCalQL` | `CalQL` |
| `WSRL` | `WSRL` (also accepts `CalQL`, `CQL`) |

An algorithm class always loads checkpoints saved under its own class name
(including a bare subclass of a registered algorithm) and declares any other
accepted names via `_compatible_checkpoint_algorithms`. Mismatched checkpoints
raise a `ValueError` describing the incompatibility.

Checkpoints saved before the unified observation/encoder redesign (`obs`/
`encoder`/`obs_groups`, `rl_garden/observations/`) are **not supported**.
There is no migration script: loading one raises on the old observation
metadata format instead of silently reinterpreting it. Retrain from scratch
against the current `obs`/`encoder` surface.
