# Checkpoint Save & Load

(checkpoint-what-gets-saved)=

## What gets saved

`save_final_checkpoint=True` (default) writes a `.pt` file containing:

- Model weights (actor, critic, encoder)
- Optimizer state (Adam momentum/variance buffers)
- LR scheduler state
- Global step counters (`_global_step`, `_global_update`)
- Observation/action space metadata (for compatibility validation)
- Hyperparameters
- The run configuration (`metadata["run_config"]`), the content of the run's
  `config.json`: resolved Args, selected phase and algorithm, active environment.
  An agent saved outside a training runner has none
- Every network the policy acts with, including those that came from another
  file: a SUPE checkpoint embeds its OPAL VAE and a JSRL checkpoint its guide
  policy, with the options that rebuild them

(checkpoint-replay-buffer)=

## Replay buffer

Replay buffer snapshots are **off by default** because buffers can be large
(GB-scale). With `--save_replay_buffer`, the buffer is written next to the
checkpoint as `replay_buffer_<suffix>.pt`: `replay_buffer_final.pt` for
`final.pt`, and `replay_buffer_<n>.pt` for `checkpoint_<n>.pt`.

For off-policy algorithms (SAC, CQL, Cal-QL, WSRL), the replay buffer is part
of the training state. Loading a checkpoint without its replay buffer starts the
resumed run with an empty buffer. The first several thousand gradient steps
train on highly correlated, low-diversity online data — a distribution shift
that often degrades the policy before the buffer refills. This is inherent to
off-policy learning, not a bug in the checkpoint mechanism.

| Scenario | `--save_replay_buffer` |
|---|---|
| One-off artifact (e.g. final model for deployment) | omitted (default) — saves disk space |
| Checkpoint will be used to **resume training** later | passed — preserves full training state |
| Offline-only checkpoint (Cal-QL/CQL pretraining) | Usually not needed — the offline dataset can be reloaded |

(checkpoint-loading)=

## Loading

```bash
# Resume SAC training from checkpoint
python examples/train_online.py sac \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --total_timesteps 500000

# Load with replay buffer
python examples/train_online.py sac \
  --load_checkpoint runs/<run>/checkpoints/final.pt \
  --load_replay_buffer \
  --total_timesteps 500000
```

`--load-replay-buffer` opts in to loading the buffer. On the CLI it is off by
default; in the Python API, `load(path, load_replay_buffer=True)` is the default.
It is a no-op when the checkpoint was saved without a buffer; when the checkpoint
references a buffer file that is missing, loading raises `FileNotFoundError`.

Resuming takes the training arguments from the command line, never from the
embedded run configuration.

A network the policy needs comes from the checkpoint first: resuming SUPE or
JSRL restores the VAE or guide it embeds, and warns when
`--opal-checkpoint` / `--guide-checkpoint` names a file that holds another one.
A checkpoint that embeds none reads that file, and raises `FileNotFoundError`
naming it when it is gone.

A JSRL guide is rebuilt from the options its own checkpoint records. For a guide
checkpoint that does not record the actor's options, its weights decide the std
parameterization (`--guide-std-parameterization` is the fallback) and whether
the actor has normalization layers, taken as LayerNorm with a warning because
GroupNorm has the same weights. Any other mismatch raises, naming the missing
and unexpected keys.

(checkpoint-evaluating-a-checkpoint)=

## Evaluating a checkpoint

`rl-garden-eval` (from a source checkout: `python examples/eval_checkpoint.py`;
the code is `rl_garden/training/evaluation.py`) rebuilds the agent, loads the
weights without replay or optimizer state, and runs the policy until
`--num-eval-episodes` episodes finish. It needs the checkpoint and an
evaluation environment: no run directory, dataset, warm-start, guide, teacher or
world-model file, and no training environment.

```bash
rl-garden-eval --checkpoint-path runs/<run>/checkpoints/final.pt \
  --num-eval-envs 16 --num-eval-episodes 50 --output-json /tmp/eval.json
```

- `--phase auto --algorithm auto` (the defaults) pick the registry entry whose
  algorithm class is the checkpoint's `algorithm_class`; the agent is built by
  that entry's builder, so an algorithm registered by another package is
  evaluated the same way as an in-tree one. Evaluation exits when the class
  matches no entry or more than one: `WSRL` is registered offline and off2on and
  needs `--phase`, and `SACDDP` needs `--algorithm sac`.
- The training args come from the run configuration, including the env backend's
  sub-config, then from the checkpoint's hyperparameters, then from the eval
  flags. The run configuration is the file `--config-path` names, else the one
  embedded in the checkpoint, else `<checkpoint dir>/../config.json`. With none
  of them evaluation exits unless `--env-backend` and `--env-id` are passed, and
  then warns that every setting the checkpoint does not record takes its
  default.
- `--env-backend`, `--env-id` and `--control-mode` replace the run's values only
  when passed (`--guide-checkpoint` / `--opal-checkpoint` likewise say where the
  file of a checkpoint that embeds no guide or VAE is); `--obs.*` replaces the run's observation config only when it
  differs from the default one. Left out, the run's environment is used, so
  evaluating a `gymnasium`-backend run needs no backend flags. A backend
  sub-config field (e.g. `--gymnasium.device cuda:1`) replaces the run's value
  only when set to a non-default value.
- Inputs only training needs are dropped before the agent is rebuilt (the
  registry entry's `training_only_inputs`, see
  [Extending](extending.md#the-registry-call)), and encoders do not read
  `--encoder.pretrained-weights` again: the checkpoint holds the weights.
- `--device` is the device of the agent, for every algorithm whose class takes
  one; the simulator's device is a backend field. Replay buffers get no storage
  (`--buffer-device meta`, the default).
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

(checkpoint-compatibility)=

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
