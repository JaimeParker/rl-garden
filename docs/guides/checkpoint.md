# Checkpoint Save & Load

(checkpoint-what-gets-saved)=

## What gets saved

`save_final_checkpoint=True` (default) writes a `.pt` file containing:

- Model weights (actor, critic, encoder)
- Optimizer state (Adam momentum/variance buffers)
- LR scheduler state
- FlashSAC's reward-normalizer state, with `--normalize_reward`
- Global step counters (`_global_step`, `_global_update`)
- Observation/action space metadata (for compatibility validation)
- Hyperparameters
- The run configuration (`metadata["run_config"]`), the content of the run's
  `config.json` without its absolute paths and command line: resolved Args,
  selected phase and algorithm, active environment. An agent saved outside a
  training runner has none
- Every network the policy acts with, including those that came from another
  file: a SUPE checkpoint embeds its OPAL VAE and a JSRL checkpoint its guide
  policy, with the options that rebuild them

(checkpoint-no-absolute-paths)=

### Absolute paths

The checkpoint file holds no absolute path (POSIX, `~`, `file:`, Windows) and no
command line. Whatever `save()` is about to write, the embedded networks
included, a string that is or contains an absolute path is replaced by
`"<absolute path removed>"` (`REMOVED_PATH` in `rl_garden.common.checkpoint`) and
`runtime.argv` is left out. In a policy graph only the node parameter that is a path is replaced. A
name that starts with `/` (a ROS topic, a USD prim path) looks like a path and
is replaced too. Relative paths, URLs, import paths and names stay.

Only the checkpoint is changed: `config.json`, the printed configuration and
what the logger (TensorBoard, W&B) receives hold the paths and the command line,
as does `agent.run_config` in memory. Other files keep what they always held.
Checkpoints saved earlier keep the paths they have; they load, resume and
evaluate with them.

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

A snapshot usually ends in the middle of an episode. The resumed run resets its
environments, so `learn()` ends the stored unfinished episodes at that point
(`BaseReplayBuffer.cut_running_episodes`), as it also does for an opened mmap
buffer, for a dataset kept by `--online_replay_mode append`, and at a second
`learn()` in one process. The cut is a truncation: no stored `dones` flag
changes, and no buffer below takes the last row before it and the first row
after it for consecutive steps of one episode.

| Buffer | Rows of the cut episode |
|---|---|
| `ReplayBuffer` | unchanged: a sample is one stored row |
| `ReBRACReplayBuffer` | all but the last, whose next action was never stored (in this repository only the offline ReBRAC uses it) |
| `NStepReplayBuffer` | all sampled; a return that reaches the last row stops there and bootstraps from that row's stored `next_obs` |
| `ChunkedReplayBuffer` | as `NStepReplayBuffer`. A chunk that starts near the end still holds the actions of the rows after it, marked `valid=False`, as at any episode end; a critic that takes the whole chunk reads them; ACRLPD, ACFQL, QAM and QGF give such a window zero weight in the losses Q-chunking weights. |
| `MemoryEfficientReplayBuffer` | sampled with their own stored `next_obs`, except, as always, a row whose earlier frames were overwritten |
| `LazyNextNStepReplayBuffer` | all but the last, whose `next_obs` was never stored; its `obs` is the final observation |
| `SequenceReplayBuffer` in the strict and uniform modes (also inside `RehearsalSequenceBuffer`) | in windows up to the last row's `obs`; no window contains rows from both sides of the cut |
| `RecurrentReplayBuffer`, `TransformerReplayBuffer` | in windows up to the last row's `obs`; no hidden state, return or bootstrap of a window reads rows from both sides of the cut |
| `MCReplayBuffer`, `SarsaMCReplayBuffer` | not sampled: an unfinished trajectory has no Monte Carlo return. Rows whose return was stored with them (by a dataset loader, or in an old replay file without per-row validity) keep it and are sampled; `SarsaMCReplayBuffer` gives the last one `next_action_valid=False` |

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
The same holds for the growing demo buffer of `dagger` and `rlpd_hybrid`
(`demo_buffer_<suffix>.pt`). A checkpoint with a replay buffer that does not record
whether a demo buffer was saved with it resumes with an empty one and a warning
naming the file.

Resuming takes the training arguments from the command line, never from the
embedded run configuration. The exception is the online stage of an off2on run
(below).

One-time setup on a dataset is not repeated on resume: the observation
statistics (`td3_bc`, `bcq`, `plas`, `spot`, `awac`) are the checkpoint's, and
`spot` / `plas` pretrain a VAE only when the checkpoint does not record that the
pretraining finished. The dataset
itself is loaded again wherever a run keeps it: an offline run and `rlpd`,
`acrlpd`, `explore` and `supe` load it on every start, resume included.

### Resuming an off2on run

A checkpoint of the **offline stage** (`pretrain_offline`, `offline_final.pt`, or
one written during the offline steps) continues with `--num-offline-steps`
further offline steps and then switches with the command line's replay mode.
`--offline-dataset` fills its replay buffer unless `--load-replay-buffer` restored
one that holds transitions; that is reported, with their number.

A checkpoint of the **online stage** continues in the state it was saved in: the
replay mode and offline ratio (`auto` included), the buffer classes, the CQL
switch and weight, the warmup phase and how far it got, and the evaluation
cadence are recorded in it. `--online-eval-freq` or `--eval-freq` changes the
cadence. Any other of these settings passed with another value is an error, as
is `--num-offline-steps` above 0; a default that differs is reported and the
saved value used.

`--save-replay-buffer` also writes the offline buffer of the `mixed` mode, as
`offline_buffer_<suffix>.pt`. The checkpoint names the file, and a named file that is
missing raises `FileNotFoundError` with `--load-replay-buffer`. Periodic
checkpoints need `--checkpoint-freq`: an interrupted run has no `final.pt`.
What the resumed run does with `--offline-dataset` depends
on the saved mode:

| Saved mode | Buffer restored with `--load-replay-buffer` | Not restored |
|---|---|---|
| `mixed` (its offline buffer) | dataset not loaded | dataset into the offline buffer; without one, an error, unless the ratio is 0 or there was no offline data |
| `append` (its replay buffer) | dataset not loaded again | dataset into the replay buffer; without one, a warning that the offline data is gone |
| `empty` | dataset not loaded | dataset not loaded |

A dataset that is given and not loaded is reported with a warning.

`so2` never writes or reads its offline pool: as in the original
(`serial_entry_offline2online.py:78-86`, branch `origin/code`), every start,
resume included, builds it from the dataset, and online transitions are pushed
into it as they are collected. A `mixed` run of `so2` resumed from either stage
therefore needs `--offline-dataset` (in the offline stage it replaces a restored
replay buffer); the replay buffer of the online stage is restored as above.

An online-stage checkpoint that does not record its replay mode takes it from
its embedded run configuration when that run did not itself load a checkpoint,
or from a Cal-QL or WSRL replay file only the `empty` mode writes; otherwise
pass `--online-replay-mode` (and `--offline-data-ratio`). Its warmup phase is
the one its hyperparameters record, when they record one.

A resume whose return range differs from the checkpoint's (`--min-reward` /
`--max-reward` of `value_flows`, `--r-min` / `--r-max` of `floq`) is reported
with the flags that keep the saved one; like every other hyperparameter, the
command line's value is used.

In Python, `load(path)` restores the offline buffer only if it was saved.
Otherwise (`agent.lacks_offline_data`) fill
`agent.restore_offline_replay_buffer()` before training a `mixed` checkpoint;
training without it raises.

A network the policy needs comes from the checkpoint first: resuming SUPE or
JSRL restores the VAE or guide it embeds, without `--opal-checkpoint` /
`--guide-checkpoint`, and warns when the flag names a file that holds another one.
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

- `--phase auto --algorithm auto` (the defaults) pick the registry entry the
  embedded run configuration selected. Without one, or when no such entry is
  registered, it is the entry whose algorithm class is the checkpoint's
  `algorithm_class`; a class registered offline and off2on (`WSRL`) is the
  offline entry for a checkpoint of the offline stage and the off2on entry for
  one of the online stage. The agent is built by that entry's builder, so an
  algorithm registered by another package is evaluated the same way as an
  in-tree one. Evaluation exits when no entry or more than one matches
  (`SACDDP` without a run configuration needs `--algorithm sac`).
- The evaluated agent is in the stage the checkpoint was saved in, which is
  printed and written to the JSON as `training_stage` for the off2on algorithms:
  `offline` or `online`, or `unknown` when the checkpoint records none. `WSRL`
  adds its objective: `offline (Cal-QL)` before the online switch, and after it
  the one `--online-use-cql-loss` chose, `online (SAC)` by default. No buffer or
  dataset of the other stage is needed.
- The training args come from the run configuration, including the env backend's
  sub-config, then from the checkpoint's hyperparameters, then from the eval
  flags. The run configuration is the file `--config-path` names, else the one
  embedded in the checkpoint, else `<checkpoint dir>/../config.json`. The
  [removed paths](#absolute-paths) of an embedded one come from
  that `config.json` when it is the file of the run that saved the checkpoint (the
  checkpoint records its hash). Otherwise such a field is as if never set. For
  the environment (`env_id`, `control_mode`, `obs`, the fields of the backend in
  use, such as `robomimic.dataset_path`) that is an error naming the field, its
  flag and `--config-path`; a builder that needs another one fails as it does
  without it, after a log message that names them. With none
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
