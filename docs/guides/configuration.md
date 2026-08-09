# Configuration System

rl-garden keeps its dataclass + [Tyro](https://brentyi.github.io/tyro/) + registry
architecture. A single effective-config pipeline resolves presets, CLI values,
the selected backend, and runtime-derived values for both inspection and normal
training. It reports facts from parsing and materialization rather than trying to
infer a static graph of every parameter consumer.

## Resolution order

Values are resolved from lowest to highest priority:

1. Dataclass defaults and subclass overrides.
2. One strict YAML preset passed with `--config`.
3. Explicit CLI flags.

A launcher in `scripts/` is a thin wrapper around a checked-in preset. Its preset
fields are identified as `preset` in `sources`; flags appended to the launcher
remain explicit CLI overrides. To choose a different preset, invoke the Python
entrypoint directly instead of passing a second `--config` to a launcher.

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml \
  --gamma 0.95
```

Preset files contain argument fields only. The entrypoint and algorithm stay on
the command line:

```yaml
env_id: PickCube-v1
obs_mode: state
num_envs: 16
gamma: 0.99
```

YAML loading is strict: unknown fields, duplicate keys, wrong types, multiple preset files, and
top-level `training_phase` or `algorithm` selectors fail before training. There
are deliberately no includes or preset inheritance. Backend sections use the same
nested names as Tyro:

```yaml
env_backend: maniskill
maniskill:
  sim_backend: physx_cpu
  reward_mode: normalized_dense
```

Backend dataclasses are omitted from `inputs`; the selected backend appears once
under `active_environment`. Explicitly
setting an inactive field is also an error—for example, setting `robotwin.step_lim`
while `env_backend` is `maniskill`, or setting `encoder` with `obs_mode: state`.
This catches overrides that would otherwise look valid but have no effect.

Training and logging values are configured only through YAML and CLI. Environment
variables are reserved for third-party runtime requirements such as renderer,
cache, or external simulator paths. The removed logging variables map directly to
`--std-log`/`--no-std-log`, `--log-type`, `--log-keywords`, `--wandb-project`,
`--wandb-entity`, and `--wandb-group`.

## Inspect before training

`--print-config` performs pure parsing and static validation. It does not import a
simulator backend or create an environment, logger, agent, replay buffer, or run
directory:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml \
  --print-config | python -m json.tool
```

`--dry-run` goes one step further. It may load the selected backend and dataset
metadata, materializes the environment request, concrete backend configs,
observation/action spaces, devices, agent, and configured checkpoint state, then
exits before full dataset/replay loading, W&B initialization, checkpoint writes,
or any training call:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml \
  --dry-run | python -m json.tool
```

Use `--explain-param` to inspect one field's resolved value and source:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml \
  --gamma 0.95 \
  --explain-param gamma
```

The JSON contains exactly `path`, `value`, `type`, and `source`. A field that was
not overridden reports `source.kind: default`; preset, CLI, and runtime-derived
values report their final source.
Inactive fields fail instead of returning a misleading value. These commands are
machine-readable interfaces intended equally for humans, scripts, and coding agents.

The three inspection actions are mutually exclusive.

## EffectiveConfig v3

Inspection and persisted `config.json` files use the same schema:

```json
{
  "schema_version": 3,
  "status": "preflight",
  "selection": {"training_phase": "online", "algorithm": "sac"},
  "inputs": {},
  "active_environment": {},
  "algorithm": {},
  "derived": {},
  "sources": {},
  "runtime": {}
}
```

`inputs` contains the active Args values consumed by the runner after runtime
normalization, excluding backend dataclasses. `sources` is intentionally sparse:
it contains only fields actually overridden by a preset, CLI, or runtime
normalization. An absent path means the dataclass default won. `derived` records
changed runtime values as
`before`/`after`/`reason`; this currently covers CUDA buffer fallback,
evaluation-budget resolution, and Minari live-environment selection.
Normal runs atomically write a `preflight` snapshot to
`{log_dir}/{run_name}/config.json`, then replace it with a `materialized` snapshot
after environment and agent construction. Evaluation accepts v3 `inputs` and
the legacy v1 `args` section.

The completeness boundary is rl-garden: the snapshot contains the selected
backend's concrete configuration, environment request and spaces, the exact
captured agent constructor kwargs, plus runner-derived values. Preflight leaves
`algorithm` empty because no agent has been constructed; materialization fills only
`algorithm.target` and `algorithm.constructor_kwargs`. It
does not recursively serialize arbitrary third-party simulator internals.

## Adding or changing parameters

Place algorithm parameters in the relevant dataclass under
`rl_garden/training/{online,offline,off2on}/_args.py`; keep algorithm-specific
fields in the public algorithm Args subclass. Environment fields belong in
`rl_garden/common/env_args.py`, and shared logging/checkpoint fields belong in
`rl_garden/common/cli_args.py`.

Agent builders call `construct_agent()` so the materialized snapshot records the
actual constructor call. There is no parallel owner/mapping/consumption schema to
keep synchronized with Python code. When a field must reach a builder or multiple
consumers, add an algorithm-near test that asserts the exact resulting kwargs or
runner behavior; this is more reliable than a global name-based inference table.

Experiment-specific values belong in one `configs/<phase>/*.yaml` file. Launcher
scripts should only select a fixed algorithm and preset, then append user CLI flags
unchanged. Do not duplicate or interpret training parameters in shell code. New
presets do not need a matching launcher. Keep shell scripts for third-party runtime
setup or release-time reproduction workflows that coordinate more than one command.

Before committing a new field:

1. Add focused parsing and invalid-input tests.
2. Add a builder/runner assertion when the field must be forwarded or transformed.
3. Run `--print-config` and `--explain-param <field>`.
4. Run `--dry-run` when the relevant optional backend is available and inspect the
   captured constructor kwargs.
