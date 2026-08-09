# Configuration System

rl-garden keeps its dataclass + [Tyro](https://brentyi.github.io/tyro/) + registry
architecture. A single effective-config pipeline now resolves presets, CLI values,
backend configuration, runtime-derived values, and provenance for both inspection
and normal training.

## Resolution order

Values are resolved from lowest to highest priority:

1. Dataclass defaults and subclass overrides.
2. One strict YAML preset passed with `--config`.
3. Supported `RLG_*` logging environment variables.
4. Explicit CLI flags.

A launcher in `scripts/` is a thin wrapper around a checked-in preset. Its preset
fields are identified as `launcher` in provenance; flags appended to the launcher
remain explicit CLI overrides.

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

Only the selected backend is included in the effective configuration. Explicitly
setting an inactive field is also an error—for example, setting `robotwin.step_lim`
while `env_backend` is `maniskill`, or setting `encoder` with `obs_mode: state`.
This catches overrides that would otherwise look valid but have no effect.

The supported environment variables are `RLG_STD_LOG`, `RLG_LOG_TYPE`,
`RLG_LOG_KEYWORDS`, `RLG_WANDB_PROJECT`, `RLG_WANDB_ENTITY`, and
`RLG_WANDB_GROUP`.

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

Use `--explain-param` to locate one field and its consumer:

```bash
python examples/train_online.py sac \
  --config configs/online/sac_state.yaml \
  --gamma 0.95 \
  --explain-param gamma
```

The JSON reports its value, type-defining location, owner, active condition,
mapped destination, current source, and override history. These commands are
machine-readable interfaces intended equally for humans, scripts, and coding
agents.

The three inspection actions are mutually exclusive.

## EffectiveConfig v2

Inspection and persisted `config.json` files use the same schema:

```json
{
  "schema_version": 2,
  "status": "preflight",
  "selection": {"training_phase": "online", "algorithm": "sac"},
  "inputs": {},
  "active_environment": {},
  "algorithm": {},
  "derived": {},
  "provenance": {},
  "runtime": {}
}
```

`inputs` contains the final values consumed by the runner after runtime
normalization. `derived` records every changed value as
`before`/`after`/`reason`, while provenance marks its source as
`runtime-derived`; this currently covers CUDA buffer fallback, evaluation-budget
resolution, and Minari live-environment selection.
Normal runs atomically write a `preflight` snapshot to
`{log_dir}/{run_name}/config.json`, then replace it with a `materialized` snapshot
after environment and agent construction. Evaluation accepts both v2 `inputs` and
the legacy v1 `args` section.

The completeness boundary is rl-garden: the snapshot contains the selected
backend's concrete configuration, environment request and spaces, the exact
captured agent constructor kwargs, explicit Args-to-constructor mappings, truly
unused implicit constructor defaults, plus runner-derived values. It
does not recursively serialize arbitrary third-party simulator internals.

## Adding or changing parameters

Place algorithm parameters in the relevant dataclass under
`rl_garden/training/{online,offline,off2on}/_args.py`; keep algorithm-specific
fields in the public algorithm Args subclass. Environment fields belong in
`rl_garden/common/env_args.py`, and shared logging/checkpoint fields belong in
`rl_garden/common/cli_args.py`.

Every registered public algorithm has a `ConfigContract`. The contract assigns
each Args field to environment, agent, runner, logging, or checkpoint ownership,
and records its mapped consumer. Agent builders call `construct_agent()` so the
materialized snapshot records the actual constructor call rather than inferring
consumption from object attributes. Contract tests require every field to be covered
and reject important required constructor parameters that are neither mapped nor
explicitly declared runtime-derived. Strict contracts also compare active agent
fields with the exact captured constructor kwargs during materialization, so a
builder that forgets to forward a field fails before training starts.

Registering with `registry.register(..., contract_mode="passthrough")` skips this
completeness check instead of enforcing it -- use it only for entrypoints that don't
have a full contract yet (currently the `real_world` `serl`/`hil_serl` algorithms).
The active mode is always visible under `algorithm.mode` in `--print-config` and
`--dry-run` output.

Experiment-specific values belong in one `configs/<phase>/*.yaml` file. Launcher
scripts should only select a preset, forward logging environment values, and append
user CLI flags. Do not duplicate a preset's hyperparameters in shell code.

Before committing a new field:

1. Add focused parsing and invalid-input tests.
2. Confirm its contract owner and mapped destination.
3. Run `--print-config` and `--explain-param <field>`.
4. Run `--dry-run` when the relevant optional backend is available.
