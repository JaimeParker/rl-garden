# Configuration System

rl-garden keeps its dataclass + [Tyro](https://brentyi.github.io/tyro/) + registry
architecture. A single effective-config pipeline resolves presets, CLI values,
the selected backend, and runtime-derived values for both inspection and normal
training. It reports facts from parsing and materialization rather than trying to
infer a static graph of every parameter consumer.

## Resolution order

Values are resolved from lowest to highest priority:

1. Dataclass defaults. This layer itself has two levels, resolved by ordinary
   Python subclassing (MRO), not by the config system:
   - Shared fields declared once in `rl_garden/common/cli_args.py` (logging,
     checkpoint) and `rl_garden/common/env_args.py` (environment) -- the
     generic baseline every algorithm inherits unchanged.
   - An algorithm's own `*Args` subclass, which may re-declare one of those
     fields with a different default (e.g. `DrQv2Args`/`TD3Args` overriding
     `load_replay_buffer`). This should stay the exception: most algorithms
     never need to touch a shared field's default, and doing so is a visible
     redeclaration in that subclass, not a hidden global change.
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
while `env_backend` is `maniskill`, or setting `encoder.*` to a non-default value
while `obs` requests no `rgb`/`depth` cameras (state-only observations have
nothing to encode). This catches overrides that would otherwise look valid but
have no effect.

Training and logging values are configured only through YAML and CLI. Environment
variables are reserved for third-party runtime requirements such as renderer,
cache, or external simulator paths. The removed logging variables map directly to
`--std-log`/`--no-std-log`, `--log-type`, `--log-keywords`, `--wandb-project`,
`--wandb-entity`, and `--wandb-group`.

## Observation and encoder configuration

Every algorithm shares one `obs`/`encoder`/`obs_groups`/`critic_encoder`/
`encoder_sharing` surface (`ObservationArgs`, `rl_garden/common/cli_args.py`).
`obs` decides *what* is observed (state / cameras / image size / frame
stacking); it is a backend/dataset-side request, honored exactly or rejected:

```yaml
obs:
  state: true
  rgb: [base_camera]       # camera names -> keys "rgb_<cam>"
  depth: []
  extra_state: [object_pose]  # auxiliary low-dim keys -> "state_<name>"
  image_size: [64, 64]     # (H, W); omit to use the backend's own default
  frame_stack: 1
encoder:
  backbone: plain_conv    # plain_conv | resnet10 | resnet18 | vit | drqv2_conv | cnn3d
  features_dim: 256
```

State-only is the default (`obs: {state: true}` implicitly); omit `obs`
entirely for a state preset. `encoder` is only meaningful once `obs` requests
at least one camera -- setting `encoder.*` to a non-default value with no
`rgb`/`depth` cameras is a preflight error (see above). Camera names are
backend-specific: ManiSkill sensor names (e.g. `base_camera`), RoboTwin
`head`/`left_wrist`/`right_wrist`, OGBench's single visual camera `ogbench`,
RLBench (`left_shoulder`/`right_shoulder`/`overhead`/`wrist`/`front`), and
Meta-World's six fixed cameras (`corner`/`corner2`/`corner3`/`corner4`/
`behindGripper`/`gripperPOV`).

### Low-dim observations and privileged state

`obs.extra_state` declares auxiliary/privileged low-dim observations beyond
the base `state` key, creating keys named `state_<name>` in the observation
space. These follow the same strict validation as the base `state` key: each
must be a 1-D float vector. The privileged-critic idiom uses `obs_groups` to
give the critic access to extra state that the actor does not see:

```yaml
obs:
  state: true
  rgb: [base_camera]
  extra_state: [object_pose, gripper_state]  # creates state_object_pose, state_gripper_state
obs_groups:
  actor: [rgb_base_camera, state]
  critic: [rgb_base_camera, state, state_object_pose, state_gripper_state]
```

All state keys (base `state` plus each `state_<name>`) are concatenated
together by the encoder's proprio branch into one dense vector; there is no
per-key MLP, only one shared proprio branch that reads all state keys.
`encoder_sharing` controls whether that dense vector is computed once (shared)
or twice (actor and critic each compute independently).

The privileged-critic idiom also works without cameras (state-only observations):

```yaml
obs: {state: true, extra_state: [object_pose]}
obs_groups:
  actor: [state]
  critic: [state, state_object_pose]
```

CLI equivalent:

```bash
--obs.state true --obs.extra-state object_pose \
  --obs-groups.actor state --obs-groups.critic state state_object_pose
```

### Actor/critic encoder asymmetry

`obs_groups` optionally splits which observation keys the actor and critic
each consume (asymmetric/privileged critic); `critic_encoder` optionally
gives the critic its own encoder hyperparameters:

```yaml
obs_groups:
  actor: [rgb_base_camera]
  critic: [rgb_base_camera, state]
critic_encoder:
  backbone: resnet18    # different from encoder.backbone
  features_dim: 512
```

When asymmetric `obs_groups` are provided (actor observes different keys than
critic), `encoder_sharing` is automatically inferred to `separate`. Use
`--print-config` or `--explain-param encoder_sharing` to inspect the resolved
value and its origin (`inferred (asymmetric obs_groups)`, `inferred (critic_encoder)`,
or `<algo> default`). You may explicitly set `--encoder-sharing` to override
the default, but a non-`separate` value that contradicts asymmetric groups
raises `ObservationContractError` at construction time.

Algorithms that support asymmetric `obs_groups` all have a critic/value head
(every algorithm except BC-only families: BC, DiffusionBC, FlowBC, MeanFlowBC,
A2ABC, ConsistencyDistillBC, DAgger). Algorithms with recurrent or transformer
state abstractions (RecurrentSAC, RecurrentPPO, TransformerSAC, TransformerPPO)
cannot use separate encoders (both cannot use `encoder_sharing="separate"`) because
there is only one RNN/attention mechanism shared between the encoder and the heads;
attempting to pass asymmetric `obs_groups` to these families raises
`ObservationContractError` at construction time.

**Activation rule:** `obs_groups` and `encoder_sharing` are always active.
`critic_encoder` and the image-specific fields of `encoder` (all except
`normalize_obs`) apply only when `obs` includes at least one camera
(`rgb` or `depth` key). `encoder.normalize_obs` always applies.

### H5 dataset observation layout

Offline H5 trajectory files (`rl_garden/buffers/h5_dataset.py`) are held to
the same strict key vocabulary as a live env: each `traj_*/obs` (or
`/observations`) node is either a flat `Dataset` -- inferred as a state-only
observation, wrapped into `Dict({"state": Box})` -- or a `Group` whose child
keys must already be exactly `state`, `rgb_<cam>`, or `depth_<cam>`. Any
other key name (e.g. a producer's own `left_shoulder_rgb`, `proprio`) raises
`ObservationContractError` at load time; this generic loader does no
renaming of its own. A producer with different native field names (RLBench,
robomimic) maps them onto the contract in its own dataset loader before the
data ever reaches an H5 file this loader reads -- see
[RLBench Integration](rlbench-integration.md) and
[robomimic Integration](robomimic-integration.md) for those two producers'
own mappings.

### `frame_stack` vs `cond_steps`

`obs.frame_stack` (`ObservationConfig`) and `cond_steps` (the chunked-BC
family's own args, e.g. `DiffusionBCTrainingArgs`) both add a leading
dimension to an observation, but at different layers and for different
reasons -- they are deliberately not merged:

- `obs.frame_stack` is an **env-side** concern: the backend stacks the last
  `frame_stack` raw image frames into a leading time dimension
  (`ImageFrameStackWrapper`) before the observation ever reaches an
  algorithm. It only applies to image keys (`rgb_<cam>`/`depth_<cam>`);
  vector `state` stays single-frame.
- `cond_steps` is an **algorithm-side** concern specific to the chunked-BC
  imitation family (`BC`, `DiffusionBC`, `FlowBC`, `MeanFlowBC`, `A2ABC`,
  `ConsistencyDistillBC`): it is how much observation *history* (as a time
  dimension) the chunked-dataset loader (`load_h5_dataset_as_chunks`,
  `horizon_steps`/`cond_steps`) keeps per training example, independent of
  whether the underlying observation is state or image.

A vision chunked-BC run can use both at once (env-side `frame_stack` on the
raw camera feed, algorithm-side `cond_steps` on the resulting per-step
features) -- they compose rather than substitute for each other.

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
