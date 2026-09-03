# robomimic Integration

This guide covers installing the `robomimic` env backend, downloading its
datasets, and running RLPD offline-to-online training against them. It
complements [`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md),
which this backend follows structurally (mirrors the already-shipped
`d4rl_legacy` backend).

`robomimic` (ARISE-Initiative) wraps `robosuite`/MuJoCo single-arm
manipulation tasks with an old-style 4-tuple gym API and HuggingFace-hosted
demonstration datasets. Scope of this integration: `low_dim` (state)
observations only, `ph`/`mh` dataset types, single-arm tasks (`lift`, `can`,
`square`, `transport`, `tool_hang`). Image observations and the `mg`
dataset type are out of scope for now.

## Install

```bash
python -m pip install -e '.[robomimic]'
```

The `robomimic` extra pins two things beyond what upstream publishes on
PyPI, both found by actually installing and running the stack, not just
reading docs:

- **`robomimic` itself, pinned to git tag `v0.5.0`.** Plain
  `pip install robomimic` installs `0.3.0`, a pre-HuggingFace-migration
  release with no `HF_REPO_ID` (only the older `DATASET_REGISTRY`) — the
  dataset download command below needs `v0.5.0`+.
- **`mujoco>=3.3.0,<3.10`.** Published `robosuite==1.5.2` metadata only
  declares `mujoco>=3.3.0` (no upper bound). robosuite's controller code
  breaks on `mujoco>=3.10` (a `mj_fullM` signature change) — this repo hit
  that exact crash (`AssertionError` in
  `robosuite/utils/binding_utils.py:get_joint_qpos_addr`) with plain
  `pip install robosuite` pulling `mujoco==3.12.0`, fixed by pinning the
  same upper bound robosuite's own (unreleased-to-PyPI) `setup.py` uses.

## Downloading datasets

```bash
python -m robomimic.scripts.download_datasets \
  --tasks lift \
  --dataset_types ph \
  --hdf5_types low_dim
```

- `--tasks`: one or more of `lift`, `can`, `square`, `transport`,
  `tool_hang` (plus others upstream hosts; these five are what this
  integration has been scoped/tested against).
- `--dataset_types`: `ph` (proficient-human) or `mh` (multi-human).
  `mg` (machine-generated) is out of scope — its sparse/dense reward
  variant selection hasn't been verified against this backend's
  metadata-driven env construction.
- `--hdf5_types`: use `low_dim` for this integration. `image` hdf5 files
  are not directly hosted for these tasks and need local generation via
  robomimic's `dataset_states_to_obs.py` — out of scope here.

Files download by default under `<robomimic_repo>/datasets/<task>/<dataset_type>/`
(pass `--download_dir` to redirect). They are small: `lift/ph/low_dim_v15.hdf5`
is ~21MB (200 demos, 9666 transitions) — cheap enough to download in full
rather than truncating. Larger tasks (`transport`, `tool_hang`) are bigger
but still MB-scale for `low_dim`, not GB-scale.

Keep downloaded datasets outside the repository, same convention as
`d4rl_legacy` (see
[D4RL Legacy Manipulation Baselines](d4rl-legacy-expansion.md)).

## Usage

```python
from rl_garden.envs.robomimic import RobomimicEnvConfig, make_robomimic_env

env = make_robomimic_env(
    RobomimicEnvConfig(
        env_id="Lift",
        num_envs=4,
        dataset_path="/path/to/lift/ph/low_dim_v15.hdf5",
        device="cpu",
    )
)
```

When `dataset_path` is set, the env's robot/controller/camera config is
read directly from the dataset's own `env_args` metadata
(`robomimic.utils.file_utils.get_env_metadata_from_dataset`), guaranteeing
the online env matches whatever prior data it's paired with — this is the
whole point of metadata-driven construction for offline-to-online mixing.
If `env_id` and the dataset's own `env_name` disagree, construction raises
rather than silently picking one. Omitting `dataset_path` falls back to
`env_id` + `--robomimic.env-kwargs-json` (a JSON-encoded `env_kwargs` dict,
same escape-hatch convention as `ManiSkillConfig.env_kwargs_json`).

RLPD end-to-end (env backend + offline prior data from the same file):

```bash
python examples/train_online.py rlpd \
  --env-backend robomimic --env-id Lift \
  --robomimic.dataset-path /path/to/lift/ph/low_dim_v15.hdf5 \
  --obs-mode state \
  --dataset-backend robomimic \
  --offline-dataset /path/to/lift/ph/low_dim_v15.hdf5 \
  --num-envs 4 --num-eval-envs 2 \
  --total-timesteps 100000 --learning-starts 1000 --batch-size 256
```

This exact shape (smaller step counts) was run as a real end-to-end smoke
test — actual robosuite/MuJoCo sim, actual downloaded dataset, full RLPD
training loop to completion — during implementation.

### Key config fields (`--robomimic.<field>`)

- `dataset_path`: hdf5 path; drives metadata-based env construction (see
  above).
- `horizon`: episode length before `TimeLimit` truncation. Default `400`
  (robomimic's own standard for these tasks; the dataset's `env_args` has
  no `horizon` key of its own).
- `terminate_on_success`: default `False`. robosuite's `EnvRobosuite`
  always rolls out to the fixed horizon (`is_done()` unconditionally
  returns `False`, and `ignore_done=True` is force-set internally) — there
  is no sim-native termination signal. With the default, `terminated` is
  always `False` online and the offline loader's `dones` are all zeros to
  match (keeps RLPD's online/offline bootstrap semantics consistent).
  Setting this to `True` flips *both* sides together: online gets a
  success-termination wrapper, offline derives `dones` from the success
  signal instead of the raw per-demo end marker.
- `env_kwargs_json`: fallback env construction when `dataset_path` is
  unset.

## Tests

```bash
pytest -q \
  tests/test_robomimic_dataset.py \
  tests/test_robomimic_env.py \
  tests/test_robomimic_obs_parity.py \
  tests/test_prior_data_replay.py
```

These use synthetic hdf5 fixtures and a monkeypatched fake env — no
network access or robosuite/MuJoCo install required.
`test_robomimic_obs_parity.py` is the highest-value test: it asserts the
online env's flattened observation and the offline loader's flattened
observation use the exact same key order, since a silent divergence there
would misalign RLPD's mixed batches without any other test catching it
(trains, loss goes down, policy never actually improves).

## Current limits

- `low_dim` only; no `image` observation support yet.
- `mg` dataset type untested (sparse/dense reward variant selection not
  verified).
- `can`/`square`/`transport`/`tool_hang` are architecturally supported
  (same hdf5 schema, same metadata-driven construction) but only `lift`
  has been run end to end so far.
- No confirmed `seed()`-equivalent on robosuite's env for
  `reset(seed=...)` — seed is accepted by the adapter but not forwarded.
