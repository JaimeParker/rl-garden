# D4RL Legacy Manipulation Baselines

This campaign extends the Cal-QL offline-to-online entrypoint to the legacy
Adroit Binary, standard D4RL Adroit, and Kitchen tasks. It intentionally uses
the old Gym/D4RL stack; install the isolated optional dependencies with:

```bash
python -m pip install -e '.[d4rl-legacy-manip,wandb]'
```

Adroit Binary additionally requires the upstream Cal-QL `mj_envs` fork. Its
setuptools metadata omits `mj_envs.utils` from a normal wheel, so install the
fixed source checkout in editable mode as required by the upstream README:

```bash
git clone --recursive https://github.com/nakamotoo/mj_envs.git "$MJ_ENVS_ROOT"
git -C "$MJ_ENVS_ROOT" checkout 88fadb11c38e8141dfe7bbde92a8954858c4b9f2
git -C "$MJ_ENVS_ROOT" submodule update --init --recursive
python -m pip install -e "$MJ_ENVS_ROOT"
```

Set the MuJoCo 2.1 environment variables required by `mujoco-py` before
importing the environments. Keep raw datasets, converted H5 files, run logs,
and checkpoints outside the repository.

## Binary datasets

Download the official Cal-QL Adroit archive linked from the upstream README.
For each task, combine its expert and BC files into one generic trajectory H5:

```bash
python tools/conversion/convert_calql_adroit_binary.py \
  --task pen-binary-v0 \
  --expert-path "$DATA_ROOT/raw/pen2_sparse.npy" \
  --bc-path "$DATA_ROOT/raw/pen_bc_sparse4.npy" \
  --output-path "$DATA_ROOT/h5/pen-binary-v0.h5"
```

Repeat for `door-binary-v0` and `relocate-binary-v0`. The converter drops
trajectories without a success reward, truncates at the last `reward == 0`,
clips actions to `0.99999`, and records source hashes and trajectory provenance.
It preserves raw rewards; the checked-in Cal-QL presets apply `10r + 5` to both
offline and online rewards.

Run a Binary preset by supplying the machine-local H5 locator explicitly:

```bash
python examples/train_off2on.py calql \
  --config configs/off2on/calql_pen_binary_v0_paper.yaml \
  --offline-dataset "$DATA_ROOT/h5/pen-binary-v0.h5"
```

## Task matrix

- Paper protocol: `pen/door/relocate-binary-v0`, seeds 0-5.
- Paper protocol: `kitchen-{partial,mixed,complete}-v0`, seeds 0-5.
- Secondary CORL protocol: `pen/door/relocate-{human,cloned,expert}-v1`.
  Run seed 0 for all nine combinations, then expand every valid, non-degenerate
  run to seeds 0-3. These results are not Cal-QL paper reproductions.

Binary presets use 20K offline updates and 300K online steps for pen or 1M for
door/relocate. Kitchen presets use 500K offline updates and 1.25M online steps.
The standard Adroit presets use 1M offline and 1M online steps. Use the complete
checked-in preset as the protocol source rather than reconstructing flags from
this summary.

## Metrics and boundaries

- Binary reports success rate from `goal_achieved` and terminates immediately
  on success.
- Kitchen reports `num_stages_solved` and `normalized_score = stages / 4 * 100`.
- Standard Adroit reports the legacy D4RL normalized score.
- `bootstrap_at_done: truncated` stops TD bootstrap at task termination but
  retains bootstrap across TimeLimit truncation.

Each actual process is one expnote run. Record the effective config, command,
commit, dependency commits, dataset SHA256, machine, seed, artifact paths,
failures, and run-level analysis. Aggregate each task family with the mean and
95% Student-t confidence interval; retain invalid and stopped runs in the record.
