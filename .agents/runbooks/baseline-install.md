# JAX Baseline Install/Run Runbook

Use this runbook to install and run an official JAX baseline (Cal-QL, wsrl,
IQL-jax) for numeric comparison against rl-garden's own PyTorch ports. For
remote execution, also read [the remote training SOP](../rules/remote-training-sop.md)
and the ignored `.agents/local/personal_config.md` before running any command.

## Selecting a baseline

Three names are registered in [`baselines/baselines.yaml`](../../baselines/baselines.yaml):
`cal_ql`, `wsrl`, `iql_jax`. Each entry records its submodule path, pinned
JAX/Python version, required extra pip installs, its D4RL fork (they are
NOT all the same fork — see the manifest's `notes` field per entry), and its
`baselines/<name>/` orchestrator module.

`qgf` and `CORL` are explicitly out of scope for this system (`qgf` uses
OGBench with an incompatible CUDA/JAX stack; `CORL` is read-for-porting
only, never run standalone) — see the manifest's `reference:` section.

## Bootstrap from zero

```bash
# Source only (read-only reference use, no venv):
scripts/install_baseline.sh cal_ql --source-only

# Full install (submodule + dedicated venv + pinned deps):
scripts/install_baseline.sh cal_ql
scripts/install_baseline.sh wsrl
scripts/install_baseline.sh iql_jax
```

This is selective by design — never run `git submodule update --init
--recursive` on the whole repo. Each baseline gets its own venv
(`${RL_GARDEN_BASELINE_VENV_ROOT:-$HOME/.venvs/rl-garden-baselines}/<venv_name>`,
`/opt/venv/baseline-<name>` on remote hosts per the existing `/opt/venv/<purpose>`
convention) because their JAX/CUDA pins are mutually incompatible in any
shared environment (wsrl wants CUDA11/jax==0.4.20; other baselines pin
differently). Re-running without `--force` is a no-op on an existing venv
(just reinstalls/upgrades deps in place); `--force` recreates it.

MuJoCo 2.1 must already be installed (`$HOME/.mujoco/mujoco210` or
`MUJOCO_PY_MUJOCO_PATH`) before a full install — the script checks and exits
with a pointer to setup steps rather than auto-downloading binaries.

### On a Mutagen-mirrored remote host (no `.git`)

`git submodule update --init` requires a parent `.git`, but a Mutagen
one-way-mirrored checkout never has one (`.git` is a required sync ignore —
see [`mutagen-sync-sop.md`](../rules/mutagen-sync-sop.md)). On such a host the
script detects the missing `.git` and falls back to `git clone <remote_url>`
+ `git checkout`, which needs the pinned commit passed explicitly:

```bash
# Get the pinned commit from a real (git) checkout first:
git ls-tree HEAD 3rd_party/cal_ql   # -> 160000 commit <sha> 3rd_party/Cal-QL

scripts/install_baseline.sh cal_ql --source-only --pinned-commit <sha>
```

If the target directory already has non-git content there (e.g. an old ad
hoc reference copy), the script refuses to overwrite it — move it aside
first. This fallback only materializes the submodule *source*; it does not
change how the venv step works.

## Running an orchestrator

Orchestrators are always invoked as modules, from the repo root, with the
baseline's own venv python and the repo root on `PYTHONPATH`:

```bash
PYTHONPATH="$(pwd)" \
  ${RL_GARDEN_BASELINE_VENV_ROOT:-$HOME/.venvs/rl-garden-baselines}/baseline-calql/bin/python \
  -m baselines.cal_ql.run_offline \
  --calql-source 3rd_party/Cal-QL \
  --dataset <path-to-minari-derived-npz> \
  --output-dir <run-output-dir> \
  --env-python <minari-venv-python> \
  --minari-datasets-path <minari-datasets-dir>
```

This is the only supported invocation shape — there is no bare-script
fallback (`python baselines/cal_ql/run_offline.py ...` from an arbitrary
cwd will fail to import `baselines.core.*`).

**Process topology**: the orchestrator process (baseline's own JAX venv) ↔
imports policy/critic/loss code directly from the submodule via
`sys.path.insert` (Cal-QL) or runs with `cwd=<submodule path>` (wsrl,
IQL-jax — their own scripts import sibling modules relative to their own
directory) ↔ steps rl-garden's canonical environments through a THIRD
subprocess, `baselines.core.env_server` (its own Minari/gymnasium venv,
via `--env-python`), communicating over the length-prefixed binary protocol
in `baselines/core/wire_protocol.py`. This lets the official JAX code run
in its own JAX/CUDA stack without ever importing JAX and rl-garden's own
torch/gym stack in the same process.

`wsrl`'s and `iql_jax`'s orchestrators are not yet implemented (stub
packages only — see their module docstrings and `baselines/baselines.yaml`);
only `cal_ql`'s is runnable today.

## Do not modify or install-editable the submodules

The three submodules are read-only, same as every other entry under
`3rd_party/` (see `AGENTS.md`). Never `pip install -e` one — `wsrl` ships
its own `setup.py`; installing it editable would write `.egg-info` into the
submodule tree and dirty a "read-only" checkout. `scripts/install_baseline.sh`
only installs each baseline's *dependencies* into its dedicated venv and
runs orchestrators via `sys.path.insert` or `cwd`, never a package install
of the baseline itself.

## MuJoCo 2.1 prerequisite

See [`docs/guides/d4rl-legacy-expansion.md`](../../docs/guides/d4rl-legacy-expansion.md)
for the manual MuJoCo 2.1 binary install steps (all three baselines need it).

## Recording results

Use `.agents/rules/expnote-recording-sop.md`'s existing `implementation`
metadata field (e.g. `official-calql-jax` vs `rl-garden`) and its
`rl-garden vs official JAX, same env/seed` `relation` convention to record a
comparison run — no changes needed there, it already models this pattern.
