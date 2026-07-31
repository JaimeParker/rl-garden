# Configuration System

rl-garden uses [tyro](https://brentyi.github.io/tyro/) to parse CLI arguments
from Python dataclasses. Configuration is resolved from four sources in a fixed
priority order; understanding this order tells you exactly where to put a new
default and how to override it.

---

## Priority order (lowest → highest)

```
① Dataclass field defaults    (rl_garden/common/cli_args.py, env_args.py, training/*/_args.py)
② Subclass field overrides    (VisionSACTrainingArgs over SACTrainingArgs)
③ RLG_* environment variables (logging fields only)
④ Explicit CLI flags          (highest priority — always wins)
```

### ① Dataclass field defaults

All hyperparameters live as Python dataclass fields with explicit default
values. The inheritance chain for a typical online algorithm is:

```
LoggingArgs          ← log_dir, log_type, wandb_*, eval_freq, ...
└─ EnvRunArgs        ← env_id, num_envs, seed, control_mode, ...
   └─ CheckpointArgs ← checkpoint_freq, load_checkpoint, save_final_checkpoint, ...
      └─ SACTrainingArgs   ← total_timesteps, buffer_size (1M), batch_size (1024), utd (0.5), ...
         └─ VisionSACTrainingArgs (+ VisionArgs)
```

Key shared mixin files:

| File | Dataclasses |
|------|-------------|
| `rl_garden/common/cli_args.py` | `LoggingArgs`, `CheckpointArgs`, `VisionArgs` |
| `rl_garden/common/env_args.py` | `EnvRunArgs`, `EnvBackendArgs`, `ManiSkillConfig`, `RoboTwinConfig` |
| `rl_garden/training/online/_args.py` | `SACTrainingArgs`, `VisionSACTrainingArgs`, `PPOTrainingArgs`, ... |
| `rl_garden/training/offline/_args.py` | `OfflineCommonArgs`, algorithm-specific mixin args |
| `rl_garden/training/off2on/_args.py` | `WSRLTrainingArgs`, `VisionWSRLTrainingArgs`, ... |

### ② Subclass field overrides

When the same algorithm has meaningfully different resource budgets across
observation modalities, the pattern is to define a base class with state-obs
defaults and a `Vision*` subclass that overrides the fields that need tightening
for GPU memory:

```python
@dataclass
class SACTrainingArgs(EnvRunArgs, CheckpointArgs):
    buffer_size: int = 1_000_000   # state obs: large buffer is cheap
    batch_size: int = 1024
    utd: float = 0.5

@dataclass
class VisionSACTrainingArgs(SACTrainingArgs, VisionArgs):
    buffer_size: int = 200_000     # visual obs: images eat VRAM
    batch_size: int = 512
    utd: float = 0.25
```

`SACArgs` (the CLI entry point) inherits from `VisionSACTrainingArgs`, so visual
defaults apply by default. Passing `--obs_mode state` on the CLI does not
automatically switch to `SACTrainingArgs`; the user must also increase
`--buffer_size` if they want state-obs resource budgets.

### ③ `RLG_*` environment variables

A small set of logging fields can be set through the shell environment without
touching code. The registry applies these **before** tyro parses argv, so they
act as user-level defaults that explicit CLI flags can still override.

| Variable | Field | Default |
|---|---|---|
| `RLG_STD_LOG` | `std_log` | `True` |
| `RLG_LOG_TYPE` | `log_type` | `"wandb"` |
| `RLG_LOG_KEYWORDS` | `log_keywords` | `None` |
| `RLG_WANDB_PROJECT` | `wandb_project` | `"rl-garden"` |
| `RLG_WANDB_ENTITY` | `wandb_entity` | `None` |
| `RLG_WANDB_GROUP` | `wandb_group` | `None` |

```bash
# Example: redirect all runs to tensorboard during local debugging
export RLG_LOG_TYPE=tensorboard
export RLG_STD_LOG=1
python examples/train_online.py sac --env_id PickCube-v1
```

Shell scripts in `scripts/` propagate these as `exec env RLG_LOG_TYPE="$LOG_TYPE" ...`
so that values parsed from the script's own flags reach the Python process.

### ④ Explicit CLI flags

Tyro builds a subcommand CLI from the registered `Args` dataclasses. All fields
are exposed directly:

```bash
python examples/train_online.py sac \
    --env_id StackCube-v1 \
    --num_envs 32 \
    --buffer_size 500000 \
    --gamma 0.99
```

Backend sub-configs use a dot-separated prefix matching the field name in
`EnvBackendArgs`:

```bash
python examples/train_online.py sac \
    --env_backend maniskill \
    --maniskill.sim-backend physx_cpu \
    --maniskill.reward-mode normalized_dense

python examples/train_online.py sac \
    --env_backend robotwin \
    --robotwin.task_name pick_cube \
    --robotwin.reward_mode sparse
```

When running through a `scripts/` launcher, any flags appended after the script
invocation are forwarded as `FORWARD_ARGS` and take final precedence:

```bash
# Override the script's default --env_id and add a custom --gamma:
./scripts/train_sac_rgbd.sh --env_id StackCube-v1 --gamma 0.95
```

---

## Inspecting the final merged config

Before launching a real training run, inspect all hyperparameters with
`--print-config`. This resolves the full priority stack — dataclass defaults,
subclass overrides, env vars, and any CLI flags you pass — and prints the
result as JSON. No environments, agents, or buffers are created.

```bash
python examples/train_online.py sac \
    --env_id StackCube-v1 \
    --maniskill.sim-backend physx_cpu \
    --print-config 2>/dev/null | python -m json.tool
```

Output format:
```json
{
  "training_phase": "online",
  "algorithm": "sac",
  "args": {
    "env_id": "StackCube-v1",
    "buffer_size": 200000,
    "batch_size": 512,
    "utd": 0.25,
    "maniskill": { "sim_backend": "physx_cpu", ... },
    ...
  }
}
```

The same JSON is written to `runs/<run_name>/config.json` at the start of every
training run by `persist_resolved_config()` (in `rl_garden/common/resolved_config.py`).
This file is a record for reproducibility; it is not read back by subsequent runs.

---

## Where to put a new default

Use this decision tree:

**Is it an algorithm hyperparameter** (`gamma`, `utd`, `buffer_size`, `my_lr`...)?
→ Add it as a field with a default in the algorithm's `TrainingArgs` dataclass
  in the relevant `_args.py`. For visual-obs-specific tuning, override the field
  in the `Vision*TrainingArgs` subclass.

**Is it observation-modality-specific** (different resource budget for visual vs. state)?
→ Define the default in the base `TrainingArgs`, then override it in
  `Vision*TrainingArgs`. Do not branch on `args.obs_mode` in the training code.

**Is it an experiment preset** (which task to train on, how many envs, logging sink)?
→ Put it as a hardcoded CLI flag in a `scripts/*.sh` launcher. Users override it
  by appending their own flags.

**Is it a logging / W&B setting** that applies across all experiments for a user?
→ Set the corresponding `RLG_*` environment variable in the shell profile.

**Rules that must not be violated:**
- Do not hardcode algorithm hyperparameters (`gamma`, `utd`, network widths) in
  `scripts/`. Scripts are experiment recipes; the dataclass is the algorithm
  source of truth.
- Do not set `learning_starts`, `batch_size`, or similar in a `scripts/` file
  unless you are consciously creating an experiment-specific recipe that deviates
  from the algorithm's validated default.
- Do not duplicate a default across dataclass and script. If the script needs
  a non-default value, that value belongs in the script. If it is the correct
  default for the algorithm, it belongs in the dataclass (and the script should
  omit the flag entirely).

---

## Writing a new launcher script

If you need a reproducible experiment preset for a new task or modality,
follow the pattern in `scripts/train_sac_rgbd.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Capture RLG_* from the environment (with script-level defaults)
STD_LOG="${RLG_STD_LOG:-1}"
LOG_TYPE="${RLG_LOG_TYPE:-wandb}"
LOG_KEYWORDS="${RLG_LOG_KEYWORDS:-}"

# 2. Parse --std_log / --log_type / --log_keywords from $@,
#    accumulate everything else in FORWARD_ARGS
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --std_log) STD_LOG="$2"; shift 2 ;;
        --log_type) LOG_TYPE="$2"; shift 2 ;;
        --log_keywords) LOG_KEYWORDS="$2"; shift 2 ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

# 3. Launch with hardcoded experiment defaults + FORWARD_ARGS last
#    (FORWARD_ARGS appear after hardcoded flags so they override them)
exec env RLG_STD_LOG="$STD_LOG" RLG_LOG_TYPE="$LOG_TYPE" RLG_LOG_KEYWORDS="$LOG_KEYWORDS" \
    python examples/train_online.py sac \
    --env_id MyTask-v1 \
    --obs_mode rgbd \
    --num_envs 16 \
    --log_type "$LOG_TYPE" \
    "${FORWARD_ARGS[@]}"
```

Key points:
- `FORWARD_ARGS` appended last ensures user-supplied flags always win.
- `exec env RLG_* python ...` propagates logging env vars into the child process.
- Only set flags that deviate from the dataclass defaults or that define the
  experiment (task, obs mode, env count).
