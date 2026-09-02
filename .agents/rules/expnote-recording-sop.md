# expnote Recording SOP

This SOP defines how training runs in this repository get recorded in `expnote`
(local-first experiment notes, SQLite-backed, Obsidian Markdown as a
projection only — SQLite is authoritative). Read `AGENTS.md` first for the
project-level agent rules, and `.agents/rules/remote-training-sop.md` before
any remote command referenced here.

`expnote` itself lives in a separate repository (`~/Projects/expnote`, not
part of this checkout). Its core schema (MOC / topic / run / doc / artifact /
relation / benchmark, with a free-form `metadata` JSON object on each run) is
project-agnostic by design; rl-garden-specific integration lives in an
adapter (`expnote/adapters/rlgarden.py`), not in the core tool.

## Hierarchy

```
MOC (e.g. "Baseline MOC")
 └─ Topic (one experiment batch, e.g. "260813-Cal-QL Adroit Binary + Kitchen legacy D4RL seed-0")
     └─ Run ×N (one entry per training process)
         ├─ Artifact (checkpoint/plot/file reference)
         └─ Relation (comparison/derivation link to another run)
 └─ Doc (cross-run analysis, linked via --run-id to the runs it covers)
 └─ Benchmark (task x algo matrix summarizing results across runs)
```

## 1. Before starting: pick the workspace explicitly

Do not assume a fixed workspace name. Run `expnote workspace list`, show the
candidates, and confirm the target workspace with the user before any write
operation. Multiple workspaces can coexist for unrelated experiment lines;
picking the wrong one silently pollutes someone else's records.

```bash
expnote workspace list
expnote workspace use <confirmed-workspace-name>
```

## 2. Starting a run

1. New batch of runs sharing one purpose → create or reuse a topic:
   ```bash
   expnote topic add "<YYMMDD-algo-scope[-seed]>" --moc-id <moc-id> --summary "<batch purpose and scope>"
   ```
2. One `run add`/`run create` per training process:
   - **`--run-id`**: if the run is tracked by wandb, use the wandb run id
     directly as the expnote run id — do not invent a separate readable id.
     The run's purpose already carries the descriptive intent; duplicating it
     into the id adds nothing and makes wandb/expnote/Obsidian lookups
     diverge. If a run is not wandb-tracked, do not guess a convention —
     confirm one with the user before creating the run.
   - `--purpose`: one line, e.g. `"Phase <N> seed-<K> <implementation> <algorithm> <env> formal offline-to-online training"`.
   - `--topic-id`: the topic created/reused in step 1.
   - `--status running` (explicit, matches the default).
   - `--meta` (repeatable) or `--metadata-json`: see the field checklist below.

### Metadata field checklist

Required when applicable:

| Field | Meaning |
|---|---|
| `environment` | env/task id (e.g. `antmaze-medium-play-v2`) |
| `implementation` | which codebase produced the run (e.g. `rl-garden`, `official-calql-jax`) |
| `seed` | training seed |
| `gpu`, `host` | where the process is running |
| `tmux_session` | session name for reattaching |
| `log_path` | path to the raw training log on the host it ran on |
| `wandb_entity`, `wandb_project`, `wandb_run_id`, `wandb_url` | **required whenever the run is wandb-tracked** — `wandb_url` must be filled in, not left for later |

For official baselines, keep requested launch intent separate from observed
runtime behavior. Some upstream launchers ignore or rewrite fields such as
WandB project/group. In that case record both sides explicitly, e.g.
`requested_wandb_project=d4rl`, `requested_wandb_group=<requested-group>`,
`wandb_project=<actual-project>`, and `wandb_group=<actual-group>`. Also record
the upstream source revision (`<baseline>_commit`), the dedicated Python
environment, the resolved launcher command or `command.json`, and any necessary
runtime deviations from the upstream requirements (for example a CUDA/cuDNN
compatibility pin or missing dependency install).

Use consistent path fields when a run spans host/container/local locations:

| Field | Meaning |
|---|---|
| `log_path` | raw training log path on the host where the run executed |
| `remote_run_dir` or `output_dir` | host-side run/output directory |
| `container_run_dir` | in-container path for the same run directory, if different |
| `tensorboard_run_dir` | host-side TensorBoard event directory |
| `container_tensorboard_run_dir` | in-container TensorBoard path, if different |
| `remote_config_path` or `command_json` | host-side resolved config/command file |
| `container_config_path` | in-container resolved config path, if different |
| `final_checkpoint_path` | host-side final checkpoint path |

`import rlgarden <config.json> --topic <topic> [--run-id <wandb-run-id>]
[--wandb-url <url>]` is the preferred way to fill in `environment`/
`implementation`/`seed`/`log_dir`/`wandb_project`/`wandb_entity`/etc. from
rl-garden's resolved config. It reads the current `EffectiveConfig v2`
schema (`schema_version: 3`) and rejects any other `schema_version` with an
error rather than silently misparsing it — if it errors on `schema_version`,
the adapter needs updating in the `expnote` repo before importing further.

It does not cover everything, so always follow it with `run update --meta`
(or `--metadata-json`) for:
- `gpu`, `host`, `tmux_session` — deployment info, never part of
  `config.json`.
- `wandb_run_id` in `metadata` — `--run-id` sets the run's id but does not
  also copy the value into `metadata.wandb_run_id`; add it explicitly if a
  queryable copy is needed (e.g. `run query --where "metadata.wandb_run_id =
  ..."`).

## 3. During training

Leave `status: running` untouched during the run. Update it manually on
completion/failure (`expnote run update <id> --status finished|failed|stopped`).
`expnote sync wandb-status [--apply]` exists to suggest/apply status changes
from wandb directly — not yet adopted as mandatory in this workflow; treat it
as an optional cross-check until confirmed reliable, not a replacement for
reading a run's actual state.

Failed setup, import, dataset-loading, or launcher smoke attempts are not
training runs. Do not register them as ExpNote runs and do not attach their
logs to a later successful training run. Keep smoke/debug output in a separate
directory from the formal run output directory so failed-start artifacts cannot
be mistaken for part of the accepted training record.

## 4. After training: writing result and analysis

**Never write `result` or `analysis` from assumption.** "Training finished"
is not an analysis. Before writing either field, pull real evidence:

1. Prefer the wandb mcp tools (e.g. `get_run_history_tool`,
   `diagnose_run_tool`, `compare_runs_tool`), keyed off the run's
   `wandb_entity`/`wandb_project`/`wandb_run_id` metadata, to read actual
   summary/history metrics.
2. If wandb data is insufficient (run not wandb-tracked, metric missing,
   eval never completed), fall back to the real training log at
   `metadata.log_path` on the host recorded in `metadata.host`, per
   `.agents/rules/remote-training-sop.md`.
3. `--result`: one line, the core headline metric only (return / success rate
   / equivalent). No status words (status already lives in `--status`), no
   reasoning, no comparisons. If the metric isn't available yet, say so in
   one clause (e.g. "success rate not collected, see analysis") — do not
   leave it silently blank without explanation.
4. `--analysis` / `--append-analysis` (or `--analysis-file -` / a file for
   long text): convergence diagnostics, root causes, cross-run comparisons,
   caveats. Every claim here must trace back to a metric or log line actually
   pulled in step 1 or 2 — not inferred from the run's config or from what
   "should" have happened.

## 5. Cross-run organization

- Comparisons between two specific runs (e.g. rl-garden vs. official JAX,
  same env/seed): `expnote relation add <src> <dst> --kind <...> --note <...>`.
- Files worth keeping (checkpoints, plots): `expnote artifact add <run_id> <uri> --kind <...> --note <...>`.
- Task x algo summary tables: link runs into the relevant `benchmark` record,
  then `expnote benchmark matrix <benchmark-id>` to render it.
- Long-form analysis spanning multiple runs (not a single run's `--analysis`):
  `expnote doc add --doc-id <...> --moc-id <...> --title <...> --run-id <run1> --run-id <run2> --body-file -`.

## 6. Sync and validate

```bash
expnote sync all --json
expnote validate --json
```

Cross-device work uses `expnote workspace pack`/`unpack` since the SQLite
workspace directory is not part of any git checkout.
