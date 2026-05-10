# Mutagen Sync SOP

This SOP defines the shared Mutagen workflow for rl-garden. Read `AGENTS.md`
first for the project-level agent rules. Personal SSH aliases, usernames, host
paths, and local checkout paths must stay in ignored local runtime files such
as `mutagen.yml` and `.agents/local/personal_config.md`, not in committed docs.

Use `mutagen.yml.example` as the copyable configuration skeleton.

## Local Setup

Create a local `mutagen.yml` from `mutagen.yml.example` and fill:

- `<mutagen-session-name>`: the sync session name used by local commands.
- `<local-project-path>`: local rl-garden checkout path.
- `<ssh-alias>`: SSH alias for the remote host.
- `<remote-project-path>`: remote host path that receives the synced project.

Keep local `mutagen.yml` ignored by git.

## Sync Direction

Use `mode: one-way-replica` for the code sync:

- `alpha` is the local checkout and source of truth.
- `beta` is the remote host checkout and should mirror local code.
- Do not edit source files directly under `beta` unless intentionally
  recovering or debugging sync state.

## Required Ignores

Keep these categories out of Mutagen sync:

- VCS metadata and local locks: `.git`, `mutagen.yml.lock`.
- Python/build caches: `__pycache__`, `*.pyc`, `.pytest_cache`, `dist/`, `build/`.
- External repos: `3rd_party/`.
- Local environments and agent configs: `.venv`, `venv`, `.codex`, `.agents/local/personal_config.md`.
- IDE files: `.cursor/`, `.vscode/`, `.idea/`, `.github/`.
- Training artifacts: `runs/`, `logs/`, `wandb/`, `checkpoints/`.

Only copy selected artifacts back manually when needed.

## Start And Verify

Start or repair the configured project sync:

```bash
cd <local-project-path>
mutagen daemon start
mutagen project start
mutagen sync list <mutagen-session-name>
```

Expected state:

- `Status: Watching for changes`.
- Both endpoints connected.
- No unresolved conflicts.

The repo helper may be used when present:

```bash
cd <local-project-path>
scripts/mutagen_ensure.sh
```

## Troubleshooting

Inspect the session:

```bash
mutagen sync list <mutagen-session-name> --long
```

If the session is stale or unhealthy, cycle only this project session:

```bash
mutagen sync terminate <mutagen-session-name>
mutagen project start
mutagen sync list <mutagen-session-name>
```

If files are missing on the remote, first check whether they are intentionally
ignored. Files under `3rd_party/`, `runs/`, `logs/`, `wandb/`, checkpoints, and
local runtime config files are not expected to sync.

## Remote Runtime Check

After sync is healthy, verify the remote/container workspace separately using
the remote training SOP:

```bash
ssh <ssh-alias> "docker exec <container-name> bash -lc 'cd <container-workspace-path> && git status --short | head'"
```
