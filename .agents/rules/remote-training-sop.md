# Remote Training SOP

This SOP defines the shared remote execution workflow for rl-garden. Read
`AGENTS.md` first for the project-level agent rules. Personal server names,
usernames, absolute home paths, container names, credentials, and private
artifact locations must stay out of committed docs.

Before running training, evaluation, or remote debugging, load environment
bindings from `.agents/local/personal_config.md`. This file is local-only and
ignored by git; create it from `.agents/local/personal_config.md.example` when
setting up a new checkout. If the file is missing, do not guess server,
container, path, or Python environment values.

## Required Local Bindings

The active local runtime must provide:

- `<ssh-alias>`: SSH config alias for the remote host.
- `<local-project-path>`: local checkout path.
- `<remote-project-path>`: path where local edits sync on the remote host.
- `<remote-bind-mount-path>`: host path mounted into the container.
- `<container-name>`: Docker container used for rl-garden runs.
- `<container-workspace-path>`: project path inside the container.
- `<python-env-bin-path>`: directory containing the intended Python executable.
- `<mutagen-session-name>`: Mutagen session name, if Mutagen is used.

## Execution Model

rl-garden training is GPU-first. Use the local machine for code editing and run
training, evaluation, and long smoke tests in the configured remote/container
environment unless the task explicitly calls for a local CPU fallback.

Normal workflow:

1. Edit locally.
2. Sync code to the remote host, for example with Mutagen.
3. Run inside the remote container from the bind-mounted workspace.
4. Keep high-churn artifacts such as `runs/`, `logs/`, `wandb/`, and
   checkpoints out of git and out of broad sync rules.

The container should see the remote host project through a bind mount. Do not
copy the repo into the container with `docker cp` or a tar stream unless the
bind mount is intentionally absent or broken.

## Health Checks

Check sync state before running experiments:

```bash
cd <local-project-path>
scripts/mutagen_ensure.sh
mutagen sync list <mutagen-session-name>
```

Check remote GPU and tmux state:

```bash
ssh <ssh-alias> "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
ssh <ssh-alias> "tmux ls 2>/dev/null || true"
```

Check that the container sees the synced workspace:

```bash
ssh <ssh-alias> "docker exec <container-name> bash -lc 'cd <container-workspace-path> && pwd && git status --short | head'"
```

## Command Pattern

Run commands inside the container with the intended Python environment first on
`PATH`:

```bash
ssh <ssh-alias> "docker exec <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  MPLCONFIGDIR=/tmp <command>
'"
```

Examples:

```bash
# State SAC
ssh <ssh-alias> "docker exec <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  MPLCONFIGDIR=/tmp scripts/train_sac_state.sh --env_id PickCube-v1
'"

# Generic RGB SAC
ssh <ssh-alias> "docker exec <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  MPLCONFIGDIR=/tmp scripts/train_sac_rgbd.sh --encoder plain_conv
'"

# Peg-only RGB SAC
ssh <ssh-alias> "docker exec <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  MPLCONFIGDIR=/tmp scripts/train_sac_rgbd_peg.sh
'"
```

## Long Runs

Use remote tmux for long-running jobs and write logs on the remote host:

```bash
ssh <ssh-alias> "mkdir -p <remote-project-path>/logs && \
  tmux new-session -d -s <session-name> \
  \"docker exec -e CUDA_VISIBLE_DEVICES=<gpu-id> <container-name> bash -lc 'cd <container-workspace-path> && export PATH=<python-env-bin-path>:\\\$PATH && MPLCONFIGDIR=/tmp <command>' 2>&1 | tee <remote-project-path>/logs/<session-name>_\$(date +%Y%m%d_%H%M%S).log\""
```

Inspect logs remotely:

```bash
ssh <ssh-alias> "tail -f <remote-project-path>/logs/<session-name>_*.log"
```

## Artifacts

Training outputs under `runs/`, `logs/`, `wandb/`, and checkpoint directories
should remain untracked. Copy back only selected artifacts when needed instead
of enabling broad sync for generated output directories.

If the container writes root-owned artifacts on the remote host, fix ownership
only for generated output directories:

```bash
ssh <ssh-alias> "sudo chown -R <remote-user>:<remote-group> <remote-project-path>/runs <remote-project-path>/logs <remote-project-path>/wandb 2>/dev/null || true"
```
