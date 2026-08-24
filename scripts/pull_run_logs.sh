#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?usage: pull_run_logs.sh <run_name> [tensorboard-log-dest] [group] [ssh-alias] [remote-bind-mount-path] [local-project-path]}"
TB_LOG_DEST="${2:-}"
GROUP="${3:-}"
SSH_ALIAS="${4:-${RL_GARDEN_SSH_ALIAS:-}}"
REMOTE_BIND_MOUNT_PATH="${5:-${RL_GARDEN_REMOTE_BIND_MOUNT_PATH:-}}"
LOCAL_PROJECT_PATH="${6:-${RL_GARDEN_LOCAL_PROJECT_PATH:-$(pwd)}}"

if [[ -z "${SSH_ALIAS}" || -z "${REMOTE_BIND_MOUNT_PATH}" ]]; then
    echo "[pull-run-logs] missing ssh alias or remote bind-mount path" >&2
    echo "[pull-run-logs] pass them as args, or set RL_GARDEN_SSH_ALIAS / RL_GARDEN_REMOTE_BIND_MOUNT_PATH" >&2
    echo "[pull-run-logs] see .agents/local/personal_config.md for this machine's values" >&2
    exit 1
fi

pull() {
    local remote_dir="$1"
    local local_dir="$2"
    mkdir -p "${local_dir}"
    echo "[pull-run-logs] rsync ${SSH_ALIAS}:${remote_dir}/ -> ${local_dir}/"
    # --update (not --delete): additive pull for long-term storage, never
    # removes or overwrites a newer local file with an older remote one.
    rsync -avz --update "${SSH_ALIAS}:${remote_dir}/" "${local_dir}/"
}

REMOTE_RUNS_ROOT="${REMOTE_BIND_MOUNT_PATH%/}/runs"
LOCAL_CHECKPOINT_DIR="${LOCAL_PROJECT_PATH%/}/runs/${RUN_NAME}"

# Checkpoints and config.json always live flat at runs/<run_name>/ -- see
# .agents/runbooks/checkpoint.md's documented {log_dir}/{run_name}/ contract.
# This is unaffected by where TensorBoard logs get viewed from.
pull "${REMOTE_RUNS_ROOT}/${RUN_NAME}" "${LOCAL_CHECKPOINT_DIR}"

# TensorBoard events instead nest one extra level under log_group (see
# Logger.create in rl_garden/common/logger.py). The destination for these is
# caller-supplied rather than hardcoded to this repo's runs/ tree, since
# where they should live locally (e.g. under an expnote workspace directory,
# so expnote's own viewer can render them) is decided outside rl-garden.
#
# Pull the whole group directory, not just this run's leaf dir: expnote's
# tensorboard_dir viewer (event_multiplexer.AddRunsFromDirectory) reads every
# run subdirectory under the given path and disambiguates them in the chart
# legend as "<run>: <tag>", so pointing tensorboard_dir at a group directory
# gets multi-run comparison for free. Safe to re-run per-run: rsync --update
# only adds/refreshes this run's own subtree, never touches sibling runs.
if [[ -n "${TB_LOG_DEST}" && -n "${GROUP}" ]]; then
    pull "${REMOTE_RUNS_ROOT}/${GROUP}" "${TB_LOG_DEST}"
    echo "[pull-run-logs] tensorboard logs (group '${GROUP}'): ${TB_LOG_DEST}"
    echo "[pull-run-logs] register: expnote run update <run_id> --meta tensorboard_dir=\"${TB_LOG_DEST}\""
elif [[ -n "${TB_LOG_DEST}" || -n "${GROUP}" ]]; then
    echo "[pull-run-logs] need both <tensorboard-log-dest> and <group> to pull TensorBoard events -- skipped" >&2
else
    echo "[pull-run-logs] no tensorboard-log-dest/group given -- skipped runs/<group>/${RUN_NAME}/ (TensorBoard events)" >&2
    echo "[pull-run-logs] group defaults to env_id, or algorithm_offline_pretrain for offline runs" >&2
fi

echo "[pull-run-logs] checkpoints/config.json: ${LOCAL_CHECKPOINT_DIR}"
