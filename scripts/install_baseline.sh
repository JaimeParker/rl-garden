#!/usr/bin/env bash
# Selectively install one runnable JAX baseline (submodule + optional venv).
#
# Usage: scripts/install_baseline.sh <cal_ql|wsrl|iql_jax> [options]
#   --source-only     Only materialize the submodule source; skip venv
#                      creation entirely (for a local read-only checkout).
#   --python PATH      Interpreter to build the venv with. Default:
#                      `python<python_version>` from the manifest, resolved
#                      via PATH. Fails loudly if not found -- never silently
#                      falls back to the host default `python3`.
#   --venv-dir PATH     Full override of the venv's location. Default:
#                      `${RL_GARDEN_BASELINE_VENV_ROOT:-$HOME/.venvs/rl-garden-baselines}/<venv_name>`.
#   --force            Recreate the venv from scratch if it already exists.
#   --pinned-commit SHA  Required when REPO_ROOT has no `.git` (e.g. a Mutagen
#                      one-way-mirrored remote checkout -- `.git` is never
#                      synced, see .agents/rules/mutagen-sync-sop.md). Falls
#                      back to `git clone <remote_url>` + `git checkout SHA`
#                      instead of `git submodule update --init`. Get SHA from
#                      `git ls-tree HEAD <path>` on a real checkout.
#
# See .agents/runbooks/baseline-install.md and baselines/baselines.yaml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAME="${1:-}"
if [[ -z "${NAME}" ]]; then
    echo "usage: scripts/install_baseline.sh <cal_ql|wsrl|iql_jax> [--source-only] [--python PATH] [--venv-dir PATH] [--force] [--pinned-commit SHA]" >&2
    exit 1
fi
shift

SOURCE_ONLY=0
PYTHON_OVERRIDE=""
VENV_DIR_OVERRIDE=""
FORCE=0
PINNED_COMMIT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-only) SOURCE_ONLY=1; shift ;;
        --python) PYTHON_OVERRIDE="$2"; shift 2 ;;
        --venv-dir) VENV_DIR_OVERRIDE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --pinned-commit) PINNED_COMMIT_OVERRIDE="$2"; shift 2 ;;
        *) echo "[install-baseline] unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "[install-baseline] resolving manifest entry for '${NAME}'"
MANIFEST_JSON="$(cd "${REPO_ROOT}" && python3 -c "
from baselines.core.manifest import get_baseline
import dataclasses, json, sys
print(json.dumps(dataclasses.asdict(get_baseline(sys.argv[1]))))
" "${NAME}")"

field() {
    python3 -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "${MANIFEST_JSON}" "$1"
}

REL_PATH="$(field path)"
REMOTE_URL="$(field remote_url)"
PYTHON_VERSION="$(field python_version)"
VENV_NAME="$(field venv_name)"
D4RL_FORK="$(field d4rl_fork)"

SUBMODULE_DIR="${REPO_ROOT}/${REL_PATH}"

if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "[install-baseline] submodule: ${REL_PATH}"
    git -C "${REPO_ROOT}" submodule update --init "${REL_PATH}"
else
    # No parent .git -- e.g. a Mutagen one-way-mirrored remote checkout, which
    # never syncs .git (see .agents/rules/mutagen-sync-sop.md). git-submodule
    # tooling cannot run here, so clone the submodule directly instead.
    echo "[install-baseline] ${REPO_ROOT} has no .git -- git-submodule tooling unavailable, falling back to direct clone"
    if [[ -z "${PINNED_COMMIT_OVERRIDE}" ]]; then
        echo "[install-baseline] pass --pinned-commit SHA (from 'git ls-tree HEAD ${REL_PATH}' on a real checkout)" >&2
        exit 1
    fi
    if [[ -e "${SUBMODULE_DIR}" && ! -d "${SUBMODULE_DIR}/.git" ]]; then
        echo "[install-baseline] ${SUBMODULE_DIR} already exists and is not a git checkout -- refusing to overwrite." >&2
        echo "[install-baseline] move it aside first; it may be pre-existing reference content." >&2
        exit 1
    fi
    if [[ ! -d "${SUBMODULE_DIR}" ]]; then
        echo "[install-baseline] cloning ${REMOTE_URL} into ${REL_PATH}"
        git clone --quiet "${REMOTE_URL}" "${SUBMODULE_DIR}"
    fi
    echo "[install-baseline] checking out ${PINNED_COMMIT_OVERRIDE}"
    git -C "${SUBMODULE_DIR}" checkout --quiet "${PINNED_COMMIT_OVERRIDE}"
fi

PINNED_COMMIT="$(git -C "${SUBMODULE_DIR}" rev-parse HEAD)"
echo "[install-baseline] pinned commit: ${PINNED_COMMIT}"

if [[ "${SOURCE_ONLY}" -eq 1 ]]; then
    echo "[install-baseline] --source-only: done (no venv built)"
    exit 0
fi

VENV_ROOT="${RL_GARDEN_BASELINE_VENV_ROOT:-${HOME}/.venvs/rl-garden-baselines}"
VENV_DIR="${VENV_DIR_OVERRIDE:-${VENV_ROOT}/${VENV_NAME}}"

PYTHON_BIN="${PYTHON_OVERRIDE:-python${PYTHON_VERSION}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[install-baseline] '${PYTHON_BIN}' not found on PATH." >&2
    echo "[install-baseline] install python${PYTHON_VERSION} first (e.g. via pyenv/conda) or pass --python." >&2
    exit 1
fi

if [[ -d "${VENV_DIR}" && "${FORCE}" -eq 1 ]]; then
    echo "[install-baseline] --force: removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[install-baseline] creating venv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "[install-baseline] reusing existing venv at ${VENV_DIR}"
fi

VENV_PIP="${VENV_DIR}/bin/pip"
"${VENV_PIP}" install -U pip

while IFS= read -r req; do
    [[ -z "${req}" ]] && continue
    echo "[install-baseline] pip install -r ${REL_PATH}/${req}"
    "${VENV_PIP}" install -r "${REPO_ROOT}/${REL_PATH}/${req}"
done <<< "$(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])['requirements_files']))" "${MANIFEST_JSON}")"

while IFS= read -r extra; do
    [[ -z "${extra}" ]] && continue
    echo "[install-baseline] pip install ${extra}"
    "${VENV_PIP}" install "${extra}"
done <<< "$(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])['extra_pip']))" "${MANIFEST_JSON}")"

if [[ "${D4RL_FORK}" != "None" && -n "${D4RL_FORK}" ]]; then
    echo "[install-baseline] pip install ${D4RL_FORK}"
    "${VENV_PIP}" install "${D4RL_FORK}"
fi

MUJOCO_DIR="${MUJOCO_PY_MUJOCO_PATH:-${HOME}/.mujoco/mujoco210}"
if [[ ! -d "${MUJOCO_DIR}" ]]; then
    echo "[install-baseline] MuJoCo 2.1 not found at ${MUJOCO_DIR}." >&2
    echo "[install-baseline] see docs/guides/d4rl-legacy-expansion.md and .agents/local/9990.md for setup steps." >&2
    exit 1
fi

INVOCATION_CWD="$(field invocation_cwd)"
echo "[install-baseline] done."
echo "[install-baseline]   venv: ${VENV_DIR}"
echo "[install-baseline]   never 'pip install -e' ${REL_PATH} -- it must stay read-only"
if [[ "${INVOCATION_CWD}" != "None" ]]; then
    echo "[install-baseline]   run orchestrators with cwd=${INVOCATION_CWD} (or sys.path.insert equivalent)"
fi
