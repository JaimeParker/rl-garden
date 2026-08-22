#!/usr/bin/env bash
# Selectively install one runnable JAX baseline (submodule + optional venv).
#
# Usage: scripts/install_baseline.sh <cal_ql|wsrl|iql_jax> [options]
#   --source-only     Only run `git submodule update --init`; skip venv
#                      creation entirely (for a local read-only checkout).
#   --python PATH      Interpreter to build the venv with. Default:
#                      `python<python_version>` from the manifest, resolved
#                      via PATH. Fails loudly if not found -- never silently
#                      falls back to the host default `python3`.
#   --venv-dir PATH     Full override of the venv's location. Default:
#                      `${RL_GARDEN_BASELINE_VENV_ROOT:-$HOME/.venvs/rl-garden-baselines}/<venv_name>`.
#   --force            Recreate the venv from scratch if it already exists.
#
# See .agents/runbooks/baseline-install.md and baselines/baselines.yaml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAME="${1:-}"
if [[ -z "${NAME}" ]]; then
    echo "usage: scripts/install_baseline.sh <cal_ql|wsrl|iql_jax> [--source-only] [--python PATH] [--venv-dir PATH] [--force]" >&2
    exit 1
fi
shift

SOURCE_ONLY=0
PYTHON_OVERRIDE=""
VENV_DIR_OVERRIDE=""
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-only) SOURCE_ONLY=1; shift ;;
        --python) PYTHON_OVERRIDE="$2"; shift 2 ;;
        --venv-dir) VENV_DIR_OVERRIDE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
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
PYTHON_VERSION="$(field python_version)"
VENV_NAME="$(field venv_name)"
D4RL_FORK="$(field d4rl_fork)"

echo "[install-baseline] submodule: ${REL_PATH}"
git -C "${REPO_ROOT}" submodule update --init "${REL_PATH}"

PINNED_COMMIT="$(git -C "${REPO_ROOT}/${REL_PATH}" rev-parse HEAD)"
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
