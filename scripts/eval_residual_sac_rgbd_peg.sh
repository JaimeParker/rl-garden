#!/usr/bin/env bash
# Evaluate a ResidualSAC RGBD checkpoint on PegInsertionSidePegOnly-v1 and
# record [base_camera | hand_camera] eval video from policy observations.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

exec env PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -u "$REPO_DIR/examples/eval_residual_sac_rgbd_peg.py" "$@"
