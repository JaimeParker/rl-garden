#!/usr/bin/env bash
# Generic visual ResidualSAC launcher. Pass env-specific options explicitly.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

exec env PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -u "$REPO_DIR/examples/train_online.py" residual_sac \
    --config "$REPO_DIR/configs/online/residual_sac_rgb.yaml" \
    "$@"
