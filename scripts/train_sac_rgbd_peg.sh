#!/usr/bin/env bash
# RGBD SAC launcher for PegInsertionSidePegOnly-v1 with GPU defaults.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

exec "$PYTHON_BIN" "$REPO_DIR/examples/train_sac_rgbd_peg.py" \
    --env_id PegInsertionSidePegOnly-v1 \
    --obs_mode rgb \
    --control_mode pd_ee_delta_pose \
    --camera_width 64 --camera_height 64 \
    --total_timesteps 1000000 \
    "$@"
