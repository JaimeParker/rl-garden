#!/usr/bin/env bash
# PPO launcher for RoboTwin place_empty_cup with 64x64 RGB observations.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

ROBOTWIN_ROOT="${RLG_ROBOTWIN_ROOT:-}"
if [[ -z "$ROBOTWIN_ROOT" ]]; then
    echo "Error: set RLG_ROBOTWIN_ROOT to the RoboTwin checkout path." >&2
    exit 1
fi
ASSETS_PATH_ARG="${RLG_ROBOTWIN_ASSETS_PATH:-$ROBOTWIN_ROOT}"

exec env \
    HOME="${HOME:-/tmp}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" \
    ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}" \
    "$PYTHON_BIN" -u "$REPO_DIR/examples/train_online.py" ppo \
    --config "$REPO_DIR/configs/online/ppo_robotwin_place_empty_cup_rgb.yaml" \
    --robotwin.robotwin-root "$ROBOTWIN_ROOT" \
    --robotwin.assets-path "$ASSETS_PATH_ARG" \
    "$@"
