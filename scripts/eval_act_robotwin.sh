#!/usr/bin/env bash
# ACT-only RoboTwin evaluator for residual-SAC diagnostics.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

ROBOTWIN_ROOT="${RLG_ROBOTWIN_ROOT:-/home/RoboTwin}"
if [[ -z "$ROBOTWIN_ROOT" ]]; then
    echo "Error: set RLG_ROBOTWIN_ROOT to the RoboTwin checkout path." >&2
    exit 1
fi
ASSETS_PATH_ARG="${RLG_ROBOTWIN_ASSETS_PATH:-$ROBOTWIN_ROOT}"
CUDA_VISIBLE_DEVICES_ARG="${RLG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --assets_path|--assets-path)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            ASSETS_PATH_ARG="$2"
            shift 2
            ;;
        --assets_path=*|--assets-path=*)
            ASSETS_PATH_ARG="${1#*=}"
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

exec env \
    HOME="${HOME:-/tmp}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG" \
    ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}" \
    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u "$REPO_DIR/examples/eval_act_robotwin.py" \
    --env-id place_empty_cup \
    --base-ckpt-path pretrained_models/place_empty_cup.ckpt \
    --base-act-stats-path pretrained_models/dataset_stats_place_empty_cup.pkl \
    --num-eval-envs 1 \
    --num-eval-episodes 10 \
    --robotwin.robotwin-root "$ROBOTWIN_ROOT" \
    --robotwin.assets-path "$ASSETS_PATH_ARG" \
    --robotwin.random-background false \
    --robotwin.cluttered-table false \
    --robotwin.clean-background-rate 1.0 \
    --robotwin.random-head-camera-dis 0 \
    --robotwin.random-table-height 0 \
    --robotwin.step-lim 200 \
    --control-mode joint_pos \
    --camera-width 320 \
    --camera-height 240 \
    --capture-video true \
    --action-diagnostics true \
    --diagnostic-steps 20 \
    "${FORWARD_ARGS[@]}"
