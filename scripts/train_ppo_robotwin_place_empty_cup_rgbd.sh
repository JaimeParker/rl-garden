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
STD_LOG="${RLG_STD_LOG:-1}"
LOG_TYPE="${RLG_LOG_TYPE:-wandb}"
LOG_KEYWORDS="${RLG_LOG_KEYWORDS:-}"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --std_log|--std-log)
            STD_LOG=1
            shift
            ;;
        --no_std_log|--no-std-log)
            STD_LOG=0
            shift
            ;;
        --log_type|--log-type)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            LOG_TYPE="$2"
            shift 2
            ;;
        --log_type=*|--log-type=*)
            LOG_TYPE="${1#*=}"
            shift
            ;;
        --log_keywords|--log-keywords)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            LOG_KEYWORDS="$2"
            shift 2
            ;;
        --log_keywords=*|--log-keywords=*)
            LOG_KEYWORDS="${1#*=}"
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
    RLG_STD_LOG="$STD_LOG" \
    RLG_LOG_TYPE="$LOG_TYPE" \
    RLG_LOG_KEYWORDS="$LOG_KEYWORDS" \
    "$PYTHON_BIN" -u "$REPO_DIR/examples/train_ppo_robotwin_rgbd.py" \
    --env-id place_empty_cup \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --camera-width 64 \
    --camera-height 64 \
    --num-envs 4 \
    --num-eval-envs 2 \
    --total-timesteps 1000000 \
    --num-steps 16 \
    --step-lim 200 \
    --embodiment piper piper 0.6 \
    --no-collect-wrist-camera \
    --encoder plain_conv \
    --image-fusion-mode per_key \
    --reward-mode dense \
    --control-mode delta_joint_pos \
    "${FORWARD_ARGS[@]}"
