#!/usr/bin/env bash
# State-based SAC launcher for ManiSkill.
# Forward all args to the python script; sensible defaults are in Args.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
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

exec env RLG_STD_LOG="$STD_LOG" RLG_LOG_TYPE="$LOG_TYPE" RLG_LOG_KEYWORDS="$LOG_KEYWORDS" "$PYTHON_BIN" -u "$REPO_DIR/examples/train_sac_state.py" \
    --env_id PickCube-v1 \
    --num_envs 16 \
    --capture_video \
    --video_fps 30 \
    --render_mode rgb_array \
    --eval_freq 10000 \
    --total_timesteps 1000000 \
    "${FORWARD_ARGS[@]}"
