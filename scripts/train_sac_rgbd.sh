#!/usr/bin/env bash
# RGBD SAC launcher with selectable image encoder.
# Examples:
#   scripts/train_sac_rgbd.sh                                 # plain_conv
#   scripts/train_sac_rgbd.sh --encoder resnet10
#   scripts/train_sac_rgbd.sh --encoder resnet10 --pretrained_weights resnet10-imagenet
set -euo pipefail

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

ENCODER="plain_conv"
for ((i = 0; i < ${#FORWARD_ARGS[@]}; i++)); do
    arg="${FORWARD_ARGS[$i]}"
    case "$arg" in
        --encoder)
            if (( i + 1 < ${#FORWARD_ARGS[@]} )); then
                ENCODER="${FORWARD_ARGS[$((i + 1))]}"
            fi
            ;;
        --encoder=*)
            ENCODER="${arg#*=}"
            ;;
    esac
done
PRESET="$REPO_DIR/configs/online/sac_rgb.yaml"
if [[ "$ENCODER" == resnet* ]]; then
    PRESET="$REPO_DIR/configs/online/sac_rgb_resnet.yaml"
fi

exec env RLG_LAUNCHER="${BASH_SOURCE[0]}" RLG_LAUNCHER_PRESET="$PRESET" RLG_STD_LOG="$STD_LOG" RLG_LOG_TYPE="$LOG_TYPE" RLG_LOG_KEYWORDS="$LOG_KEYWORDS" "$PYTHON_BIN" -u "$REPO_DIR/examples/train_online.py" sac \
    --config "$PRESET" \
    "${FORWARD_ARGS[@]}"
