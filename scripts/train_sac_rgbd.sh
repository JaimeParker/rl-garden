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
FORWARD_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --std_log|--std-log)
            STD_LOG=1
            ;;
        --no_std_log|--no-std-log)
            STD_LOG=0
            ;;
        *)
            FORWARD_ARGS+=("$arg")
            ;;
    esac
done

exec env RLG_STD_LOG="$STD_LOG" "$PYTHON_BIN" -u "$REPO_DIR/examples/train_sac_rgbd.py" \
    --env_id PickCube-v1 \
    --obs_mode rgb \
    --num_envs 16 \
    --camera_width 64 --camera_height 64 \
    --total_timesteps 1000000 \
    "${FORWARD_ARGS[@]}"
