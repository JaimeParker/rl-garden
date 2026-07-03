#!/usr/bin/env bash
# ResidualSAC on RoboTwin via the unified train_online.py entrypoint.
# Usage:  bash scripts/train_residual_sac_robotwin.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

STD_LOG="${RLG_STD_LOG:-1}"
LOG_TYPE="${RLG_LOG_TYPE:-wandb}"
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
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

exec env RLG_STD_LOG="$STD_LOG" RLG_LOG_TYPE="$LOG_TYPE" \
    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u "$REPO_DIR/examples/train_online.py" residual_sac \
    --env-backend robotwin \
    --env-id place_empty_cup \
    --obs-mode rgb \
    --base-policy act \
    --base-ckpt-path pretrained_models/place_empty_cup.pt \
    --num-envs 1 \
    --num-eval-envs 0 \
    --robotwin.robotwin-root /home/RoboTwin \
    --control-mode ee_delta_pose \
    # TODO: downsample the camera images to reduce memory usage and speed up training
    --camera-width 320 --camera-height 240 \
    --total-timesteps 100000 \
    --buffer-size 10000 \
    --buffer-device cpu \
    --batch-size 256 \
    --learning-starts 512 \
    --training-freq 256 \
    --utd 0.25 \
    --robotwin.step-lim 200 \
    --robotwin.control-step-cap 16 \
    --robotwin.disable-topp \
    --encoder resnet18 \
    --encoder-features-dim 64 \
    --image-fusion-mode per_key \
    --robotwin.no-include-wrist-cameras \
    --eval-freq 0 \
    --log-freq 16 \
    --exp-name residual_sac_robotwin_place_empty_cup \
    --wandb_project robotwin \
    "${FORWARD_ARGS[@]}"
