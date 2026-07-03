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
    --camera-width 224 --camera-height 224 \
    --capture-video \
    --total-timesteps 100000 \
    --buffer-size 512 \
    --buffer-device cpu \
    --batch-size 16 \
    --learning-starts 0 \
    --training-freq 16 \
    --utd 0.5 \
    # --step_lim 200 \
    --robotwin.step-lim 200 \
    --robotwin.control-step-cap 16 \
    --robotwin.disable-topp \
    --encoder resnet10 \
    --encoder-features-dim 64 \
    --image-fusion-mode per_key \
    --robotwin.no-include-wrist-cameras \
    --eval-freq 0 \
    --log-freq 16 \
    --exp-name residual_sac_robotwin_place_empty_cup \
    --wandb-project robotwin \
    "${FORWARD_ARGS[@]}"
