#!/usr/bin/env bash
# Critic-only ResidualSAC ablation: ACT rollout, zero residual, and zero entropy.
# ACT uses 640x480 camera images; the environment exposes 128x128 agent images.
# Usage:  bash scripts/train_residual_sac_robotwin.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python interpreter not found in PATH (tried: python, python3)." >&2
    exit 1
fi

ASSETS_PATH_ARG="${RLG_ROBOTWIN_ASSETS_PATH:-/home/RoboTwin}"
CUDA_VISIBLE_DEVICES_ARG="${RLG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-1}}"
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
    RLG_STD_LOG="$STD_LOG" \
    RLG_LOG_TYPE="$LOG_TYPE" \
    RLG_LOG_KEYWORDS="$LOG_KEYWORDS" \
    ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}" \
    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u "$REPO_DIR/examples/train_online.py" residual_sac \
    --env-backend robotwin \
    --env-id place_empty_cup \
    --obs-mode rgb \
    --per-camera-rgbd \
    --control-mode joint_pos \
    --base-policy act \
    --base-ckpt-path pretrained_models/place_empty_cup.ckpt \
    --base-act-stats-path pretrained_models/dataset_stats_place_empty_cup.pkl \
    --num-envs 1 \
    --num-eval-envs 0 \
    --critic-only-steps 1000000 \
    --no-critic-only-freeze-encoder \
    --ent-coef 0 \
    --residual-action-coordinates raw_joint_delta \
    --residual-action-scale 0 \
    --robotwin.robotwin-root /home/RoboTwin \
    --robotwin.assets-path "$ASSETS_PATH_ARG" \
    --robotwin.head-camera-type D435 \
    --robotwin.wrist-camera-type D435 \
    --robotwin.no-random-background \
    --robotwin.no-cluttered-table \
    --robotwin.clean-background-rate 1.0 \
    --robotwin.random-head-camera-dis 0 \
    --robotwin.random-table-height 0 \
    --robotwin.step-lim 200 \
    --camera-width 640 \
    --camera-height 480 \
    --total-timesteps 100000 \
    --learning-starts 32 \
    --training-freq 1 \
    --utd 1.0 \
    --buffer-size 1000 \
    --buffer-device cpu \
    --robotwin.agent-image-size 128 \
    --batch-size 32 \
    --gamma 0.99 \
    --bootstrap-at-done truncated \
    --q-lr 0.0001 \
    --encoder resnet18 \
    --encoder-features-dim 64 \
    --image-fusion-mode per_key \
    --eval-freq 0 \
    --num-eval-steps 200 \
    --eval-output-dir "$REPO_DIR/runs/residual_sac_robotwin_eval_videos" \
    --video-fps 30 \
    --log-freq 16 \
    --checkpoint-freq 10000 \
    --capture-video \
    --log-type "$LOG_TYPE" \
    --wandb-project robotwin-place_empty_cup \
    --wandb-group critic_only_zero_residual_ent0 \
    --exp-name raw_joint_delta_critic_only_zero_residual_ent0 \
    "${FORWARD_ARGS[@]}"
