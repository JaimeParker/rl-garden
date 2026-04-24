#!/usr/bin/env bash
# RGBD SAC launcher with selectable image encoder.
# Examples:
#   scripts/train_sac_rgbd.sh                                 # plain_conv
#   scripts/train_sac_rgbd.sh --encoder resnet10
#   scripts/train_sac_rgbd.sh --encoder resnet10 --pretrained_weights resnet10-imagenet
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hazyparker/Applications/anaconda3/envs/residual_rl/bin/python}"

exec "$PYTHON_BIN" "$REPO_DIR/examples/train_sac_rgbd.py" \
    --env_id PickCube-v1 \
    --obs_mode rgb \
    --num_envs 16 \
    --camera_width 64 --camera_height 64 \
    --total_timesteps 1000000 \
    "$@"
