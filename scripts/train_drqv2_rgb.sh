#!/usr/bin/env bash
set -euo pipefail

# DrQ-v2 RGB training on PickCube-v1 with default hyperparameters
# matching the reference implementation in 3rd_party/drqv2.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env RLG_LAUNCHER="${BASH_SOURCE[0]}" RLG_LAUNCHER_PRESET="$REPO_DIR/configs/online/drqv2_rgb.yaml" python "$REPO_DIR/examples/train_online.py" drqv2 \
    --config "$REPO_DIR/configs/online/drqv2_rgb.yaml" \
    "$@"
