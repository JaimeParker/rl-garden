#!/usr/bin/env bash
# State-based SAC launcher for ManiSkill.
# Forward all args to the python script; sensible defaults are in Args.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hazyparker/Applications/anaconda3/envs/residual_rl/bin/python}"

exec "$PYTHON_BIN" "$REPO_DIR/examples/train_sac_state.py" \
    --env_id PickCube-v1 \
    --num_envs 16 \
    --total_timesteps 1000000 \
    "$@"
