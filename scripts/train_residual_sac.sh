#!/usr/bin/env bash
# Compatibility alias for the generic Residual SAC RGBD launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train_residual_sac_rgbd.sh" "$@"
