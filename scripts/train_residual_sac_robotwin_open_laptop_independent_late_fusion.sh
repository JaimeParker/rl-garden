#!/usr/bin/env bash
set -euo pipefail

# Required local inputs. No machine-specific path or credential is committed.
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the RoboTwin checkout}"
: "${ROBOTWIN_ASSETS_PATH:?Set ROBOTWIN_ASSETS_PATH to the RoboTwin assets root}"
: "${ACT_CHECKPOINT:?Set ACT_CHECKPOINT to the RGB ACT checkpoint}"
: "${RESIDUAL_WARMUP_CHECKPOINT:?Set RESIDUAL_WARMUP_CHECKPOINT to the state residual checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_NAME="${EXP_NAME:-open_laptop-residual-independent-late-fusion-train200k}"
LOG_DIR="${LOG_DIR:-runs}"

args=(
  --env-backend robotwin
  --env-id open_laptop
  --obs-mode rgb
  --include-state
  --image-keys rgb,rgb_left_wrist,rgb_right_wrist
  --control-mode delta_ee
  --base-policy act
  --base-ckpt-path "$ACT_CHECKPOINT"
  --base-act-observation-width 320
  --base-act-observation-height 240
  --base-act-temporal-agg-k 0.01
  --num-envs 1
  --num-eval-envs 0
  --camera-width 64
  --camera-height 64
  --total-timesteps 200000
  --buffer-size 100000
  --buffer-device cuda
  --batch-size 64
  --learning-starts 5000
  --training-freq 64
  --utd 0.25
  --gamma 0.99
  --residual-action-scale 0.1
  --residual-gripper-action-scale 0.20
  --residual-warmup-scale 0
  --residual-warmup-policy-checkpoint "$RESIDUAL_WARMUP_CHECKPOINT"
  --residual-warmup-policy-probability 0.5
  --residual-log-std-init -3
  --encoder drqv2_independent_late_fusion
  --encoder-features-dim 256
  --image-fusion-mode stack_channels
  --image-augmentation random_shift
  --actor-log-std-mode clamp
  --actor-log-std-min -5
  --actor-use-layer-norm
  --critic-use-layer-norm
  --critic-only-steps 7000
  --no-critic-only-freeze-encoder
  --policy-lr 0.00003
  --q-lr 0.0003
  --alpha-tuning log_alpha
  --ent-coef 0.01
  --robotwin.robotwin-root "$ROBOTWIN_ROOT"
  --robotwin.assets-path "$ROBOTWIN_ASSETS_PATH"
  --robotwin.embodiment aloha-agilex
  --robotwin.reward-mode dense
  --robotwin.reward-shaping-mode hybrid
  --robotwin.dense-success-reward 10
  --robotwin.potential-discount 0.99
  --robotwin.potential-weight 5
  --robotwin.dense-weight 0.03
  --robotwin.relative-weight 3
  --robotwin.step-penalty 0.003
  --robotwin.stall-threshold 0.0001
  --robotwin.stall-penalty 0.035
  --robotwin.backtrack-penalty 0.06
  --robotwin.gripper-delta-scale 0.20
  --robotwin.step-lim 500
  --robotwin.clear-cache-freq 1
  --robotwin.no-random-background
  --robotwin.no-cluttered-table
  --robotwin.random-head-camera-dis 0
  --robotwin.random-table-height 0
  --robotwin.crazy-random-light-rate 0
  --robotwin.delta-ee-command-reference
  --robotwin.delta-ee-command-reanchor
  --robotwin.delta-ee-planner-type mplib_screw
  --robotwin.delta-ee-command-reanchor-position-tolerance 0.005
  --robotwin.delta-ee-command-reanchor-rotation-tolerance 0.03490658503988659
  --robotwin.delta-ee-terminal-settle-tolerance 0.0005
  --robotwin.delta-ee-terminal-settle-max-ticks 100
  --robotwin.device cuda
  --no-capture-video
  --eval-freq 0
  --log-freq 256
  --checkpoint-freq 25000
  --save-replay-buffer
  --seed 0
  --log-dir "$LOG_DIR"
  --exp-name "$EXP_NAME"
)

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  args+=(--log-type wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    args+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  args+=(--log-type none)
fi

exec "$PYTHON_BIN" -u examples/train_online.py residual_sac "${args[@]}"
