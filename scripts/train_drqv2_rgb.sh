#!/usr/bin/env bash
set -euo pipefail

# DrQ-v2 RGB training on PickCube-v1 with default hyperparameters
# matching the reference implementation in 3rd_party/drqv2.

python examples/train_drqv2_rgb.py \
  --env_id PickCube-v1 \
  --total_timesteps 1000000 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --seed 1 \
  --feature_dim 50 \
  --hidden_dim 1024 \
  --nstep 3 \
  --gamma 0.99 \
  --tau 0.01 \
  --training_freq 32 \
  --utd 0.5 \
  --policy_lr 1e-4 \
  --q_lr 1e-4 \
  --stddev_schedule "linear(1.0,0.1,500000)" \
  --stddev_clip 0.3 \
  --num_expl_steps 2000 \
  --image_fusion_mode stack_channels \
  --image_augmentation random_shift \
  --image_random_shift_pad 4 \
  --learning_starts 4000 \
  --eval_freq 10000 \
  --num_eval_steps 50 \
  --camera_width 128 \
  --camera_height 128 \
  "$@"
