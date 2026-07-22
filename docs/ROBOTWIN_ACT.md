# RoboTwin ACT delta_ee Pipeline

This document covers ACT data conversion, training, and evaluation for
RoboTwin tasks in rl-garden. The target task name is `open_laptop`; use this
underscore form in commands even if the task is described as `open-laptop`.

## Sources

- Runtime environment: `3rd_party/RoboTwin` on the `RLinf_support` branch.
- Official ACT reference code: `3rd_party/robotwin-main/policy/ACT`.
- rl-garden control mode for RoboTwin end-effector deltas: `delta_ee`.

The official RoboTwin ACT code trains on absolute joint targets. rl-garden's
RoboTwin ACT path trains a 14D normalized `delta_ee` action:

```text
left dxyz, left drotvec, left gripper delta,
right dxyz, right drotvec, right gripper delta
```

The adapter converts this to RoboTwin's 16D delta EE command and calls
`take_action(..., action_type="delta_ee")`.

## Collect Official RoboTwin Data

Use RoboTwin's own collection script. The checked-in `demo_clean.yml` and
`demo_randomized.yml` already collect `rgb`, `qpos`, and `endpose`.

```bash
cd 3rd_party/RoboTwin
python script/collect_data.py open_laptop demo_clean
```

The expected output layout is:

```text
3rd_party/RoboTwin/data/open_laptop/demo_clean/
  seed.txt
  data/episode0.hdf5
  data/episode1.hdf5
  ...
```

## Convert To rl-garden ACT Data

Run conversion from the rl-garden repository root:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python examples/convert_robotwin_act_delta_ee.py \
  --source-dir 3rd_party/RoboTwin/data/open_laptop/demo_clean \
  --output-path demos/robotwin/open_laptop_delta_ee.h5 \
  --task-name open_laptop \
  --task-config demo_clean \
  --robotwin-root 3rd_party/RoboTwin \
  --conversion-mode auto \
  --camera-width 320 \
  --camera-height 240 \
  --num-episodes 50
```

`--conversion-mode auto` reads `/endpose` directly when present. If an older
official dataset lacks `/endpose`, it replays `/joint_action/vector` in
`3rd_party/RoboTwin` to reconstruct EE poses. Replay requires `seed.txt`,
`--robotwin-root`, and the matching `--task-config`.

The converter writes:

```text
demos/robotwin/open_laptop_delta_ee.h5
demos/robotwin/open_laptop_delta_ee.json
```

The HDF5 schema is ManiSkill/ACT-style:

```text
traj_i/actions                         T x 14 float32
traj_i/obs/extra/state                 T+1 x 14 float32
traj_i/obs/sensor_data/head_camera/rgb T+1 x H x W x 3 uint8
traj_i/obs/sensor_data/left_camera/rgb T+1 x H x W x 3 uint8
traj_i/obs/sensor_data/right_camera/rgb
traj_i/obs/sensor_param/
```

The JSON sidecar records `control_mode="delta_ee"`, camera order, scale
parameters, transition counts, and clipping diagnostics.

## Train ACT

```bash
PYTHONPATH=$PWD:$PYTHONPATH python examples/train_act_robotwin.py \
  --demo-path demos/robotwin/open_laptop_delta_ee.h5 \
  --env-id open_laptop \
  --control-mode delta_ee \
  --exp-name open_laptop_act \
  --total-iters 100000 \
  --batch-size 256 \
  --num-queries 30 \
  --image-width 224 \
  --image-height 224 \
  --camera-names head_camera left_camera right_camera \
  --save-freq 5000
```

The final checkpoint is:

```text
runs/open_laptop_act/checkpoints/final.pt
```

It contains `agent`, `ema_agent`, `norm_stats`, and config metadata. For
`delta_ee`, `norm_stats` is normally `None`, so the ACT provider emits actions
directly in normalized env action space.

## Evaluate ACT On RoboTwin

`examples/eval_act_robotwin.py` remains the ACT eval entrypoint. Use the
checkpoint from training:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python examples/eval_act_robotwin.py \
  --env-id open_laptop \
  --control-mode delta_ee \
  --base-ckpt-path runs/open_laptop_act/checkpoints/final.pt \
  --num-eval-episodes 10 \
  --num-eval-envs 1 \
  --camera-width 320 \
  --camera-height 240 \
  --base-act-image-width 224 \
  --base-act-image-height 224 \
  --robotwin.robotwin-root 3rd_party/RoboTwin \
  --robotwin.step-lim 700 \
  --robotwin.reward-mode sparse \
  --robotwin.include-wrist-cameras \
  --capture-video true
```

Video and diagnostic outputs default to a directory next to the checkpoint:

```text
runs/open_laptop_act/checkpoints/act_robotwin_eval_videos/
```

## Notes

- Use `delta_ee` for RoboTwin env commands. Other environments keep their own
  existing control-mode names.
- Camera order is head, left wrist, right wrist. The eval provider sees these
  as `rgb`, `rgb_left_wrist`, and `rgb_right_wrist`.
- `open_laptop` currently uses sparse reward in rl-garden. Pass
  `--robotwin.reward-mode sparse` for evaluation.
- If conversion reports high `action_clip_fraction`, increase the corresponding
  scale only if you also use the same scale in eval.
