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

### Stride-two demonstration sampling

The ACT base policy used by the [privileged-state and visual residual routes](OPEN_LAPTOP_RECIPES.md)
was trained by retaining every second observation from the same 50 demonstrations,
plus the terminal observation. RGB, control state, end-effector poses and grippers
use identical indices. Actions are recomputed from the retained endpoint poses
using relative SE(3) transforms; existing actions are not simply subsampled and
rotation vectors are not added.

Convert the original demonstrations to a separate stride-two dataset:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python examples/convert_robotwin_act_delta_ee.py \
  --source-dir 3rd_party/RoboTwin/data/open_laptop/demo_clean \
  --output-path demos/robotwin/open_laptop_delta_ee_stride2.h5 \
  --task-name open_laptop --task-config demo_clean \
  --conversion-mode direct --num-episodes 50 \
  --camera-width 320 --camera-height 240 --frame-stride 2 \
  --ee-delta-pos-scale 0.06 \
  --ee-delta-rot-scale 0.30 \
  --gripper-delta-scale 0.2
```

The default stride remains 1. If the final interval is shorter than the stride,
it is retained. The HDF5 stores source frame indices, and the JSON records the
stride, action scales, clipping diagnostics and short terminal intervals.
Inspect clipping diagnostics before training. Stride does not automatically
change action scales; supply matching scales during conversion and execution.

The same procedure supports larger integer strides, such as `--frame-stride 3`
or `--frame-stride 4`: synchronize all retained observations and recompute actions
between their endpoint poses. Choose action scales from the resulting motion
range rather than automatically multiplying them by the stride.

Existing ACT experiments show that a suitable larger demonstration interval can
substantially reduce the number of decision steps needed to complete a task and
improve task success within an appropriate range. In the open_laptop comparison,
stride-two ACT achieved 54.4% success versus 47.2% for the historical stride-one
reference; on 190 scenes completed by both, mean decision steps fell from 208.03
to 112.91. These runs also differed in action scales and runtime setup, so the
success-rate difference cannot be attributed to stride alone. Further ACT
comparisons with absolute joint targets also showed improvements at intermediate
strides, but success did not increase monotonically with stride. Those targets
use future absolute joint positions, not the relative SE(3) labels in this converter.

Larger intervals also reduce feedback density and may skip important contact or
gripper events. Select the interval using closed-loop evaluation of success and
decision steps. Fewer decision steps do not imply the same reduction in physical
simulation time or wall-clock runtime.

The base policy uses three RGB views resized to 224×224, the original 14-dimensional
control state, a 30-action chunk, batch size 256, 100k updates, seed 1, ResNet18,
learning rates 1e-4 for the policy and 1e-5 for the backbone, EMA decay 0.999,
and KL weight 10. The converted dataset has 5,217 transitions instead of 10,412.
Keep the existing zero padding and unnormalized delta-action training convention.
Use the training command below with
`--demo-path demos/robotwin/open_laptop_delta_ee_stride2.h5`.
For standalone ACT evaluation, add `--robotwin.ee-delta-pos-scale 0.06`,
`--robotwin.ee-delta-rot-scale 0.30` and `--robotwin.gripper-delta-scale 0.2`.
The two residual recipes already configure these execution scales.

This is offline demonstration resampling, not online action skipping or a fixed
simulation control period. Raw physics timestamps were not retained, so stride 2
does not imply an exact time interval or halved runtime. Chunk length remains 30,
covering a longer span of the demonstration. Existing compatible ACT weights do
not require runtime changes; do not add extra action repetition or skipping.

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
