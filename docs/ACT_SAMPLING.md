# ACT demonstration sampling intervals

Use `--frame-stride N` to retain every Nth demonstration observation, including
the terminal frame. RGB, control state, poses and grippers use the same indices.
Translation, rotation and gripper changes are recomputed between retained frames,
rather than simply subsampling existing actions. The default is 1.

Suitable larger intervals can reduce completion decision steps and improve ACT
success in our experiments, but larger is not always better. Choose the interval
using closed-loop evaluation; decision-step reductions are not equivalent to
reductions in physical execution time.

## Convert and train

The [open_laptop residual routes](OPEN_LAPTOP_RECIPES.md) use stride-two ACT data:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python examples/convert_robotwin_act_delta_ee.py \
  --source-dir 3rd_party/RoboTwin/data/open_laptop/demo_clean \
  --output-path demos/robotwin/open_laptop_delta_ee_stride2.h5 \
  --task-name open_laptop --task-config demo_clean \
  --conversion-mode direct --num-episodes 50 \
  --camera-width 320 --camera-height 240 --frame-stride 2 \
  --ee-delta-pos-scale 0.06 --ee-delta-rot-scale 0.30 \
  --gripper-delta-scale 0.2
```

Follow the [ACT training guide](ROBOTWIN_ACT.md#train-act), using the generated
file as `--demo-path`. Keep the 30-action chunk and existing delta-action padding.
The output records source frame indices, sampling settings and clipping statistics.

## Larger intervals and execution

- Set `--frame-stride 3`, `4` or another positive integer to use a larger interval.
  The final shorter interval is retained.
- Choose action scales for the resulting motion range and inspect clipping;
  scales do not change automatically with stride.
- Use the same scales during evaluation. For the stride-two configuration above,
  pass `--robotwin.ee-delta-pos-scale 0.06`, `--robotwin.ee-delta-rot-scale 0.30`
  and `--robotwin.gripper-delta-scale 0.2` to standalone ACT evaluation.
  The residual recipes already set these values.
- This is offline resampling, not online action skipping or repetition. Larger
  intervals make feedback coarser and may skip contact or gripper events.
