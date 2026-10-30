# open_laptop: privileged-state and visual residual routes

Both routes train residual SAC on a frozen ACT base policy. The privileged-state route reads simulated object state. The visual route uses a frozen, supervised-pretrained encoder to combine estimated object geometry with measured robot feedback.

## 1. Choose a route

| Setting | Privileged-state residual | Visual residual |
| --- | --- | --- |
| Actor and critic observation features | 146 privileged-state dimensions | 163 visual and proprioceptive dimensions |
| Object information | Simulated poses, hinge, geometry and contacts | Three RGBD views, calibration and four-frame causal history |
| Robot information | Measured joints and control references | Measured q, dq, gripper width, FK, Jacobians and control references |
| Frozen visual model | Not required | Base visual encoder, 128×128 or 320×240 input |
| ACT input | Three 320×240 RGB views and the original 14-dimensional control state | Same |
| Warmup precollection policy | Frozen privileged-state policy | Same; privileged inputs are used only during warmup |
| Replay | States, actions, rewards, termination masks and base actions | Estimated states and the same transition fields, not full images |
| Training budget | 220k environment steps | 220k environment steps |

The actor additionally receives the 14-dimensional ACT base action; the critic receives the final action. Both encoder resolutions have produced good geometric estimates in existing experiments. Capture preprocessing adjusts RGBD and camera intrinsics together. See the [encoder guide](ENCODER_INTERFACE.md) and [observation interface](OBSERVATION_INTERFACE.md).

## 2. Environment and model preparation

See [RoboTwin integration](ROBOTWIN.md) for installation and assets. These routes use the optimized delta-EE executor from the [RoboTwin fork](https://github.com/Nole326/RoboTwin), described in its [executor documentation](https://github.com/Nole326/RoboTwin/blob/77ed01b786767dc1f912ff8e1cf0148ed7a7aae2/docs/DELTA_EE_EXECUTOR.md). Training and evaluation use the same executor.

Prepare an ACT base policy, a warmup precollection policy and, for the visual route, pretrained encoder weights. This repository provides code and model descriptions, not encoder weights or datasets.

Model loading constructs a network from code and reads its learned weights. Training code is not a weight file. Full continuation additionally requires optimizer, replay, counters and random states.

Loading checks interface compatibility rather than private filenames or training steps. ACT must match the 14-dimensional delta-EE action interface; the warmup policy must match the 146-dimensional privileged observation and action units. The default encoder must match the documented geometry architecture. Renaming an incompatible file does not make it compatible.

The robot is Aloha. `assets_path` points to a root containing `assets/embodiments/aloha-agilex`. Gripper-width conversion uses the [calibration file](../configs/open_laptop/aloha_gripper_calibration.json).

The ACT base policy for these configurations uses stride-two demonstration sampling,
with actions recomputed between retained poses. See the [ACT data and training guide](ROBOTWIN_ACT.md#stride-two-demonstration-sampling).
The position, rotation and gripper normalization units are 0.06m, 0.30rad and 0.2.
This sampling choice requires no additional online action skipping.

## 3. Construction and switching

[open_laptop_recipes.py](../rl_garden/training/online/open_laptop_recipes.py) provides `privileged-state` and `visual-state`.

```python
from rl_garden.training.online.open_laptop_recipes import recipe_args, build_agent

recipe = 'visual-state'  # Use 'privileged-state' for simulated object state.
args = recipe_args(
    recipe,
    act_checkpoint='models/act.pt',
    teacher_checkpoint='models/warmup-policy.pt',
    robotwin_root='/path/to/RoboTwin',
    assets_path='/path/to/assets-root',
    log_dir='runs/open_laptop',
)
# Create env and logger first, but do not reset env yet.
visual = {} if recipe == 'privileged-state' else dict(
    encoder_checkpoint='models/encoder.pt',
    gripper_calibration='configs/open_laptop/aloha_gripper_calibration.json',
    encoder_image_size=(128, 128),  # Height, width; alternatively (240, 320).
)
agent = build_agent(recipe, args, env, logger, None, **visual)
```

Switching routes requires a matching environment and a new residual learner. ACT and the warmup policy can be reused. Keep the visual encoder frozen. Replacing it requires retraining and evaluating the residual policy for the new estimated-state distribution.

The standalone [entrypoint](../rl_garden/training/online/laptop_entry.py) exposes training and evaluation:

```bash
python -m rl_garden.training.online.laptop_entry --help
```

`train` runs training, `evaluate` uses a supplied fixed scene list, and `smoke` runs a short integration check. See [checkpoints](CHECKPOINT.md) for the general storage format.

Choose `--encoder-resolution 128x128` or `--encoder-resolution 320x240`. Training and evaluation must use matching preprocessing. Changed weights or input settings cannot reuse the original complete continuation directory.

## 4. Training configuration

The warmup policy collects the first 10k environment steps. Only the critic updates from 10k to 20k; actor, critic and temperature updates follow. Training finishes the current episode after reaching 220k steps.

| Parameter | Value |
| --- | --- |
| Environments; batch size | 1; 64 |
| Replay capacity | 100000 |
| Update schedule | Every 64 environment steps; UTD 0.25 |
| Actor; critic learning rate | 3e-5; 3e-4 |
| Discount; target update coefficient | 0.99; 0.01 |
| Networks | Three 256-unit layers for actor and critic; LayerNorm; two critics |
| Bootstrap | truncated |
| Initial temperature; residual log std | 0.01; −3, with a lower bound of −5 |
| ACT temporal aggregation | Enabled, k=0.01 |
| Training episode limit | 200 decision steps |
| Scenes | clean; head-camera displacement, table-height and lighting randomization disabled |

Position, rotation and gripper residual scales are 0.1, 0.2 and 0.2. The warmup policy's rotation residual scale is 0.1; multiplying its normalized rotation output by 0.5 preserves its physical action. Reward uses the hybrid configuration; see [defaults](../rl_garden/training/online/open_laptop_defaults.json).

This branch defaults to and recommends `truncated`. Selected comparisons showed more stable late Q estimates and critic loss than `always`; see [bootstrap behavior](SAC_BOOTSTRAP.md).

## 5. Evaluation

Use five fixed groups of 100 scenes, each limited to 700 decision steps. Report each group's success rate and the 500-episode mean. A scene file is a JSON list of environment seeds, such as `[10001, 10002]`.

Visual evaluation reuses buffers within the same exposure and uses `clear_cache_freq=5`. Results are saved per completed episode; repeating an identical command resumes unfinished episodes. Keep weights, observation processing, residual scales and executor settings consistent with training.
