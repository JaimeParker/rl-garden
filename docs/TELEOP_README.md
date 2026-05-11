# Teleoperation

This document describes the current end-effector twist teleoperation interface
under `robot_infra/teleop`.

## EETwist TeleOp Interface

The main interface is `EETwistTeleOpWrapper` in
`robot_infra/teleop/utils/telo_op_control_twist.py`.

Currently supported device:

- `pico`

Reserved but not implemented:

- `spacemouse`

The wrapper receives device data from a ZeroMQ `SUB` socket. The Pico server is
expected to publish JSON arrays to:

```python
tcp://*:7777
```

The client connects with `--zmq-url`, for example:

```bash
--zmq-url tcp://192.168.6.2:7777
```

### Pico Data Format

Pico data can be either one hand (`13` values) or two hands (`26` values). For
two hands, the first `13` values are the left hand, and the second `13` values
are the right hand.

For each hand:

| Index | Meaning |
| --- | --- |
| `0:3` | hand position |
| `6:10` | hand quaternion, `[x, y, z, w]` |
| `10` | gripper button |
| `11` | bind / clutch button |
| `12` | episode end button |

`bind` is held while choosing the current reference pose. During bind, twist is
zero. After release, hand pose changes are converted to EE twist.

`poll()` returns a `TeleOpSample`:

```python
action       # twist + gripper
twist        # shape (6,)
gripper      # scalar gripper command
bind_pressed
episode_end
intervened
```

When no new device data arrives, `twist=0`, `gripper` keeps the previous value,
and `intervened=False`.

### Adding New Devices

To add a new teleoperation device:

1. Add the device name to `EETwistTeleOpWrapper.__init__`.
2. Implement a parser like `_parse_pico()`.
3. Dispatch it in `_parse_device()`.
4. Convert the device-specific data into `DeviceSample`.

The rest of the twist computation should stay shared.

## Basic Usage

Visual teleoperation:

```bash
python robot_infra/teleop/examples/test_teleop.py \
  --env-id PegInsertionSidePegOnly-v1 \
  --device pico \
  --zmq-url tcp://192.168.6.2:7777
```

Print incoming teleop state without connecting to an environment:

```bash
python robot_infra/teleop/examples/test_twist_input.py \
  --device pico \
  --zmq-url tcp://192.168.6.2:7777
```

`test_teleop.py` creates the ManiSkill environment, polls teleop actions, applies
controller action normalization when needed, and steps the environment.

## Recording WSRL Data

`robot_infra/teleop/examples/record_teleop_wsrl.py` records teleoperation
episodes into the same H5 format used by the WSRL data generator.

Example:

```bash
python robot_infra/teleop/examples/record_teleop_wsrl.py \
  --output-path demos/peg_teleop_wsrl.h5 \
  --env-id PegInsertionSidePegOnly-v1 \
  --obs-mode rgb \
  --control-mode pd_ee_twist \
  --device pico \
  --zmq-url tcp://192.168.6.2:7777
```

`--output-path` must be an H5 file path, not a directory. A `.h5` suffix is
recommended. The file is opened with write mode, so an existing file at the same
path will be overwritten.

Only `intervened=True` samples are stepped in the environment and saved. When
`intervened=False`, the script keeps polling the device and does not record a
transition.

The Pico episode-end button ends the current episode. After each episode, the
terminal prints the episode length, `max_episode_steps`, success flag, and
return, then asks whether to save the episode.

### Important Arguments

- `output_path`: H5 dataset file to write.
- `dt`: teleop polling period in seconds. The default is `1 / 30`, roughly
  `30 Hz`. Smaller values poll and step faster.
- `env_id`: ManiSkill environment id.
- `obs_mode`: observation mode, usually `state`, `rgb`, or `rgbd`.
- `include_state`: include proprioceptive state for visual observations.
- `control_mode`: should match the EE twist controller, usually `pd_ee_twist`.
- `sim_backend`, `render_backend`: ManiSkill simulation and rendering backends.
- `reward_mode`: reward mode passed to the environment.
- `robot_uids`: robot id for the environment.
- `fix_box`, `fix_peg_pose`, `peg_density`, `debug_pose_vis`: custom peg env
  options.
- `env_kwargs_json`: escape hatch for extra environment kwargs.
- `end_on_env_done`: if true, env termination, truncation, or success can end
  the episode. The default is false, so only teleop episode-end stops recording.
- `pos_scale`, `rot_scale`, `twist_limit`: optional overrides for twist mapping.
- `intervention_threshold`: twist norm threshold used to decide whether human
  input is active.

The saved H5 contains `traj_*` groups with:

```text
obs
actions
rewards
terminated
truncated
```

This file can be passed to WSRL, CQL, or Cal-QL offline training through
`--offline_dataset_path`.

For real-robot data where you only want to do offline pretraining (no sim
env, no eval), use `examples/pretrain_offline.py --algorithm cql` or
`--algorithm calql` for standalone offline CQL/Cal-QL checkpoints. Use
`--algorithm wsrl-calql` when the checkpoint should be resumed by the WSRL
offline→online flow. This entrypoint infers obs/action specs from the H5 and
produces pretrained checkpoints that can later be loaded into compatible
live-env training runs for online fine-tuning. See
[`WSRL_README.md`](WSRL_README.md#offline-only-pretraining-no-sim-env) for
the full workflow.
