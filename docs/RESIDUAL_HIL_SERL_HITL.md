# Residual HIL-SERL HITL Training

This document describes the `residual_hil_serl` human-in-the-loop training path.
It is a split actor/learner workflow for ResidualSAC with HIL-SERL-style
intervention replay mixing. It supports both `maniskill` and `franka_real`
environment backends.

## Overview

The entrypoint is:

```bash
python examples/train_hitl.py residual_hil_serl ...
```

This is intentionally separate from:

- `examples/train_online.py`: single-process rollout + update via `agent.learn()`.
- `examples/train_real_world.py`: existing real-robot-only SERL/HIL-SERL methods.

`residual_hil_serl` uses two long-running processes:

- **actor**: owns the live environment, runs inference, applies teleop
  intervention, and pushes transitions.
- **learner**: owns the full replay buffers, optimizers, logger, checkpoints,
  and policy updates.

The two processes communicate through the existing HTTP sync layer:

- actor sends transitions to learner with `POST /transition`;
- actor polls learner parameters with `GET /policy_params`;
- learner publishes the latest `policy.state_dict()` every `publish_freq`
  learner-loop iterations.

The processes can run on different machines. Bind the learner to a reachable
address and point the actor at that address:

```bash
# Learner machine
python examples/train_hitl.py residual_hil_serl \
  --role learner \
  --sync_host 0.0.0.0 \
  --sync_port 6000 \
  ...

# Actor machine
python examples/train_hitl.py residual_hil_serl \
  --role actor \
  --sync_host <learner-ip> \
  --sync_port 6000 \
  ...
```

The sync protocol has no authentication or encryption. Use it only on a trusted
LAN or through an SSH tunnel.

## Code Structure

Main files:

- `examples/train_hitl.py`: thin HITL registry dispatcher.
- `rl_garden/training/hitl/_registry.py`: HITL training registry.
- `rl_garden/training/hitl/_args.py`: shared actor/learner/sync/teleop/demo
  arguments.
- `rl_garden/training/hitl/residual_hil_serl.py`: env construction, agent
  construction, actor/learner/eval role dispatch, and the residual actor loop.
- `rl_garden/algorithms/residual_hil_serl.py`: `ResidualHilSerlSAC`, a
  `ResidualSAC` subclass that adds a growing intervention demo buffer.
- `rl_garden/envs/wrappers/teleop_intervention.py`: teleop override wrapper,
  extended with `record_gripper`.

Existing `ResidualSAC` is not modified. The HIL-specific demo-buffer behavior is
kept in `ResidualHilSerlSAC`.

## Transition Semantics

Residual replay does not store raw environment actions. It stores normalized
final actions in residual coordinates, plus the base-policy actions needed by
the residual actor and critic.

Every actor-side transition sent to the learner contains:

```text
obs
next_obs
action              # normalized final action used by ResidualSAC replay
reward
done
base_actions
next_base_actions
intervened
```

When there is no human intervention:

```text
base_action + residual_action -> normalized final action
normalized final action -> env action via ActionScaler.unscale()
```

The actor executes the env action and stores the normalized final action.

When there is human intervention:

```text
teleop action -> env.step(...)
teleop env action -> normalized final action via ActionScaler.scale()
```

The replay action is therefore still in ResidualSAC's normalized action
coordinate system, even though the environment executed the teleop action.

The learner uses HIL-SERL routing:

- every transition is added to the online replay buffer;
- `intervened=True` transitions are also copied into the demo buffer;
- training samples mix online and demo data according to `demo_data_ratio`.

## Teleop Action Shape

The HITL path assumes `control_mode=pd_ee_twist`.

Teleop sources produce a 7D action:

```text
[twist_x, twist_y, twist_z, twist_rx, twist_ry, twist_rz, gripper]
```

`TeleopInterventionWrapper(record_gripper=...)` controls how much of that action
is used during intervention:

- `record_gripper=True`: use all 7 dimensions.
- `record_gripper=False`: use only the first 6 twist dimensions.

The wrapper validates the processed human action against
`env.single_action_space.shape`. Use:

- `--teleop_record_gripper true` for `franka_real` and ManiSkill controllers
  with a gripper action dimension;
- `--teleop_record_gripper false` for ManiSkill fixed-gripper / no-gripper
  `pd_ee_twist` controllers.

On actor startup the teleoperation source is checked before training begins.
For Pico, the actor waits for the first ZMQ input sample for
`--teleop_init_timeout_s` seconds (default: `120.0`) and raises if none arrives.
For SpaceMouse, initialization raises if the HID device cannot be opened. Set
`--teleop_init_timeout_s 0` to skip the Pico startup wait.

## ManiSkill Usage

ManiSkill uses its own environment reward and termination. The HITL path does
not apply the real-world reward classifier wrapper for `env_backend=maniskill`.

Example learner:

```bash
python examples/train_hitl.py residual_hil_serl \
  --role learner \
  --env_backend maniskill \
  --env_id PegInsertionSidePegOnly-v1 \
  --control_mode pd_ee_twist \
  --base_policy zero \
  --sync_host 0.0.0.0 \
  --sync_port 6000 \
  --log_type none
```

Example actor on the same machine:

```bash
python examples/train_hitl.py residual_hil_serl \
  --role actor \
  --env_backend maniskill \
  --env_id PegInsertionSidePegOnly-v1 \
  --control_mode pd_ee_twist \
  --base_policy zero \
  --sync_host 127.0.0.1 \
  --sync_port 6000 \
  --teleop_record_gripper false \
  --log_type none
```

Example actor with SpaceMouse:

```bash
python examples/train_hitl.py residual_hil_serl \
  --role actor \
  --env_backend maniskill \
  --env_id PegInsertionSidePegOnly-v1 \
  --control_mode pd_ee_twist \
  --render_mode human \
  --base_policy zero \
  --sync_host 127.0.0.1 \
  --sync_port 6000 \
  --teleop_device spacemouse \
  --teleop_record_gripper false \
  --log_type none
```

Set `--teleop_record_gripper true` if the selected ManiSkill robot/controller
exposes a 7D action space with gripper control. SpaceMouse does not use
`--teleop_init_timeout_s`; if the HID device cannot be opened, actor startup
fails immediately with a SpaceMouse connection error.

## Real-World Usage

For `env_backend=franka_real`, the path can use the same real-robot wrapper
concepts as the existing HIL-SERL implementation:

- optional quaternion-to-rotvec observation conversion via `--convert_obs_rotation`;
- optional reward classifier via `--classifier_checkpoint`;
- teleop intervention through Pico or SpaceMouse.

Example learner:

```bash
python examples/train_hitl.py residual_hil_serl \
  --role learner \
  --env_backend franka_real \
  --control_mode pd_ee_twist \
  --sync_host 0.0.0.0 \
  --sync_port 6000 \
  --franka_real.bridge_url http://robot-pc:5000
```

Example actor:

```bash
python examples/train_hitl.py residual_hil_serl \
  --role actor \
  --env_backend franka_real \
  --control_mode pd_ee_twist \
  --sync_host <learner-ip> \
  --sync_port 6000 \
  --franka_real.bridge_url http://localhost:5000 \
  --teleop_record_gripper true
```

Use SpaceMouse on the real-robot actor by adding:

```bash
--teleop_device spacemouse
```

Actor and learner must use matching environment/action-space configuration and
the same base-policy checkpoint path.

## Demo Buffer and Recovery

`ResidualHilSerlSAC` uses the same offline replay slot that ResidualSAC already
uses for offline data. Because of that, `--offline_dataset_path` and
`init_demo_buffer()` cannot both populate the same agent. The learner raises
instead of silently replacing an offline dataset.

The learner snapshots received transitions for crash recovery:

```text
<checkpoint_dir>/buffer/transitions_<n>.pkl
<checkpoint_dir>/demo_buffer/transitions_<n>.pkl
```

The `buffer/` directory contains all received transitions. The `demo_buffer/`
directory contains only intervened transitions. Both are reloaded on learner
startup. Pre-collected demo snapshots can also be supplied with
`--demo_dataset_paths`.

## Current Limits

- HITL actor supports exactly one environment: `num_envs=1`.
- Only `control_mode=pd_ee_twist` is supported.
- Only `env_backend=maniskill` and `env_backend=franka_real` are accepted.
- FWBW reset-free training is not part of this first residual HITL path.
- The sync layer is not secured; use trusted networking.
- The local test environment used during implementation did not have
  `torch`/`gymnasium`, so pytest could not be run there. Syntax compilation and
  `git diff --check` passed.
