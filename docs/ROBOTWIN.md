# RoboTwin Integration

This guide records the design choices behind RoboTwin support in rl-garden,
how the current implementation is structured, and how to start PPO training on
RoboTwin tasks.

The short version:

```python
from rl_garden.envs import RoboTwinEnvConfig, make_robotwin_env

env = make_robotwin_env(
    RoboTwinEnvConfig(
        task_name="place_shoe",
        num_envs=4,
        robotwin_root="/path/to/RoboTwin",
        device="cuda",
    )
)
```

`RoboTwinEnv` is the only environment object the algorithms see. Everything
else, including threaded execution, task lifecycle management, action
conversion, and reward injection, is internal to the RoboTwin env package.

## Design Decisions

RoboTwin support was designed to fit rl-garden, not to copy RLinf directly.
RLinf was the main reference because its `RoboTwinEnv`, `VectorEnv`, and
RoboTwin `RLinf_support` branch show what is needed to train RoboTwin with RL:

- a Gym-style wrapper around RoboTwin tasks;
- a threaded vector environment around many task instances;
- dense rewards configured per task;
- seed/reset management and auto-reset behavior compatible with RL loops.

The resulting rl-garden design keeps the same ideas but changes the public
boundary:

- **One public env:** algorithms receive `RoboTwinEnv`, built by
  `make_robotwin_env()`, matching the existing ManiSkill-style vector env
  contract.
- **Internal vector execution:** `ThreadedRoboTwinExecutor` and `SubEnv` are
  implementation details. They run RoboTwin task instances in a thread pool but
  are not exposed as algorithm APIs.
- **External RoboTwin runtime:** rl-garden does not vendor RoboTwin assets,
  cameras, robots, or full task files. RoboTwin must be importable, or the
  caller must pass `robotwin_root`.
- **Dense reward by default:** v1 defaults to task-specific dense rewards
  adapted from RoboTwin `RLinf_support`. Sparse success reward remains
  available through `reward_mode="sparse"`.
- **Python reward factories:** task reward definitions live in Python, not
  YAML, because they reference live task objects such as `task.shoe`,
  `task.hammer`, `task.plate`, generated poses, and contact helpers.
- **Tensor-only observations:** language instructions are kept in `infos`,
  not in `obs`. rl-garden's PPO/SAC policies and buffers expect tensor-like
  `Box` spaces.
- **PPO-first validation:** the env contract is generic enough for SAC, but the
  current entrypoint and test path focus on PPO.

The normal training path should return CUDA torch tensors from the public env.
RoboTwin itself still produces numpy/Python objects internally; the conversion
boundary is `RoboTwinEnv`.

## Implementation Shape

The RoboTwin package lives under `rl_garden/envs/robotwin/`.

The public entry points are:

- `RoboTwinEnvConfig`
- `RoboTwinEnv`
- `make_robotwin_env()`

`RoboTwinEnvConfig` contains runtime paths, reset settings, observation/action
settings, and reward settings. The most important fields are:

```python
RoboTwinEnvConfig(
    task_name="place_shoe",
    num_envs=4,
    robotwin_root="/path/to/RoboTwin",
    seeds_path="/path/to/train_seeds.json",
    step_lim=400,
    max_episode_steps=400,
    control_mode="delta_joint_pos",
    reward_mode="dense",
    device="cuda",
)
```

`RoboTwinEnv` owns the algorithm-facing contract:

- `num_envs`
- `single_observation_space`
- `single_action_space`
- `action_space`
- `reset()`
- `step()`
- `chunk_step()`
- `close()`

The observation keys are:

```text
rgb              head camera, uint8, B x H x W x 3
rgb_left_wrist   left wrist camera, uint8, B x H x W x 3
rgb_right_wrist  right wrist camera, uint8, B x H x W x 3
state            14D qpos/proprio vector, float32
```

If a wrist image is absent, the env fills that image key with zeros so the
observation space remains stable. Task instruction text is returned in
`infos`, not in the observation dict.

The internal execution stack is:

```text
PPO / SAC
  -> RoboTwinEnv
    -> ThreadedRoboTwinExecutor
      -> SubEnv
        -> RoboTwinTaskAdapter
          -> RoboTwin envs.<task_name> task instance
```

`ThreadedRoboTwinExecutor` owns the thread pool. `SubEnv` wraps one
`RoboTwinTaskAdapter` and uses a local lock. Reset also uses a global lock,
because SAPIEN scene creation/reset is not safe to run freely in many threads.

`RoboTwinTaskAdapter` bridges one RoboTwin task into rl-garden semantics:

- imports `envs.<task_name>` from RoboTwin;
- prepares RoboTwin task args, including camera, data type, domain
  randomization, embodiment files, and robot configs;
- calls `setup_demo()` on reset;
- injects dense reward with `build_task_reward()` when `reward_mode="dense"`;
- converts rl-garden actions to RoboTwin qpos+gripper commands;
- tracks reward state needed by dense reward primitives;
- calls `take_action(action, action_type="qpos")`;
- returns `StepResult` with obs, reward, termination, truncation, and info.

## Actions

The default control mode is:

```text
control_mode="delta_joint_pos"
```

The public action space is:

```text
Box(-1, 1, shape=(14,), dtype=float32)
```

This is interpreted as:

```text
left arm joint deltas:   6 dims
left gripper delta:      1 dim
right arm joint deltas:  6 dims
right gripper delta:     1 dim
```

The adapter reads the current RoboTwin joint state, applies
`joint_delta_scale` to arm dimensions and `gripper_delta_scale` to gripper
dimensions, clamps grippers to `[0, 1]`, then passes the resulting absolute
qpos+gripper target into RoboTwin.

`control_mode="joint_pos"` is also present for direct qpos-style actions, but
the default and intended v1 training mode is `delta_joint_pos`.

## Rewards

Dense reward primitives are adapted from RoboTwin `RLinf_support`:

- `Reward`
- `SerialTask`
- `ParallelTask`
- `Pick`
- `Contact`
- `Place`
- `Endpose`
- `Rank`
- `Stack`
- `SparseExtra`
- `Success`

Task-specific reward factories live in
`rl_garden/envs/robotwin/rewards/registry.py`. The current registry covers the
10 RoboTwin env configs that exist in RLinf:

```text
adjust_bottle
beat_block_hammer
click_bell
handover_block
lift_pot
move_can_pot
pick_dual_bottles
place_container_plate
place_empty_cup
place_shoe
```

If a task is not in the registry and `reward_mode="dense"`, reset will fail
with a clear missing-factory error. Use `reward_mode="sparse"` to train with
`check_success()` only, or add a factory for the task.

## Reset, Seeds, And Auto-Reset

`RoboTwinEnv` supports two seed paths:

- random reset ids generated from `seed`;
- success seeds loaded from `seeds_path`, using the same task-keyed JSON idea
  as RLinf.

With auto-reset enabled, done environments are reset inside `step()`. The env
stores terminal data in the Gymnasium-compatible keys that rl-garden already
expects:

```text
final_observation
final_info
_final_info
_final_observation
_elapsed_steps
```

Episode metrics are written into `infos["episode"]`:

```text
return
episode_len
reward
success_once
success_at_end   when ignore_terminations=True
```

## Training

Install optional helpers:

```bash
pip install -e ".[robotwin]"
```

RoboTwin itself must be available separately. Either install it into the
environment so `import envs.<task_name>` works, or pass `--robotwin-root` to a
RoboTwin checkout.

Minimal local command:

```bash
MPLCONFIGDIR=/tmp python examples/train_ppo_robotwin_rgbd.py \
  --env-id place_shoe \
  --robotwin-root /path/to/RoboTwin \
  --num-envs 4 \
  --num-eval-envs 2 \
  --total-timesteps 10000 \
  --num-steps 16 \
  --step-lim 400 \
  --encoder plain_conv \
  --image-fusion-mode per_key \
  --reward-mode dense \
  --control-mode delta_joint_pos \
  --device cuda
```

Remote command pattern for `6017`:

```bash
ssh 6017 "docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc '
  cd /workspace/rl-garden &&
  export PATH=/opt/venv/openvla/bin:\$PATH &&
  export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} &&
  MPLCONFIGDIR=/tmp python -u examples/train_ppo_robotwin_rgbd.py \
    --env-id place_shoe \
    --robotwin-root /workspace/RoboTwin \
    --num-envs 4 \
    --num-eval-envs 2 \
    --total-timesteps 10000 \
    --num-steps 16 \
    --step-lim 400 \
    --encoder plain_conv \
    --image-fusion-mode per_key \
    --reward-mode dense \
    --control-mode delta_joint_pos \
    --device cuda
'"
```

Adjust `--robotwin-root` to the actual RoboTwin path in the container. If
RoboTwin is already importable in that Python environment, omit it.

For long runs, use tmux and log to the remote host:

```bash
ssh 6017 "mkdir -p /data0/liuzhaohong/Projects/rl-garden/logs && \
  tmux new-session -d -s rlg_robotwin_place_shoe_ppo \
  \"docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc ' \
    cd /workspace/rl-garden && \
    export PATH=/opt/venv/openvla/bin:\\\$PATH && \
    export PYTHONPATH=/workspace/rl-garden:\\\${PYTHONPATH:-} && \
    MPLCONFIGDIR=/tmp python -u examples/train_ppo_robotwin_rgbd.py \
      --env-id place_shoe \
      --num-envs 4 \
      --num-eval-envs 2 \
      --total-timesteps 200000 \
      --num-steps 16 \
      --step-lim 400 \
      --encoder plain_conv \
      --image-fusion-mode per_key \
      --reward-mode dense \
      --device cuda \
  ' 2>&1 | tee /data0/liuzhaohong/Projects/rl-garden/logs/rlg_robotwin_place_shoe_ppo_\$(date +%Y%m%d_%H%M%S).log\""
```

## Tests

The relevant non-SAPIEN unit tests are:

```bash
MPLCONFIGDIR=/tmp pytest -q \
  tests/test_robotwin_env.py \
  tests/test_ppo.py \
  tests/test_policy_kwargs.py
```

On `6017`, use `PYTHONPATH` if the package is not installed editable:

```bash
ssh 6017 "docker exec -e CUDA_VISIBLE_DEVICES=1 liuzhaohong_maniskill_rlgarden bash -lc '
  cd /workspace/rl-garden &&
  export PATH=/opt/venv/openvla/bin:\$PATH &&
  export PYTHONPATH=/workspace/rl-garden:\${PYTHONPATH:-} &&
  MPLCONFIGDIR=/tmp pytest -q tests/test_robotwin_env.py tests/test_ppo.py tests/test_policy_kwargs.py
'"
```

This passed remotely with:

```text
35 passed, 3 warnings in 10.20s
```

Two operational notes from remote testing:

- Without `PYTHONPATH=/workspace/rl-garden`, tests can fail with
  `ModuleNotFoundError: No module named 'rl_garden'` if the package is not
  installed editable in the container.
- On crowded GPUs, CUDA tests can fail with out-of-memory. Re-run on a less
  occupied GPU, for example `CUDA_VISIBLE_DEVICES=1`.

When running tests inside the container as root, clean generated caches after
the command so Mutagen does not have to sync root-owned artifacts:

```bash
find /workspace/rl-garden -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf /workspace/rl-garden/.pytest_cache
```

## Current Limits

This is the v1 integration. The env contract and PPO unit path are in place,
but a full RoboTwin/SAPIEN training smoke has not yet been completed locally.
The next validation step is a short remote `place_shoe` PPO run in a container
where RoboTwin is importable and assets are available.
