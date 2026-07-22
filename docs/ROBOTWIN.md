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
state            14D qpos/proprio vector, float32
rgb_left_wrist   optional left wrist camera, uint8, B x H x W x 3
rgb_right_wrist  optional right wrist camera, uint8, B x H x W x 3
```

When `include_wrist_cameras=True`, missing wrist images are filled with zeros so
the observation space remains stable. When wrist cameras are disabled, those
keys are omitted entirely, and RoboTwin is also configured not to render them.
Task instruction text is returned in `infos`, not in the observation dict.

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
- retries reset with `seed + 1` if RoboTwin reports an unstable randomized
  scene;
- injects dense reward with `build_task_reward()` when `reward_mode="dense"`;
- converts rl-garden actions to RoboTwin qpos+gripper commands;
- tracks reward state needed by dense reward primitives;
- calls `take_action()` with RoboTwin's matching action type;
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

`control_mode="joint_pos"` is also present for direct qpos-style actions.

`control_mode="delta_ee"` exposes RoboTwin's native bounded 14D Cartesian delta action:

```text
left dxyz:            3 dims
left drotvec:         3 dims
left gripper delta:   1 dim
right dxyz:           3 dims
right drotvec:        3 dims
right gripper delta:  1 dim
```

The adapter scales translation by `ee_delta_pos_scale` (default `0.03`),
rotation-vector deltas by `ee_delta_rot_scale` (default `0.15`), converts
rotvecs to RoboTwin's `[w, x, y, z]` quaternion delta format, clamps grippers
to `[0, 1]`, and calls RoboTwin with `action_type="delta_ee"`.

The default and baseline mode remains `delta_joint_pos`; use
`delta_ee` for RL with RoboTwin's native end-effector planner.

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
RoboTwin env configs that rl-garden supports for dense rewards:

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
stack_bowls_three
```

If a task is not in the registry and `reward_mode="dense"`, reset will fail
with a clear missing-factory error. Use `reward_mode="sparse"` to train with
`check_success()` only, or add a factory for the task.

`open_laptop` is currently supported only through sparse rewards. Run it with
`--env-id open_laptop --robotwin.reward-mode sparse`; dense reward construction
is intentionally not registered for this task yet.

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

RoboTwin can raise `UnStableError` during `setup_demo()` when a sampled seed
places objects in an unstable physical state. `RoboTwinTaskAdapter` handles
that case the same way as the RLinf_support vector env: it logs a warning,
clears the task cache, increments the seed, and retries until setup succeeds.
Every reset records the final successful seed back into `reset_state_ids` and
reset `infos["env_seed"]`, so experiment logs reflect the scene that actually
ran. Non-`UnStableError` exceptions still fail fast because they usually
indicate an import, asset, planner, or task configuration problem.

## SAC Replay Memory

RoboTwin's native visual observations are much larger than the images normally
used by ManiSkill SAC. The default RoboTwin camera size is `224x224`, and
rl-garden exposes three RGB views when wrist cameras are enabled:

```text
rgb
rgb_left_wrist
rgb_right_wrist
```

Off-policy SAC stores both `obs` and `next_obs` in replay. Even though
rl-garden stores image observations as `uint8`, not `float32`, three `224x224`
RGB views are still too large for a CUDA-resident replay buffer. For example,
`200_000` transitions require roughly 168 GiB for image `obs + next_obs` alone:

```text
200000 * 2 * 3 views * 224 * 224 * 3 bytes ~= 168 GiB
```

ManiSkill's SAC RGBD baseline handles this by keeping replay on CUDA for
throughput, but using `uint8` image storage, `64x64` or `128x128` observations,
and tuned visual buffer sizes. RLinf's RoboTwin SAC/RLPD path makes a different
tradeoff: it moves trajectories to CPU replay (`.cpu().contiguous()`), avoiding
large GPU replay memory at the cost of CPU-to-GPU transfer during sampling.

rl-garden follows the ManiSkill-style default for RoboTwin SAC because the
framework is GPU-first and CPU replay can become a bottleneck with parallel
sampling. The RoboTwin SAC entrypoint therefore defaults to:

```text
image_size = 64x64
buffer_device = cuda
buffer_size = 100000
```

Use larger images only with a smaller buffer or explicit CPU replay:

```bash
python examples/train_online.py sac \
  --env-backend robotwin \
  --obs-mode rgb \
  --env-id place_shoe \
  --robotwin.robotwin-root /path/to/RoboTwin \
  --camera-width 224 \
  --camera-height 224 \
  --buffer-device cpu \
  --buffer-size 50000
```

The same camera cost matters for on-policy PPO throughput. `place_empty_cup`
in RLinf's RoboTwin configs uses the head camera only and the
`[piper, piper, 0.6]` embodiment. The dedicated rl-garden
`place_empty_cup` launcher follows that setup. Enabling wrist cameras triples
the rendered image streams, adds CPU resize and CUDA tensor transfer work, and
adds encoder work for every rollout step, so it should be treated as an
explicit visual-ablation setting rather than the default for this task.

## Training

Install optional helpers:

```bash
pip install -e ".[robotwin]"
```

RoboTwin itself must be available separately. Either install it into the
environment so `import envs.<task_name>` works, or pass
`--robotwin.robotwin-root` to a RoboTwin checkout.

The default PPO path is compatible with RoboTwin `main` and uses RoboTwin's
standard `take_action()` API. We also tested RoboTwin's `RLinf_support` branch
and its `gen_sparse_reward_data()` step API, but single-action PPO did not show
a meaningful FPS improvement. rl-garden therefore keeps the main-compatible
`take_action()` path as the training interface.

RoboTwin may load curobo/warp during reset. In containers, set `HOME=/tmp` and
`XDG_CACHE_HOME=/tmp` so warp writes its kernel cache to a writable temporary
directory instead of `/.cache` or a synced workspace. This avoids permission
failures and root-owned cache files.

When using `RLinf_support`, pass `--robotwin.assets-path` as the RoboTwin repository
root, not the `assets/` subdirectory. That branch resolves assets as
`$ASSETS_PATH/assets/...`.

Minimal PPO command:

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
python examples/train_online.py ppo \
  --env-backend robotwin \
  --obs-mode rgb \
  --env-id place_shoe \
  --robotwin.robotwin-root /path/to/RoboTwin \
  --robotwin.head-camera-type Train_D435_128x96 \
  --camera-width 64 \
  --camera-height 64 \
  --num-envs 4 \
  --num-eval-envs 2 \
  --total-timesteps 10000 \
  --num-steps 16 \
  --robotwin.step-lim 400 \
  --encoder plain_conv \
  --image-fusion-mode per_key \
  --robotwin.reward-mode dense \
  --control-mode delta_joint_pos
```

For `place_empty_cup`, prefer the RLinf-aligned defaults:

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
python examples/train_online.py ppo \
  --env-backend robotwin \
  --obs-mode rgb \
  --env-id place_empty_cup \
  --robotwin.robotwin-root /path/to/RoboTwin \
  --robotwin.head-camera-type Train_D435_128x96 \
  --camera-width 64 \
  --camera-height 64 \
  --num-envs 4 \
  --num-eval-envs 2 \
  --total-timesteps 10000 \
  --num-steps 16 \
  --robotwin.step-lim 200 \
  --robotwin.assets-path /path/to/RoboTwin \
  --robotwin.embodiment piper piper 0.6 \
  --robotwin.no-include-wrist-cameras \
  --encoder plain_conv \
  --image-fusion-mode per_key \
  --robotwin.reward-mode dense \
  --control-mode delta_joint_pos
```

The `fps` printed by PPO is rollout action FPS. One RoboTwin action can run
TOPP plus many internal SAPIEN physics/render steps, so this number is not
comparable to ManiSkill GPU-vectorized simulator FPS.

### Training-Speed Knobs

RoboTwin renders camera frames at its own camera-config resolution before
rl-garden resizes observations for the policy. For the 64x64 place-empty-cup
launcher, rl-garden asks RoboTwin to use `Train_D435_128x96` and then resizes
the resulting head-camera RGB to `64x64`. This is different from only setting
`--camera-width 64 --camera-height 64`, which controls the policy input size.

For RL training, rl-garden defaults RoboTwin's per-substep render update off:
`render_every_control_step=False`. RoboTwin only needs a camera frame after one
high-level action finishes, so updating the renderer inside every low-level
control substep wastes time. Keep the flag on only for debugging or workflows
that need substep-level frames.

`--control-step-cap` can cap/downsample the low-level TOPP trajectory executed
for each RoboTwin action. This can improve FPS substantially but changes the
smoothness of the physical motion. In rl-garden's RoboTwin launchers, this is
set to `16` by default because it consistently improved PPO throughput in our
place-empty-cup experiments.

rl-garden also supports `--disable-topp` for PPO RoboTwin runs. When enabled,
the environment skips RoboTwin's qpos TOPP pass and uses a simple linear joint
interpolation fallback inside the control loop. This path is kept as an
explicit opt-in experiment knob. The default remains TOPP-enabled behavior.

The RL launchers also leave `random_light=False` and
`crazy_random_light_rate=0.0` by default. Random lighting is useful for
robustness experiments, but it is unnecessary for speed profiling and can add
variance to render timing.

### Optimization Notes

Recent RoboTwin speed experiments explored TOPP worker pools, ctrl-loop
serialization, shard executors, shared-memory observation transfer, and
multi-process launchers. Those experiments remain preserved on
`dev/fast-robotwin` and RoboTwin's `dev/rl-garden-ctrl-gate` branches, but they
are not part of the minimal supported path.

The two changes intentionally kept in the maintained training path are:

- `--control-step-cap 16` as the default launcher setting
- `--disable-topp` as an opt-in switch for no-TOPP profiling and training runs

The other experiments either added too much complexity for the observed gain or
did not improve single-GPU PPO throughput reliably enough to justify carrying
them in the main integration path.

Place-empty-cup PPO with the dedicated launcher:

```bash
RLG_ROBOTWIN_ROOT=/path/to/RoboTwin \
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
scripts/train_ppo_robotwin_place_empty_cup_rgbd.sh
```

Minimal SAC command:

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
python examples/train_online.py sac \
  --env-backend robotwin \
  --obs-mode rgb \
  --env-id place_shoe \
  --robotwin.robotwin-root /path/to/RoboTwin \
  --num-envs 4 \
  --num-eval-envs 2 \
  --total-timesteps 10000 \
  --learning-starts 128 \
  --training-freq 8 \
  --batch-size 32 \
  --buffer-size 1024 \
  --robotwin.step-lim 400 \
  --encoder plain_conv \
  --image-fusion-mode per_key \
  --robotwin.reward-mode dense \
  --control-mode delta_joint_pos \
  --buffer-device cuda
```

Generic remote/container command pattern:

```bash
ssh <ssh-alias> "docker exec \
  -e CUDA_VISIBLE_DEVICES=<gpu-id> \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp \
  -e ROBOT_PLATFORM=ALOHA \
  <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  export PYTHONPATH=<container-workspace-path>:<container-robotwin-root-path>:\${PYTHONPATH:-} &&
  RLG_ROBOTWIN_ROOT=<container-robotwin-root-path> \
  MPLCONFIGDIR=/tmp scripts/train_ppo_robotwin_place_empty_cup_rgbd.sh \
    --num-envs 4 \
    --num-eval-envs 2 \
    --total-timesteps 10000
'"
```

Adjust `<container-robotwin-root-path>` to the actual RoboTwin path in the
container. Agents should read `.agents/rules/remote-training-sop.md` and their
ignored `.agents/local/personal_config.md` for the concrete remote bindings.

For long runs, use tmux and log to the remote host:

```bash
ssh <ssh-alias> "mkdir -p <remote-project-path>/logs && \
  tmux new-session -d -s rlg_robotwin_place_shoe_ppo \
  \"docker exec -e CUDA_VISIBLE_DEVICES=<gpu-id> -e HOME=/tmp -e XDG_CACHE_HOME=/tmp <container-name> bash -lc ' \
    cd <container-workspace-path> && \
    export PATH=<python-env-bin-path>:\\\$PATH && \
    export PYTHONPATH=<container-workspace-path>:<container-robotwin-root-path>:\\\${PYTHONPATH:-} && \
    MPLCONFIGDIR=/tmp python -u examples/train_online.py ppo \
      --env-backend robotwin \
      --obs-mode rgb \
      --env-id place_shoe \
      --robotwin.robotwin-root <container-robotwin-root-path> \
      --num-envs 4 \
      --num-eval-envs 2 \
      --total-timesteps 200000 \
      --num-steps 16 \
      --robotwin.step-lim 400 \
      --encoder plain_conv \
      --image-fusion-mode per_key \
      --robotwin.reward-mode dense \
  ' 2>&1 | tee <remote-project-path>/logs/rlg_robotwin_place_shoe_ppo_\$(date +%Y%m%d_%H%M%S).log\""
```

## Tests

The relevant non-SAPIEN unit tests are:

```bash
MPLCONFIGDIR=/tmp pytest -q \
  tests/test_robotwin_env.py \
  tests/test_ppo.py \
  tests/test_policy_kwargs.py
```

In a remote/container environment, use `PYTHONPATH` if the package is not
installed editable:

```bash
ssh <ssh-alias> "docker exec <container-name> bash -lc '
  cd <container-workspace-path> &&
  export PATH=<python-env-bin-path>:\$PATH &&
  export PYTHONPATH=<container-workspace-path>:\${PYTHONPATH:-} &&
  MPLCONFIGDIR=/tmp pytest -q tests/test_robotwin_env.py tests/test_ppo.py tests/test_policy_kwargs.py
'"
```

Two operational notes from remote testing:

- Without `PYTHONPATH=<container-workspace-path>`, tests can fail with
  `ModuleNotFoundError: No module named 'rl_garden'` if the package is not
  installed editable in the container.
- On crowded GPUs, CUDA tests can fail with out-of-memory. Re-run on a less
  occupied GPU, for example `CUDA_VISIBLE_DEVICES=1`.

When running tests inside the container as root, clean generated caches after
the command so Mutagen does not have to sync root-owned artifacts:

```bash
find <container-workspace-path>/rl_garden <container-workspace-path>/examples <container-workspace-path>/tests \
  -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf <container-workspace-path>/.pytest_cache
```

## Current Limits

This is the v1 integration. The env contract and PPO unit path are in place,
and a short remote PPO smoke has completed for `place_empty_cup` with two
parallel RoboTwin envs and `64x64` image observations. Broader validation is
still needed for long runs, larger vector sizes, and additional tasks.
