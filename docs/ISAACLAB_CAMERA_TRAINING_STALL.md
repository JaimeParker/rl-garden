# IsaacLab Camera-Observation Training Stall (6017-nofwd) — Unresolved

**Status: open, unresolved.** A partial mitigation is in place
(`post_update_sync()`, see below) but does **not** reliably fix it. Treat
camera-observation IsaacLab training on host `6017-nofwd` as unreliable —
runs can hang indefinitely partway through and need to be killed manually.
State-only IsaacLab training (no camera) is unaffected and fully reliable.

This document records the full investigation for whoever picks this up next,
so the diagnostic ground already covered doesn't need to be re-derived.

## Symptom

A PPO training run using the IsaacLab backend with `--obs_mode rgb` (or
`rgbd`) hangs indefinitely, typically within the first 1-2 rollout+update
cycles, sometimes later. The process:

- Stops producing any new stdout output (running with `python -u`,
  unbuffered, so this is a real freeze, not a buffering artifact).
- Stays alive, consuming CPU and GPU resources — it does **not** crash, does
  **not** raise a Python exception, does **not** get OOM-killed.
- Must be killed manually (`kill -9`); it never recovers or times out on its
  own.

The last log lines before the freeze are always the same signature — Kit
warnings that look like extension/graph teardown, immediately followed by
silence:

```
[Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
  (repeated ~8-13 times)
[Warning] [omni.graph.core.plugin] Could not find category 'Replicator' for removal
[Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Core' for removal
[Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!
[Warning] [carb] Client omni.syntheticdata.plugin Failed to acquire interface [omni::graph::core::INode v4.10] while unloading all plugins
```

**Important:** these exact lines also appear during a *normal, successful*
`env.close()` at the end of a run. Seeing them is not itself proof of a
stall — the distinguishing signal is whether the process ever produces
further output / exits afterward.

## Environment

- Host: `6017-nofwd` (shared, multi-GPU, multi-tenant — 4× RTX 4090, other
  users' jobs routinely at 90-100% utilization).
- Container: `liuzhaohong_maniskill_isaaclab` (see
  `.agents/local/personal_config.md`, "Working IsaacLab + rl-garden setup",
  for the full build recipe).
- IsaacLab installed at `/opt/envs/isaaclab` inside the container; Isaac Sim
  5.1 (`isaacsim.core.simulation_manager`, `omni.physics.tensors`).
- `torch==2.6.0+cu124`, `torchvision==0.21.0`, driver/CUDA per the container
  (see personal_config.md).
- No `gdb`, `strace`, or `py-spy` available in this container — diagnosis was
  limited to `/proc/<pid>/*`, `top -H`, and application-level instrumentation
  (adding temporary `print(..., flush=True)` statements to rl-garden's own
  code and rerunning).

## Reproduction

```bash
docker exec -d liuzhaohong_maniskill_isaaclab bash -lc '
  source /opt/venv/openvla/bin/activate
  cd /workspace/Projects/isaac_sim && source ./setup_conda_env.sh
  export PYTHONPATH="/opt/venv/openvla/lib/python3.11/site-packages:$PYTHONPATH"
  cd /workspace/Projects/rl-garden
  python -u examples/train_online.py ppo --env_backend isaaclab \
    --env_id RlGarden-Cartpole-Direct-Camera-Plain-v0 \
    --obs_mode rgb --camera_width 64 --camera_height 64 --num_envs 2 \
    --eval_freq 0 --total_timesteps 192 --num_steps 16 \
    --isaaclab.sim_device cuda:2 \
    --log-type none > /tmp/run.log 2>&1
'
```

`RlGarden-Cartpole-Direct-Camera-Plain-v0` is a diagnostic task
(`rl_garden/envs/isaaclab/tasks/cartpole_direct_camera_plain.py`) using
IsaacLab's plain `Camera` sensor instead of `TiledCamera`, kept in the repo
specifically to reproduce and test this issue in isolation from the
`RlGarden-Cartpole-Direct-Camera-v0` task (which uses `TiledCamera`, the
normal/recommended choice — see `ISAACLAB_CUSTOM_TASKS.md`). Both stall
identically (see "Ruled out" below), so either reproduces the bug.

Reproduction rate observed: **stalled in the large majority of attempts**
across this investigation. Counting only full training-run attempts (not
standalone env/encoder diagnostics that didn't go through the real PPO
loop): roughly 2 out of a dozen-plus attempts completed end-to-end
successfully across the whole investigation, and a dedicated batch of 3
back-to-back reruns of the identical command *after* the mitigation below
was implemented stalled 3/3.

## Diagnostic timeline

Numbered in the order things were actually tried, including dead ends —
kept here so they aren't retried blindly.

1. **First observed** while verifying image-observation support in the
   IsaacLab backend (`_IsaacLabVecEnvAdapter`, `rl_garden/envs/isaaclab/env.py`).
   Initial hypothesis: a bug in the new adapter code.

2. **Ruled out: adapter code correctness.** A standalone diagnostic script
   (bypassing PPO entirely) called `make_isaaclab_env()` directly, did 20
   `env.step()` calls under `torch.no_grad()` (mimicking rollout collection),
   then a real `forward()` + `backward()` + `optimizer.step()` on a small
   `nn.Conv2d` using the collected images. This completed successfully
   ("ALL DONE" printed, no stall) — proving env creation, the `Dict`
   observation space, and the adapter's `step()`/`reset()` are all correct.

3. **Ruled out: GPU memory pressure.** GPU memory usage was tracked via
   `nvidia-smi` before/during/after both successful and stalled runs.
   Baseline "other users' " memory on the target GPU was stable
   (~15.6 GiB used out of 24.5 GiB) across every observation, whether the run
   succeeded or stalled. During one stalled run, memory usage on the target
   GPU actually *dropped* (~18.3 GiB → ~16.9 GiB) right around the moment of
   the stall — the opposite of what a memory-pressure/OOM-adjacent slowdown
   would look like.

4. **Ruled out: cuDNN version mismatch (real bug, but separate from the
   stall).** Before the stall was even reachable, a `RuntimeError: cuDNN
   error: CUDNN_STATUS_NOT_INITIALIZED` occurred on the very first `Conv2d`
   forward pass. Root cause: `nvidia-cudnn-cu12` installed version
   (`9.1.0.70`) didn't match what `torch.backends.cudnn.version()` reported
   internally (`92000` = 9.2.0) — a stale/mismatched library, likely left
   over from an earlier `torch` reinstall in this container's history. Fixed
   with `pip install --force-reinstall --no-cache-dir torch==2.6.0
   torchvision==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple` in the
   `openvla` venv. **Side effect:** this force-reinstall also bumped
   `numpy`/`pillow`/`gymnasium`/`typing_extensions` to newer versions,
   breaking pins several other packages need (`isaaclab` wants
   `gymnasium==1.2.1`, but the working baseline this whole IsaacLab
   integration was built/verified against uses `gymnasium==0.29.1`, matching
   `mani-skill`'s pin — `isaaclab`'s own declared pin is not runtime-enforced,
   same as its `torch>=2.7` pin isn't). Re-pinned back:
   `pip install "numpy<2" "pillow==11.3.0" "gymnasium==0.29.1" -i
   https://pypi.tuna.tsinghua.edu.cn/simple`. This is a real, permanent fix
   for a real (separate) bug — it is **not** the cause of the render stall,
   confirmed because the stall reproduces before and after this fix, with a
   different symptom (crash vs. hang) either side.

5. **Ruled out: camera resolution mismatch (real bug, but separate from the
   stall).** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` — the
   diagnostic task's `TiledCameraCfg` used `width=100, height=100` (copied
   from IsaacLab's own official camera example), but rl-garden's default
   `PlainConv` image encoder only supports 64×64 or 128×128 (hardcoded
   conv/pool stride schedule, see `rl_garden/encoders/plain_conv.py`'s
   docstring). Fixed by using 64×64 and passing matching `--camera_width 64
   --camera_height 64` on the CLI. Real, permanent fix; not the stall's
   cause (same stall reproduces with correct resolution too).

6. **Ruled out: `clone_in_fabric=True` (real bug, separate from the
   stall).** The state-only base task's `scene` cfg sets
   `clone_in_fabric=True`; inherited unchanged by a first camera-task
   attempt, this caused `RuntimeError: CUDA error: device-side assert
   triggered` / `IndexKernel.cu:94 ... index out of bounds` inside
   `TiledCamera`'s `reset()` (mis-sized per-env sensor buffers). Fixed by
   omitting `clone_in_fabric` for camera-task scene configs, matching what
   IsaacLab's own `cartpole_camera_env.py` example does. Real, permanent
   fix; not the stall's cause.

7. **Ruled out: `TiledCamera` specifically.** Wrote a diagnostic task variant
   (`cartpole_direct_camera_plain.py`) using IsaacLab's plain `Camera` sensor
   class instead of `TiledCamera` (spawned directly in an overridden
   `_setup_scene`, bypassing the scaffold's `camera_cfg` field which is typed
   for `TiledCameraCfg`). Stalls identically. Rules out `TiledCamera`'s
   tiled-rendering/Warp-kernel reshape path as the specific culprit.

8. **Ruled out: `rendering_mode` setting.** IsaacLab's `AppLauncher` accepts
   a `rendering_mode` kwarg (`"performance"`/`"balanced"`/`"quality"`,
   controlling DLSS/post-processing presets). Tried
   `rendering_mode="performance"` (the lightest preset) when cameras are
   enabled — no difference, identical stall.

9. **Ruled out: resolution/`num_envs` scale.** Reproduces identically at
   100×100/`num_envs=4` and at 64×64/`num_envs=2` (the smallest practical
   configuration).

10. **Instrumented the real code to pinpoint the exact stall location.**
    Added temporary `print(..., flush=True)` statements throughout
    `OnPolicyAlgorithm.learn()`'s rollout loop
    (`rl_garden/algorithms/on_policy.py`) and `PPO.train()`'s minibatch
    update loop (`rl_garden/algorithms/ppo.py`), then reran the real training
    command (not a standalone diagnostic). Findings, in order of what was
    proven:
    - **All 16 rollout steps of the first cycle succeed**, including every
      real `PPOPolicy`/`CombinedExtractor`/`PlainConv` forward pass through
      the actual encoder architecture (not a simplified stand-in).
    - **The full PPO update phase succeeds too**: 3 epochs × ~10 minibatches
      = ~30 real backward passes + `optimizer.step()` calls through the CNN,
      all completing.
    - **The stall is specifically the *second* rollout cycle's first
      `env.step()`** — i.e., the first render call that follows the heavy
      backward-pass burst of a PPO update. This is the precise, reproducible
      trigger condition.
    - (All temporary prints were reverted afterward — `git diff`/`git
      checkout` on both files.)

11. **Mitigation: `torch.cuda.synchronize()` between update and next
    rollout.** Based on finding #10, added a call right after the update
    phase, before the next rollout's first `step()`. First attempt: called
    on *every* `step()` (inside `_IsaacLabVecEnvAdapter.step()`) — this made
    things **worse**, reliably reintroducing the hang (apparently disrupts
    Kit's own async cadence between consecutive render calls). Corrected
    placement: **once per rollout+update cycle**, matching exactly where the
    instrumented run in #10 proved it works — via a duck-typed
    `post_update_sync()` hook that `OnPolicyAlgorithm.learn()` calls if the
    env defines it (no-op for ManiSkill/other backends). One fully
    instrumented run with this placement completed all 6 rollout+update
    cycles cleanly. This is the mitigation currently in the code
    (`_IsaacLabVecEnvAdapter.post_update_sync`, `rl_garden/envs/isaaclab/env.py`;
    call site in `OnPolicyAlgorithm.learn()`, `rl_garden/algorithms/on_policy.py`).

12. **Reliability check: the mitigation is NOT reliable.** After confirming
    the fix in principle (#11), reran the identical command 3 more times
    with no debug instrumentation (clean code). **All 3 stalled.** This
    means the earlier single success was not a repeatable fix — either the
    dense `print(flush=True)` calls throughout the instrumented version
    incidentally provided timing/pacing that a bare sync call doesn't
    reproduce, or it was a lucky timing window under otherwise-identical
    code. A follow-up test tried the bare `torch.cuda.synchronize()` call
    exactly as used in the one successful instrumented run (no explicit
    device argument, vs. the current code's `torch.cuda.synchronize()` with
    no device arg — these turned out to already be the same after checking
    `self.device`'s actual type: a plain string `'cuda:2'`) — that specific
    variant got through 5 of 6 expected cycles before also stalling on the
    final one. So the sync **does measurably help** (delays the failure,
    sometimes past several cycles) but does not close it out reliably.

13. **Root cause mechanism identified** (prompted by the user noticing that
    IsaacLab/PhysX should be GPU-driven, so sustained high CPU usage during
    an apparent "hang" was suspicious). During a live stall:
    ```
    docker exec liuzhaohong_maniskill_isaaclab top -H -b -n 1 -p <PID>
    ```
    showed **342 threads total, only 1 running** — the main Python thread,
    pinned at 99.9% CPU — **and 341 sleeping** (all of Kit's `carb.tas+`
    task-worker threads, the `cuda-Ev+` CUDA-event thread, and the `[vkps]`
    Vulkan-present thread fully idle at 0%). Checking the hot thread:
    ```
    cat /proc/<PID>/status   # State: R (running)
    cat /proc/<PID>/wchan    # 0
    ```
    `wchan: 0` on a `Running` thread means it is **not blocked in any
    kernel wait function** — no futex, no I/O, no blocking-on-GPU-fence
    syscall. This is the signature of a **CPU busy-wait / spin-loop polling
    for a GPU completion signal that never arrives** — not genuine
    GPU-bound computation, and not a proper OS-level block.
    `/proc/<PID>/stack` and `/proc/<PID>/syscall` were both permission-denied
    in this container (no `ptrace`/ root capability for that), so the exact
    C++ call site inside Kit/PhysX/the renderer could not be captured.

    This mechanism is consistent with every other observation: `torch.cuda.
    synchronize()` only "helps" when the GPU-side work it flushes happens to
    complete *before* Kit's polling loop starts checking for it — a timing
    race, not something a single deterministic call can guarantee. GPU
    memory sometimes dropping right at the stall (see #3) is explained by
    the GPU-side work having already finished; only the CPU-side signal/fence
    delivery is stuck.

## Root cause (best current understanding)

A race/bug in Kit's (or PhysX's) CPU-GPU synchronization layer: something in
the render pipeline polls (busy-waits) for a GPU-side completion signal, and
under concurrent heavy PyTorch CUDA usage (specifically: right after a
sustained backward-pass burst), that signal can be delivered late enough, or
lost/never posted, that the poll loop spins forever. This looks like a bug
specific to this Isaac Sim/Kit build's interaction with PyTorch on this
host's driver/CUDA stack — **not a bug in rl-garden's code**. Everything in
rl-garden (`_IsaacLabVecEnvAdapter`, the `RLGardenDirectRLEnv` scaffold, the
PPO training loop) has been verified correct in isolation; the stall is
purely about what happens inside IsaacLab/Kit's own C++ layer when a render
call follows heavy compute.

## Current mitigation and its limits

`_IsaacLabVecEnvAdapter.post_update_sync()` (`rl_garden/envs/isaaclab/env.py`)
calls `torch.cuda.synchronize()` when `obs_mode != "state"`. It is invoked
once per rollout+update cycle from `OnPolicyAlgorithm.learn()`
(`rl_garden/algorithms/on_policy.py`), via a duck-typed hook lookup
(`getattr(self.env, "post_update_sync", None)`) — **zero cost/behavior
change for ManiSkill or any other backend** that doesn't define this method.

This measurably reduces (but does not eliminate) how often the stall occurs.
Treat any camera-observation IsaacLab training run on this host as likely to
hang and requiring a hard timeout + manual kill, not something safe to
launch unattended.

## Next steps if picking this back up

1. **Get a real stack trace at the moment of stall.** `gdb`/`py-spy attach`
   would need to be installed in the container (or a privileged debug
   sidecar container attached to the same PID namespace) — neither was
   available this session. This is the single highest-value next step; it
   would show the actual C++/Python frames the spinning thread is stuck in,
   turning "we know it's a spin-wait" into "we know exactly which fence/API
   call it's polling on."
2. **Test whether skrl or rsl_rl hit the same issue.** Neither is installed
   in this container. Their source code doesn't show any special
   CPU-GPU-sync workaround around the update phase (checked
   `3rd_party/skrl/skrl/envs/wrappers/torch/isaaclab_envs.py` and the
   official `scripts/reinforcement_learning/skrl/train.py` inside the
   IsaacLab install) — so it's plausible they'd hit this too, but that's
   inferred, not confirmed. Installing skrl and running its official
   camera-task example would settle whether this is IsaacLab/Kit-general or
   specific to rl-garden's exact call pattern.
3. **Try a different/newer Isaac Sim build**, if one becomes available on
   this host — this could plausibly be a fixed-in-a-later-version Kit bug.
4. **Architectural workaround (larger change, not attempted):** isolate
   rendering and PyTorch training into separate processes, communicating
   observations/actions over shared memory or IPC, so the two CUDA
   contexts/streams never directly compete within one process. This is a
   significant redesign, not something to do without discussing scope first.
