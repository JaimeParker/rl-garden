# IsaacLab Camera-Observation Stall

Camera-observation PPO training on Isaac Lab 2.x once froze after the first
rollout and update cycle. The stall no longer reproduces on any supported
route. This page records the symptom, the resolution, the workaround that
remains in the code, and what is known about driver and Isaac Sim versions.

## Symptom

A PPO run with `--env_backend isaaclab --obs.rgb <camera>` stopped producing
output, stayed alive at 100% CPU on a single thread (all other Kit threads
idle), and never raised an exception or exited. The last log lines were Kit
plugin-unload warnings, which also appear during a normal `env.close()`:

```
[Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!
[Warning] [carb] Client omni.syntheticdata.plugin Failed to acquire interface [omni::graph::core::INode v4.10] while unloading all plugins
```

Seeing these lines alone is not proof of a stall. The signal is that the
process neither prints further output nor exits.

## Resolution

The reproduction below completes on every supported route:

| Route | Condition |
|---|---|
| Isaac Lab 3.0, Newton Warp renderer (default for rl-garden camera tasks) | Works on any driver that Isaac Sim's CUDA build accepts, including 570.133.07. Kit's render pipeline is bypassed. |
| Isaac Lab 3.0, Kit RTX (`--isaaclab.renderer isaacsim_rtx`) | Needs a driver that Isaac Sim 6.2's Kit accepts (works on 570.195.03). On 570.133.07 Kit rejects the driver and the adapter raises `RuntimeError` ("produced only zero frames") instead of training on black frames. |
| Isaac Lab 2.3.x, Kit RTX (`TiledCamera`) | Works on `cuda:0`. On any other GPU, Isaac Sim 5.1's usdrt refuses the camera (`GPUs other than cuda:0 are not currently supported`) and the process spins at simulation start. The backend raises `ValueError` for `--isaaclab.sim_device` other than `cuda:0` on camera tasks, before Kit boots. |

The original stall is most likely the same device restriction: the
reproduction ran the camera task on a GPU other than `cuda:0`, which Isaac Sim
5.1 does not support for rendering. This is not proven, because the old
symptom appeared after the first PPO update and the current one at simulation
start, and the old environment used a different Isaac Sim and torch install.
It is the only difference that reproduces a hang today.

On 2.3.x the Kit app is closed explicitly after `env.close()`; without that,
the process could spin in Kit's plugin unload after a successful run.

## Workaround in the code

`_IsaacLabVecEnvAdapter.post_update_sync()` (`rl_garden/envs/isaaclab/env.py`)
calls `torch.cuda.synchronize()` once per rollout and update cycle when the
env has camera or depth observations. The on-policy training loop calls it
through a duck-typed hook (`_update_and_log` in
`rl_garden/algorithms/on_policy.py`, and the policy-distillation loop), so
backends that do not define it are unaffected. It is harmless but no longer
known to be necessary. Calling the sync on every `step()` instead made the
original hang worse.

## Reproduction and diagnostic

This command runs six rollout and update cycles, the shape that stalled on 2.x
after the first update. Set `cuda:N` to `cuda:0` on Isaac Lab 2.x.

```bash
python -u examples/train_online.py ppo --env_backend isaaclab \
  --env_id RlGarden-Cartpole-Direct-Camera-v0 --obs.rgb front \
  --num_envs 2 --eval_freq 0 --total_timesteps 192 --num_steps 16 \
  --log-freq 32 --log-type none --isaaclab.sim_device cuda:N
```

Add `--isaaclab.renderer isaacsim_rtx` to test Kit RTX on 3.0.

To tell a stall from a slow run, look at the hot thread of the hung process:
`top -H -b -n 1 -p <PID>` shows one thread at 99.9% CPU and the rest of Kit's
threads sleeping, and `/proc/<PID>/wchan` for that thread reads `0`. That is a
CPU spin-wait rather than a blocked wait.

## Facts worth keeping

- Isaac Sim 6.2's Kit RTX rejects drivers in `[570.00, 570.158.01)` and
  recommends 580.95.05; Isaac Lab 3.0's install page asks for 580.65.06 or
  newer. On a rejected driver Kit still boots, PhysX works, and the camera
  returns all-zero frames without a Python exception. The adapter's
  zero-frame check (`_check_frames_nonzero` in
  `rl_garden/envs/isaaclab/env.py`) turns that into an error.
- Isaac Sim 5.1's Kit rejects only drivers below 535.129, so the 570 drivers
  were never the problem on 2.x.
- Isaac Sim 5.1 renders cameras only on `cuda:0`. Isaac Sim 6.2 has no such
  restriction for the Newton renderer, and its RTX renderer uses physical GPU
  0 as Kit's active GPU regardless of `--isaaclab.sim_device`.
- A similar upstream report (Isaac Lab issue #4951: Isaac Sim 5.1, `TiledCamera`,
  process at 100% CPU inside the tiled render) was answered by the maintainers
  as a driver problem. Given the 5.1 driver policy above, the device
  restriction fits what was observed better.

Ruled out as the cause of the original stall:

- adapter correctness, GPU memory pressure and the `TiledCamera` sensor class
  (a plain `Camera` stalled identically)
- the `rendering_mode` preset, and the image resolution or `num_envs`

If a stall reappears, check the Kit log for the driver lines and the usdrt
device line first, then try the other renderer route on 3.0 or `cuda:0` on
2.x.
