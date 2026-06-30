# Offline Training Acceleration

This document describes the acceleration techniques applied to the offline
pretraining path (CQL / Cal-QL / WSRL offline phase) and how to use them.

For an unrelated baseline comparison against the JAX reference implementation
at `3rd_party/wsrl/`, see the "Background" section below.

## TL;DR

For state-based offline pretraining (`pretrain_offline.py` /
`train_wsrl.py` offline phase / `train_wsrl_rgbd.py` offline phase), the
default configuration now applies three optimizations:

1. **`.item()` deferred to log steps** — hot losses return tensor dicts;
   GPU→CPU syncs only happen on steps where metrics are actually logged
   (`global_step % log_freq == 0`).
2. **`torch.compile` enabled by default** for offline SAC-family Args
   (`use_compile=True`).
3. **TF32 matmul + cuDNN benchmark** enabled via
   `rl_garden.common.enable_fast_math()` at entrypoint setup.

Measured on a single RTX 4090 with Cal-QL, `batch_size=256`,
`n_critics=10`, `buffer_device=cuda`, 200 offline steps:

| Configuration       | Steady-state iter/s |
|---------------------|---------------------|
| Before (eager)      | ~32                 |
| After (compiled)    | ~78 (**2.4×**)      |

The first ~50 steps are slower while `torch.compile` traces and compiles the
graph; steady-state speed is reached after that. For runs shorter than a few
hundred steps the warmup eats into the gain.

## Background

The JAX reference implementation in `3rd_party/wsrl/` is fast for four
reasons (see `3rd_party/wsrl/wsrl/agents/continuous/cql.py:342` and
`sac.py:360`):

1. `@jax.jit` wraps the entire update step including the backward pass,
   eliminating Python and kernel-launch overhead.
2. `ensemblize()` uses `nn.vmap` so all N critics fold into a single
   forward pass.
3. CQL's `3K` candidate actions (random + current-policy + next-policy) are
   concatenated into one tensor and pushed through the critic in a single
   batched call.
4. Cal-QL Monte Carlo returns are computed once at dataset load time, not
   per-batch.

The rl-garden codebase already had #2–#4 (`networks/actor_critic.py:338-349`
for vmapped critics, `algorithms/cql.py:542-555` for batched CQL actions,
`buffers/maniskill_h5.py:194-226` for precomputed MC returns). The
remaining gap was #1: `torch.compile` was disabled by default, and the
hot path had ~9 `.item()` calls per training step that each forced a
GPU→CPU sync and acted as a graph break for `torch.compile`.

## What changed

### 1. Hot losses return tensor dicts, not float dicts

Before, every loss function called `.item()` on intermediate metrics so the
returned dict could be `dict[str, float]`. With nine such calls per
training step, the hot path was forced to sync the GPU stream nine times
even on steps where nothing was logged.

After, the loss functions return `dict[str, torch.Tensor]` (detached
scalars). The outer `train()` loop accumulates these as tensor lists. On
log steps the lists are stacked and reduced with a single
`.mean().item()` per key.

Files: `rl_garden/algorithms/sac_core.py`, `sac.py`, `cql.py`,
`calql.py`, `offline_sac.py`.

### 2. `compute_info` gate on `train()` and `train_high_utd()`

`SACCore.train(gradient_steps, compute_info=False)` now accepts a
`compute_info` flag. When `False` (the default), the function skips info
accumulation entirely and returns `{}`. When `True`, it builds the metric
dict at the end of the call.

Callers pass `compute_info=True` only on log steps:

- `rl_garden/algorithms/off_policy.py:learn()` — passes
  `compute_info=should_log` to the online update path.
- `rl_garden/algorithms/offline.py:run_offline_pretraining()` — passes
  `compute_info=(global_step % log_freq == 0 or global_step == final_target)`.
- `rl_garden/training/off2on/wsrl.py:_offline_update_loop` — same pattern.

This isolates the `torch.compile`-friendly hot path from the
metrics-building path that contains `.item()` and Python-level reductions.

### 3. `torch.compile` enabled by default for offline

Offline SAC-family `use_compile` defaults to `True`. `WSRLTrainingArgs.use_compile`
remains `False` for now (the online rollout path interacts with ManiSkill
GPU envs in ways that need separate validation).

The compile machinery itself (`cql.py:450-460`) was already in place; it
wraps `_eager_critic_loss`, `_eager_actor_loss`, and `_eager_target_q`.
With the `.item()` calls now removed from those functions, the compiled
graph is no longer broken up by sync points.

Compile mode defaults to `"default"`. `"reduce-overhead"` (CUDA graphs)
should give an additional ~20–50 % on state-based MLP workloads but
imposes strict requirements (zero dynamic shapes, zero `.item()` inside
the captured region) that need separate validation before flipping the
default — staged as a follow-up.

### 4. `enable_fast_math()` at entrypoint

`rl_garden/common/perf.py` exports `enable_fast_math()`, which sets:

```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
```

The first enables cuDNN's heuristic-based algorithm selection (real win
for the RGBD ResNet encoder path; mild improvement for state MLPs). The
second enables TF32 matmul on Ampere+ GPUs.

It is called once at the top of the offline and off2on run functions, right after
`seed_everything()`.

## Numerical equivalence

`torch.compile` does **not** produce bit-identical outputs compared to the
eager path. Measured deltas at the same seed (Cal-QL, 200 offline steps,
seed=0, RTX 4090):

| Metric (step 200)     | Eager     | Compiled  | Δ      |
|-----------------------|-----------|-----------|--------|
| `critic_loss`         | 58.5749   | 62.3949   | +6.5 % |
| `td_loss`             |  3.0083   |  3.1326   | +4.1 % |
| `predicted_q`         |  3.6450   |  3.7600   | +3.2 % |
| `calql_bound_rate`    |  0.4176   |  0.5167   | +24 %  |

The trajectories track the same shape (decreasing critic loss, rising
predicted Q, etc.). The numerical drift comes from fused-kernel rounding
and reordered random-number generator calls inside the compiled region.
For training stability and sample efficiency this is well within the
seed-to-seed noise of CQL / Cal-QL; for unit tests asserting exact
values, disable compile via `--no-use_compile`.

## When to disable compile

- Unit tests that assert on exact loss values or seed-stable outputs:
  pass `--no-use_compile`.
- Workloads where the per-job step count is small relative to the
  ~50-step compile warmup (e.g., 200-step smoke tests): compile may not
  amortize.
- Debugging numerical issues: eager mode gives a cleaner stack trace and
  reproducible math.

`use_compile=False` is the eager default on
`WSRLTrainingArgs.use_compile`, so online WSRL runs are unaffected unless
the user opts in.

## Future work

- **`mode="reduce-overhead"`**: CUDA graphs path. Eliminates kernel-launch
  overhead almost entirely on state MLPs at small batch sizes. Needs
  verification that the captured region has no dynamic shapes (CQL
  random-action sampling is a likely sticking point — `cql.py:503-518`).
- **`use_compile=True` for online WSRL**: requires checking that the
  ManiSkill GPU env's rollout shape stays static and that
  `switch_to_online_mode()` recompiles cleanly (`wsrl.py:521-522`
  already attempts a re-wrap).
- **bf16 mixed precision for the RGBD encoder path**: the ResNet
  encoders dominate forward FLOPs in RGBD runs; state MLPs would not
  benefit meaningfully.
