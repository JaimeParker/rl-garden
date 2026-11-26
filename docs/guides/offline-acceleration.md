# Compile and Fast-Math Options

This page lists the throughput options a run can set today: `torch.compile`
for the CQL family and FlashSAC, TF32/cuDNN fast math, and deferred metric
synchronisation. None of them changes what the algorithm computes beyond
floating-point rounding.

## `use_compile`

`--use-compile` / `--no-use-compile` wraps the CQL-family critic loss, actor
loss and target-Q computation in `torch.compile`. `--compile-mode` selects the
`torch.compile` mode (`default`, `reduce-overhead` or `max-autotune`, default
`default`). The flag exists only on these entrypoints:

| Entrypoint | Algorithms | Default |
|---|---|---|
| `examples/pretrain_offline.py` | `cql`, `calql`, `wsrl` | `True` |
| `examples/train_off2on.py` | `calql`, `wsrl` (offline and online phases) | `False` |
| `examples/train_online.py` | `flash_sac` | `False` |

Every other algorithm, including `iql`, `edac`, `sac` and `rlpd`, has no
compile flag; passing one is a parse error. Check the value a run will use
with `--print-config`:

```bash
python examples/pretrain_offline.py calql \
  --env-backend gymnasium --env-id Pendulum-v1 \
  --offline_dataset /path/to/dataset.h5 --print-config
```

Compile traces the graph on the first steps, so very short runs (smoke tests)
may not amortise it. Compiled kernels are fused and reorder floating-point
operations, so results are close to, not bit-identical with, the eager path.

Turn it off with `--no-use-compile` when:

- a test or comparison needs seed-stable, exact loss values;
- the run is short enough that compile warmup dominates;
- you are debugging numerics and want eager stack traces.

Turn it on for long `train_off2on.py` or `train_online.py flash_sac` runs with
`--use-compile`, and try `--compile-mode reduce-overhead` only if you can
tolerate its stricter static-shape requirements.

## Fast math

`rl_garden.common.enable_fast_math()` sets `torch.backends.cudnn.benchmark =
True` and `torch.set_float32_matmul_precision("high")` (TF32 matmul on Ampere
and newer GPUs). There is no CLI flag for it. The offline and off2on runners
and checkpoint evaluation call it right after seeding; the online runners do
not. If a job needs strict fp32 matmul, reverse it in your own code after
setup:

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
```

## Deferred metric sync

Reading a metric with `.item()` forces a GPU-to-CPU sync. Update loops therefore
pass `compute_info=True` to `train()` only on logging steps, and the
off-policy `train()` implementations return an empty dict otherwise.
This applies to `OffPolicyAlgorithm.learn()` for online runs and to
`run_offline_pretraining()` for offline pretraining, including the offline
phase of `train_off2on.py`. Set `--log-freq` larger to sync less often; custom
algorithms should honour the `compute_info` argument and keep metric
reductions out of the hot path.
