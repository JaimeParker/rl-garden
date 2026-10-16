# Testing Runbook

Run tests through `scripts/test.sh <lane> [extra pytest args]`. Plain
`pytest tests` still runs everything: nothing is excluded implicitly.

## Lanes

| lane | selection | execution |
|---|---|---|
| `fast` | `not gpu and not sim and not slow` | xdist, `--dist loadfile`, `PYTEST_WORKERS` workers (default 8) |
| `sim` | `sim and not gpu and not slow` | serial |
| `gpu` | `gpu and not slow` | serial, one card |
| `slow` | `slow` | xdist, `--dist loadgroup`, `PYTEST_SLOW_WORKERS` workers (default 4) |
| `all` | fast, sim, gpu, slow in turn | a failing lane does not stop later lanes; prints per-lane summaries, exits non-zero if any failed |

Environment knobs: `PYTEST_WORKERS`, `PYTEST_SLOW_WORKERS` (slow lane, default 4), `PYTEST_TIMEOUT` (default 600 s),
`PYTEST_MAXFAIL` (default 20), `PYTEST_CACHE_DIR`, `PYTEST_JUNIT_DIR` (writes
`<lane>.xml`). The script does not set `CUDA_VISIBLE_DEVICES` or `MUJOCO_GL`;
pass them from the caller.

## Markers

Registered in `pyproject.toml` with `--strict-markers`: `gpu`, `sim`, `smoke`, `slow`.
`tests/conftest.py` adds them automatically from the file name; explicit
`pytest.mark.*` inside a file is kept and the auto rules only add.

| marker | auto rule |
|---|---|
| `smoke` | `test_*_smoke*.py` |
| `sim` | name contains `maniskill`, `dm_control`, `mujoco`, `ogbench`, `robomimic`, `d4rl`, `metaworld`, `robotwin`, `isaaclab`, `gym_env`, `vector_env` |
| `gpu` | files listed in `GPU_FILES` in `tests/conftest.py` (those that skip without CUDA) |
| `slow` | explicit only |

Tests that bind a fixed port or spawn several processes (gloo, NCCL) must share one
`xdist_group`, for example `pytest.mark.xdist_group("ddp")`, so the slow lane runs them serially on one worker.
A new test file that needs CUDA must be added to `GPU_FILES`. Mark `slow`
explicitly (module `pytestmark` or a decorator) for multi-process spawn tests,
`torch.compile` tests, and real training loops that take more than a few seconds.
`conftest.py` also restores TF32 / matmul precision and `sys.path` after each
test, and pins torch threads to 1 inside xdist workers.

## Remote runs

- Run on a remote server (see the remote training SOP), never locally.
- Point `PYTEST_CACHE_DIR` outside the Mutagen-synced tree (for example
  `/workspace/_outputs/...`); a root-owned `.pytest_cache` in the synced tree stalls sync.
- Keep `PYTEST_WORKERS` at 8 or fewer on shared machines. The slow lane uses `loadgroup`.
  `tests/test_ddp.py` and `tests/test_sac_ddp.py` carry `xdist_group("ddp")`, so they run
  serially on one worker. Their spawned ranks rendezvous through a `file://` store in a
  per-test temp dir (no TCP port), and pause briefly before `destroy_process_group` to
  avoid a torch gloo teardown deadlock.
  Use `PYTEST_SLOW_WORKERS=1` to debug serially.
- GPU and sim lanes are serial and use one card at a time; choose a free card
  with `CUDA_VISIBLE_DEVICES`. Set `MUJOCO_GL=egl` for dm_control/mujoco.

## Useful variations

- Re-run only failures: `scripts/test.sh fast --lf` (needs a stable `PYTEST_CACHE_DIR`).
- Order dependence check (occasional, needs `pytest-randomly`):
  `scripts/test.sh fast -p randomly --randomly-seed=1`. Compare a failing file
  against a single-file run to tell ordering from real failures.
- One file: `python -m pytest tests/test_x.py -q -o cache_dir=/tmp/rl_garden_pytest_cache`.
