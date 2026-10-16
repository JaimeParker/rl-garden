"""Shared pytest configuration: filename-based lane markers and global-state isolation."""
from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path

# torch's _inductor.async_compile pre-warms a multiprocess compile pool at import time.
# Only a few torch.compile tests need it (the pool is then created on demand), and spawned
# DDP children inherit this variable so they stop launching their own compile workers.
os.environ.setdefault("TORCH_WARM_POOL", "0")

import pytest
import torch

# Filename patterns -> marker. Explicit pytest.mark.* inside a file is kept; these only add.
SMOKE_FILE_PATTERNS = ("test_*_smoke*.py",)
SIM_FILE_KEYWORDS = (
    "maniskill",
    "dm_control",
    "mujoco",
    "ogbench",
    "robomimic",
    "d4rl",
    "metaworld",
    "robotwin",
    "isaaclab",
    "gym_env",
    "vector_env",
)
# Files that skip without CUDA (module/function level skipif or runtime pytest.skip).
GPU_FILES = frozenset(
    {
        "test_bcq_core.py",
        "test_edac_core.py",
        "test_qgf_core.py",
        "test_plain_conv_encoder.py",
        "test_plas_core.py",
        "test_policy_kwargs.py",
        "test_spot_core.py",
        "test_qam_core.py",
        "test_rebrac_core.py",
        "test_sac_vmap_parity.py",
        "test_sac_smoke.py",
        "test_off_policy_episode_mode.py",
        "test_sac_core.py",
        "test_gail.py",
        "test_tdmpc2_parity.py",
    }
)


def _lane_markers_for(filename: str) -> list[str]:
    markers = []
    if any(fnmatch.fnmatch(filename, pattern) for pattern in SMOKE_FILE_PATTERNS):
        markers.append("smoke")
    if any(keyword in filename for keyword in SIM_FILE_KEYWORDS):
        markers.append("sim")
    if filename in GPU_FILES:
        markers.append("gpu")
    return markers


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in _lane_markers_for(Path(str(item.fspath)).name):
            item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture(autouse=True)
def restore_torch_global_flags():
    precision = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.fixture(autouse=True)
def restore_sys_path():
    saved = list(sys.path)
    yield
    sys.path[:] = saved


@pytest.fixture(scope="session", autouse=True)
def single_thread_under_xdist():
    if "PYTEST_XDIST_WORKER" in os.environ:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    yield
