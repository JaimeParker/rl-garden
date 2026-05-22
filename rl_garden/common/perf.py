"""Lightweight CUDA performance flags applied at training entry."""
from __future__ import annotations

import torch


def enable_fast_math() -> None:
    """Enable TF32 matmul and cuDNN benchmark mode.

    Trades tiny numerical precision for throughput. Safe defaults for
    training paths; reverse manually if a job needs strict fp32 matmul.
    """
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
