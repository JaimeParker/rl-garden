"""Weight initialization ported from ``3rd_party/tdmpc2/tdmpc2/common/init.py``.

The ``nn.Embedding``/``nn.ParameterList`` branches in the upstream version
exist only for multitask task embeddings, which this single-task port drops.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def zero_(params: Iterable[torch.nn.Parameter]) -> None:
    for p in params:
        p.data.fill_(0)
