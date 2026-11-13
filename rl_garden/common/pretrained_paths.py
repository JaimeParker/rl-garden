"""Where externally-sourced pretrained weights (ResNet backbones, ACT base
policies) are looked up by name.

These are not checkpoints produced by rl-garden's own training runs (those go
under ``--checkpoint_dir``). Resolution order:

1. ``$RL_GARDEN_PRETRAINED_DIR`` if set (one flat directory for every kind);
2. ``$XDG_CACHE_HOME/rl_garden/pretrained/<kind>/``, defaulting to
   ``~/.cache/rl_garden/pretrained/<kind>/``.

The repository tree is never searched, so a source checkout holds no weights.
"""

from __future__ import annotations

import os
from pathlib import Path

PRETRAINED_DIR_ENV = "RL_GARDEN_PRETRAINED_DIR"


def default_pretrained_root() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "rl_garden" / "pretrained"


def pretrained_dir(kind: str) -> Path:
    """Directory searched for ``<kind>`` weights (``"resnet"`` or ``"act"``)."""
    env = os.environ.get(PRETRAINED_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return default_pretrained_root() / kind
