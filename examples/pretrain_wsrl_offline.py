"""Compatibility wrapper for offline WSRL pretraining.

Prefer the generic entrypoint:

    python examples/pretrain_offline.py --algorithm wsrl --offline_dataset_path demos/real_robot.h5

This wrapper preserves the old default output filename ``offline_pretrained.pt``.
``wsrl-calql`` is accepted as a deprecated alias for ``wsrl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pretrain_offline import main as _main

from rl_garden.common.cli_args import OfflinePretrainArgs


@dataclass
class Args(OfflinePretrainArgs):
    algorithm: Literal["wsrl", "wsrl-calql"] = "wsrl"
    save_filename: Optional[str] = "offline_pretrained.pt"


def main() -> None:
    _main(Args, allowed_algorithms={"wsrl", "wsrl-calql"})


if __name__ == "__main__":
    main()
