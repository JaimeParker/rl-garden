"""Compatibility wrapper for offline CQL/Cal-QL pretraining.

Prefer the generic entrypoint:

    python examples/pretrain_offline.py --algorithm cql --offline_dataset_path demos/pickcube.h5
    python examples/pretrain_offline.py --algorithm calql --offline_dataset_path demos/pickcube.h5

The legacy ``--agent cql|calql`` alias remains supported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pretrain_offline import main as _main

from rl_garden.common.cli_args import OfflinePretrainArgs


@dataclass
class Args(OfflinePretrainArgs):
    algorithm: Literal["cql", "calql"] = "cql"
    agent: Optional[Literal["cql", "calql"]] = None


def main() -> None:
    _main(Args, allowed_algorithms={"cql", "calql"})


if __name__ == "__main__":
    main()
