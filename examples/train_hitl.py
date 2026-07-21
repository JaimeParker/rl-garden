"""Human-in-the-loop RL entry point.

Algorithm is selected via a subcommand, mirroring the other training
dispatchers. HITL methods split actor-side rollout from learner-side updates
and communicate through the sync layer.
"""
from __future__ import annotations

from rl_garden.training.hitl import registry


def main() -> None:
    registry.run_cli()


if __name__ == "__main__":
    main()
