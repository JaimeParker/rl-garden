"""Offline pretraining entry point; supports ``--print-config``.

Same CLI as the ``rl-garden-train-offline`` console script.
"""
from rl_garden.training.cli import train_offline


def main() -> None:
    train_offline()


if __name__ == "__main__":
    main()
