"""Offline-to-online training entry point; supports ``--print-config``.

Same CLI as the ``rl-garden-train-off2on`` console script.
"""
from rl_garden.training.cli import train_off2on


def main() -> None:
    train_off2on()


if __name__ == "__main__":
    main()
