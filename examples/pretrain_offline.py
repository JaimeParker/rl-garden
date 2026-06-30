"""Offline pretraining entry point; supports ``--print-config``."""
from rl_garden.training.offline import registry


def main() -> None:
    registry.run_cli()


if __name__ == "__main__":
    main()
