"""Offline-to-online training entry point; supports ``--print-config``."""
from rl_garden.training.off2on import registry


def main() -> None:
    registry.run_cli()


if __name__ == "__main__":
    main()
