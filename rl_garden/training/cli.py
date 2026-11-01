"""Console-script entry points (``[project.scripts]`` in ``pyproject.toml``).

``rl-garden-train-online``, ``rl-garden-train-offline`` and
``rl-garden-train-off2on`` dispatch through the phase registry, which also
loads algorithms registered by installed packages through the
``rlgarden.algorithms`` entry-point group; ``rl-garden-eval`` evaluates a
checkpoint. The ``examples/`` scripts of a source checkout call these same
functions. Imports stay inside the functions so ``--help`` loads one phase only.
"""

from __future__ import annotations


def train_online() -> None:
    from rl_garden.training.online import registry

    registry.run_cli()


def train_offline() -> None:
    from rl_garden.training.offline import registry

    registry.run_cli()


def train_off2on() -> None:
    from rl_garden.training.off2on import registry

    registry.run_cli()


def eval_checkpoint() -> None:
    from rl_garden.training.evaluation import main

    main()
