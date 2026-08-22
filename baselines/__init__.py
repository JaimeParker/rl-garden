"""Adapters for running unmodified official JAX baseline repos (Cal-QL, wsrl,
IQL-jax) against rl-garden's canonical environments, for numeric comparison
against rl-garden's own PyTorch ports.

Not to be confused with ``rl_garden.integrations.rlinf``, which runs
rl-garden's own algorithms as workers under RLinf -- the opposite direction.

See ``.agents/runbooks/baseline-install.md`` for the full workflow and
``baselines/baselines.yaml`` for the registered baselines.
"""
