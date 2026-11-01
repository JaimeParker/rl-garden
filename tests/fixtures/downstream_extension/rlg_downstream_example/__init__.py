"""A downstream package that extends rl-garden by subclassing and registration.

Reference implementation for ``docs/guides/extending.md``, exercised by
``tests/test_downstream_extension.py``. It adds ``aux_split_rppo``: a
``RecurrentPPO`` whose action mean is two heads on the recurrent latent, plus
an auxiliary head trained with a binary cross-entropy loss on a per-step env
info, and two extra CLI flags. Nothing in it copies rl-garden code: every
override extends its parent method through ``super()``.

- ``actor.py``: the split actor fed with one features tensor.
- ``algorithm.py``: the algorithm subclass (policy swap, aux loss, metric).
- ``registration.py``: Args, builder, run function and ``registry.register``;
  the module the ``rlgarden.algorithms`` entry point names
  (``../pyproject.toml``).
- ``toy_env.py``: a torch-native vector env for the ``gymnasium`` backend.
"""
