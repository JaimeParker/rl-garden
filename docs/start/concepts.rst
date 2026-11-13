Core concepts
=============

Backend
-------

A backend creates environments and adapts their observations, actions and
episode boundaries to rl-garden's tensor interface. You select it with
``--env-backend``; backend-specific options use a nested namespace such as
``--gymnasium.device``. CPU simulators need explicit transfers when your
agent uses CUDA. See :doc:`/guides/gymnasium-custom-envs`.

Algorithm and phase
-------------------

An algorithm owns the learning rule. Each training phase has its own registry:
online training collects new experience, offline training learns from a dataset,
and offline-to-online training combines the two. The first positional argument
selects an algorithm within that phase. Registration connects its arguments,
builder and runner; see :doc:`/guides/extending`.

Policy and actor
----------------

The policy contains the networks used by the algorithm. Its actor maps encoded
observations to actions; critic or value networks provide learning signals.
You can supply an actor factory without replacing the training loop.
See :doc:`/tutorials/08-bring-your-own-policy`.

Observation and encoder
-----------------------

An observation describes what the environment exposes. The observation
configuration selects state, cameras and optional extra state. An encoder
turns those inputs into features consumed by the policy. Observation selection
and encoder architecture are separate choices; see :doc:`/guides/configuration`
for the supported keys and sharing rules.

Effective configuration
-----------------------

Dataclass defaults, one YAML preset and explicit CLI flags resolve into the
effective configuration, in that order. Use ``--print-config`` to inspect it
without constructing an environment. A training run writes ``config.json``
first at preflight and then after construction, when runtime facts are known.
See :doc:`/guides/configuration` for the schema and inspection options.

Repository layout
-----------------

Within ``rl_garden/``, learning rules live in ``algorithms/``, composition in ``policies/``, neural modules in ``networks/`` and feature extraction in ``encoders/``.
Use ``buffers/`` for tensor, dict, Monte-Carlo, rollout and sequence storage with explicit storage/sample devices, ``datasets/`` for offline workflows, and ``envs/`` for backends.
Shared utilities live in ``common/``, dispatch in ``training/``, optional distributed adapters in ``integrations/``, ACT/reward models in ``models/``, dynamics in ``world_models/`` and decision-time planning in ``planners/``.
At the root, ``examples/`` holds dispatchers and experiments, ``scripts/`` launchers, ``tests/`` verification, ``docs/`` documentation and ``3rd_party/`` read-only research references.
