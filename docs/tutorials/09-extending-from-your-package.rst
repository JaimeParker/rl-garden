Extending from your package
===============================

Goal
--------

You discover a downstream algorithm through an installed package entry point.
You follow a tested fixture from arguments to policy, rollout data, loss, and checkpoint.
You keep the algorithm extension outside the rl-garden source tree.

Prerequisites
-----------------

Read :doc:`/guides/extending`, which remains the extension contract.
Use a source checkout so the reference fixture files are present.
The fixture lives under ``tests/fixtures/downstream_extension/``.
It is a runnable reference package, not part of rl-garden's installed algorithm catalog.

Understand the configuration flow from :doc:`01-configuration-and-cli`.
Understand checkpoint reconstruction from :doc:`06-checkpoints-and-evaluation`.
You do not need a robotics simulator for the fixture's tests.
Its toy environment and training test run on CPU.

Steps
---------

Expose a registration module through packaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fixture declares the ``rlgarden.algorithms`` entry-point group.
The entry-point value names an importable registration module.
Importing that module registers the algorithm on the appropriate phase registry.
The distribution name, entry-point name, and algorithm selector have different roles.

.. literalinclude:: ../../tests/fixtures/downstream_extension/pyproject.toml
   :language: toml

Here the installed distribution is ``rlg-downstream-example``.
The registration module is ``rlg_downstream_example.registration``.
The CLI selector registered by that module is ``aux_split_rppo``.
Use the selector when choosing an algorithm at the terminal.

The fixture's Python package must be importable in the command's interpreter.
Installing it in a different virtual environment does not affect discovery here.
The reference package declares its dependency on rl-garden in packaging metadata.
Keep a compatible version of that dependency when distributing your own extension.

Install and inspect the fixture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the local package into the same environment as rl-garden:

.. code-block:: bash

   python -m pip install --no-deps --no-build-isolation -e tests/fixtures/downstream_extension

This local editable install assumes the core package and build tools are already present.
It does not download simulator dependencies.
Start a new CLI process after installation so discovery sees the installed entry point.
The tutorial tests use an isolated import equivalent to avoid modifying package state.

List the registered algorithms:

.. code-block:: bash

   rl-garden-train-online --help

The listing now includes ``aux_split_rppo``.
Inspect the extension's flags:

.. code-block:: bash

   rl-garden-train-online aux_split_rppo --help

The algorithm-specific help includes the fixture's auxiliary weight and hidden width.
The parent algorithm's arguments remain available too.
Now inspect one override without constructing an environment:

.. code-block:: bash

   rl-garden-train-online aux_split_rppo --aux-weight 0.2 --print-config

This confirms argument ownership and registration.
It does not run the toy environment or update the policy.
Use the fixture's runtime test for those guarantees.
Do not treat successful help output as a training smoke.

Follow argument ownership and construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The registration module extends the parent's argument dataclass.
It forwards new values through a builder that reuses the parent builder.
The run function keeps the parent's lifecycle and replaces agent construction.
This lets the extension reuse logging, environment setup, and checkpoint handling.

.. literalinclude:: ../../tests/fixtures/downstream_extension/rlg_downstream_example/registration.py
   :language: python

The registry entry provides both the algorithm class and the builder.
Those are needed beyond selecting a run function.
Checkpoint evaluation uses the registration to reconstruct the right class.
Do not register an unrelated class under the same selector to load old checkpoints.

The fixture does not copy a training loop into its package.
It introduces only the fields it owns and passes them to its subclass.
Keep that ownership visible when adding more parameters.
See :doc:`/guides/extending` for the required registry and builder contracts.

Follow the policy composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The policy subclasses the recurrent PPO policy.
It changes actor construction and adds an auxiliary head as a policy submodule.
The actor splits action components across two heads that read the same latent features.
The helper preserves the parent actor's mean computation after adapting its input.

.. literalinclude:: ../../tests/fixtures/downstream_extension/rlg_downstream_example/actor.py
   :language: python

The auxiliary head belongs to the policy module tree.
That keeps its parameters in the policy optimizer and checkpoint.
A detached helper network outside that tree would need separate ownership decisions.
Follow the existing seam before introducing a new training lifecycle.

The policy declares an extra rollout field for the auxiliary target.
It also requests the recurrent latent in evaluation outputs.
Together those provide the target and prediction inputs for the auxiliary objective.

.. literalinclude:: ../../tests/fixtures/downstream_extension/rlg_downstream_example/policy.py
   :language: python

The toy environment supplies the corresponding information field.
A policy declaration alone cannot manufacture the target at rollout time.
Keep field shape and dtype consistent between producer and consumer.
The existing fixture test covers the complete path.

Follow the algorithm's additional objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithm subclasses RecurrentPPO and selects the fixture policy.
Its loss override extends the parent's auxiliary loss hook.
It reduces binary cross entropy over the selected decision rows when a mask exists.
Its training method delegates the optimizer work to the parent.

.. literalinclude:: ../../tests/fixtures/downstream_extension/rlg_downstream_example/algorithm.py
   :language: python

The extension records the new constructor fields in checkpoint metadata.
Its logging override moves the auxiliary logit statistic to rollout-style metrics.
Other update metrics continue through the inherited implementation.
This makes the extra objective inspectable without replacing the parent's accounting.

The fixture rejects a competing policy-graph configuration explicitly.
That prevents two independent mechanisms from claiming policy construction.
Check the documented extension seams before combining customization mechanisms.
A configuration error is more useful than silently ignoring one implementation.

Separate algorithm, backend, and actor registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An algorithm entry point registers phase-specific training behavior.
A backend package uses the separate ``rlgarden.env_backends`` group.
A custom Gymnasium task often needs only module import, as in
:doc:`07-bring-your-own-environment`.
Do not add a backend merely to make a standard Gymnasium task discoverable.

An external actor is an importable component passed through supported policy seams.
It does not need to masquerade as a new algorithm registration.
The tutorial :doc:`08-bring-your-own-policy` covers those component contracts.
Use :doc:`/guides/extending` now for the authoritative supported methods and ownership rules.

Check discovery and execution separately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tutorial command test imports this package using the downstream test's isolation pattern.
It checks the top-level listing, algorithm help, and printed override.
The downstream test separately simulates entry-point discovery from packaging metadata.
It also trains, saves a checkpoint, and evaluates with automatic algorithm selection.

Those tests exercise a tiny CPU configuration with the fixture's toy environment.
They do not establish performance on your task.
When you change your own extension, keep both discovery and runtime coverage.
A valid dataclass can still describe a policy that fails during an update.

What you ran
----------------

.. note::

   You installed the local fixture and ran the three inspection commands above.
   They list ``aux_split_rppo`` and show the auxiliary-weight override as 0.2.
   Help and printed configuration do not create a run directory or training metrics.
   The downstream runtime test additionally checks ``aux/bce`` and ``aux_logit_mean``
   in its update results and verifies checkpoint evaluation.

The runtime fixture test uses a temporary output root.
Its saved artifacts have this layout:

.. code-block:: text

   <temporary-root>/run/config.json
   <temporary-root>/run/checkpoints/final.pt

This is the runtime test's output, not a side effect of the help commands.
Keep your own package installed when evaluating one of its checkpoints.
The saved algorithm identity must still resolve to a compatible implementation.

Next
--------

Return to :doc:`/guides/extending` when choosing a supported customization seam.
Use :doc:`06-checkpoints-and-evaluation` to verify that your extension survives saving,
loading, and evaluation before increasing the training budget.
