Configuration and the CLI
=========================

Goal
----

You can inspect a preset, override one value, and explain where
that value came from before you create a simulator.
You also learn to recognize a configuration error early.

Prerequisites
-------------

Complete :doc:`/start/installation`.
Run the commands from the repository root so preset paths resolve.
You need the core package installed.
This chapter's inspection commands do not need ManiSkill.

Keep :doc:`/guides/configuration` available as the reference.
This chapter walks through a decision; the guide defines the schema.

Steps
-----

Select a phase and algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The executable selects the training phase.
The first positional argument selects an algorithm in that phase.
Here you use the online registry and SAC.

A name can appear in more than one phase.
Its arguments belong to the selected phase's registration.
Do not put the algorithm selector inside the preset.

The registry discovers its in-tree entries and installed extensions.
That discovery makes their argument dataclasses available for parsing.
It does not mean every optional simulator has been constructed.

Read a checked-in preset
^^^^^^^^^^^^^^^^^^^^^^^^

Start with the state-observation preset:

.. code-block:: bash

   rl-garden-train-online sac \
     --config configs/online/sac_state.yaml --print-config

Read the output before changing anything.
Find the selected algorithm under ``selection``.
Find the environment identifier under ``inputs``.
Find backend-specific settings under ``active_environment``.

The preset selects PickCube and a training budget.
These values are an experiment configuration.
They are not universal SAC defaults.

A preflight result leaves ``algorithm`` empty.
No agent constructor has run yet.
The JSON still records the selected algorithm and parsed inputs.

Override one value
^^^^^^^^^^^^^^^^^^

Keep the preset and change the discount factor on the CLI:

.. code-block:: bash

   rl-garden-train-online sac \
     --config configs/online/sac_state.yaml --gamma 0.95 \
     --print-config

Find ``gamma`` under ``inputs``.
It now has the value you supplied.
Find the same field under ``sources``.
Its source is the command line.

The precedence is defaults, then the preset, then explicit flags.
You can inspect that order in
:ref:`configuration-resolution-order`.
Use one preset per invocation.
There is no YAML include or inheritance layer.

A small override is easier to review than a copied preset.
When an experiment settles, put its stable settings in a named YAML file.
Keep the selected algorithm on the command line.

Ask why a value won
^^^^^^^^^^^^^^^^^^^

Explain the same field instead of printing the entire configuration:

.. code-block:: bash

   rl-garden-train-online sac \
     --config configs/online/sac_state.yaml --gamma 0.95 \
     --explain-param gamma

The response names the field, value, type and source.
For this invocation, the value is 0.95 and the source kind is CLI.
The explanation is about this resolved invocation.
It is not a static list of every consumer of the parameter.

Inspection actions are mutually exclusive.
Do not combine the explanation action with the print action.
For the complete inspection contract, see
:ref:`configuration-inspect-before-training`.

Recognize a strict-mapping error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you edit a preset and transpose two letters:

.. code-block:: yaml

   gamam: 0.95

The intended field is ``gamma``.
The strict mapper rejects the unknown field with ``ConfigError``.
It does not quietly ignore the misspelling or start training.

Correct the key in the preset, then inspect the result again.
The tutorial command tests exercise this error with a temporary file.
You do not need to launch an environment to catch it.

The same validation boundary rejects incompatible field types.
An option for an inactive backend can also fail preflight.
That failure prevents a plausible-looking flag from doing nothing.

Separate inspection from construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Printing the configuration performs parsing and static validation.
It does not create a run directory or a replay buffer.
It does not demonstrate that an optional simulator is installed.

The dry-run action goes further.
It constructs the environment and agent and records runtime facts.
It therefore needs the backend dependencies and appropriate hardware.
Read its scope in the configuration guide before using it.

Normal training writes ``config.json`` under the selected run directory.
It first writes a preflight snapshot.
After construction it replaces that snapshot with materialized facts.
Use that saved file when you investigate a completed run.

Read the result as evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the backend configuration with the algorithm inputs.
A CPU environment and a CUDA replay buffer are different choices.
A preset name alone does not describe either choice completely.

The ``sources`` mapping is intentionally sparse.
A missing source entry means a dataclass default won.
The ``derived`` mapping records values changed during resolution.
Do not assume every displayed value was typed on the command line.

See :ref:`configuration-effectiveconfig-v3`
for the snapshot boundary and persisted fields.
Link experiments to their saved configuration when comparing results.

What you ran
------------

.. note::

   You ran the preset inspection, the one-field override, and the explanation
   commands shown above, in order.
   They produce JSON on standard output and create no training artifacts.
   There are no loss or return metrics yet.
   A later training invocation writes
   ``runs/<run_name>/config.json`` and, when enabled,
   ``runs/<run_name>/checkpoints/final.pt``.

Before continuing, identify the selected phase and backend in your output.
Then locate the overridden field and its CLI source.
Those checks are the result of this chapter.

Next
----

Continue with :doc:`02-online-rl` to turn the inspected configuration
into a short training run.
Return to this chapter whenever a preset and a command line disagree.
