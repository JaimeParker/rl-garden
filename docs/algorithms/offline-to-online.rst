Offline-to-online learning
==========================

When to use
-----------

Use this family when a learned policy or prior dataset must transfer into live interaction.
WSRL warms up a Cal-QL checkpoint before continuing as online SAC.
Cal-QL's own continuation retains mixed offline data instead.
JSRL, ExPLORe and SUPE supply guidance or prior-data mechanisms through the online registry.

What the policy must provide
----------------------------

A checkpoint handoff must rebuild a compatible policy before loading its weights.
For the Cal-QL/WSRL actor path, use :doc:`/tutorials/08-bring-your-own-policy`.
The :ref:`extending-sac-seams` rows describe CQLCore construction and capability checks.
Pass the same importable factory when you continue a custom-actor run.

Do not apply that factory interface to every continuation algorithm by analogy.
FQL-derived and skill-based methods own different policies.
Read :doc:`/tutorials/05-offline-to-online` for what carries across phases.
Use :doc:`/tutorials/06-checkpoints-and-evaluation` for loading and evaluation.

Minimal command and construction
--------------------------------

The minimal command inspects the WSRL off2on selection without opening a dataset.
It is not a complete checkpoint-training recipe; follow chapter 05 for that sequence.
The Python example constructs WSRL against OfflineEnvSpec for its offline preparation stage.
It does not switch to online learning or create a live environment for that stage.

WSRL's warmup freezes updates and its continuation discards the offline replay.
Its default online continuation removes CQL conservatism.
Use the Cal-QL continuation when you want the retained-data design described in its module.
Read the phase-specific values rather than carrying every offline option over unchanged.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-off2on wsrl --env-backend gymnasium --env-id Pendulum-v1 \
     --offline-dataset demos/example.h5 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start wsrl
   :end-before: # docs: end wsrl

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``off2on/wsrl`` by the command above.
They are not constructor defaults or recommended settings for every task.
Inspect again after selecting a preset; it can override these values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--warmup-steps``
     - ``5000``
     - Steps collected without updates during warmup.
   * - ``--online-replay-mode``
     - ``empty``
     - Replay mode for the online transition.
   * - ``--offline-data-ratio``
     - ``0.0``
     - Fraction drawn from the offline buffer in mixed replay.
   * - ``--online-cql-alpha``
     - ``0.0``
     - Conservative weight selected for the online phase.
   * - ``--online-use-cql-loss``
     - ``False``
     - Whether online updates retain the CQL penalty.
   * - ``--utd``
     - ``0.25``
     - Update-to-data ratio after collection.

Use :doc:`/guides/configuration` for configuration precedence and observation options.
Use :doc:`/guides/checkpoint` for save/load controls rather than inferring them from this table.

Variants
--------

Each row names a phase and its registry entry. Reference keys are the Algorithm labels
in :doc:`/REFERENCES`; an em dash means no matching module row exists there.

.. list-table::
   :header-rows: 1

   * - Registry name (phase)
     - Difference
     - Reference key
   * - ``explore`` (online)
     - RLPD with optimistic relabeling of unlabeled prior data; relabeling
       and RND label single transitions, so they reject ``--nstep`` above 1.
     - ExPLORe
   * - ``jsrl`` (online)
     - Frozen guide policy acts before the online SAC policy.
     - JSRL
   * - ``supe`` (online)
     - ExPLORe over OPAL skills with a frozen low-level decoder.
     - SUPE
   * - ``wsrl`` (offline)
     - Offline preparation of a WSRL-compatible Cal-QL checkpoint.
     - WSRL
   * - ``acfql`` (off2on)
     - Action-chunked FQL continuation; rollout and evaluation execute each
       chunk one action per environment step, and a chunk that runs past its
       episode's end has zero weight in the critic loss.
     - ACFQL
   * - ``awac`` (off2on)
     - Online continuation: advantage-weighted actor-critic.
     - Off2OnAWAC
   * - ``calql`` (off2on)
     - Online continuation: calibrated conservative Q-learning.
     - Off2OnCalQL
   * - ``fac`` (off2on)
     - Online continuation: flow actor-critic with a staged training schedule.
     - FAC
   * - ``fino`` (off2on)
     - Online continuation: FQL with scheduled noise in the BC-flow branch.
     - FINO
   * - ``floq`` (off2on)
     - Online continuation: FQL actor with a flow-matching critic.
     - FloQ
   * - ``fql`` (off2on)
     - Online continuation: flow BC teacher and one-step Q-optimized student.
     - Off2OnFQL
   * - ``iql`` (off2on)
     - Online continuation: expectile value regression and advantage-weighted actor fitting.
     - Off2OnIQL
   * - ``so2`` (off2on)
     - Ensemble SAC with Bellman-target action smoothing.
     - Off2OnSO2
   * - ``spot`` (off2on)
     - Online continuation: VAE support constraint in place of the TD3-BC cloning term.
     - Off2OnSPOT
   * - ``value_flows`` (off2on)
     - Online continuation: FQL-style actor with a distributional flow critic.
     - ValueFlows
   * - ``wsrl`` (off2on)
     - Cal-QL checkpoint, update-free warmup, then online SAC.
     - WSRL

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/tutorials/05-offline-to-online`
  — Checkpoint handoff and runnable continuation pairs.

- :doc:`/design/wsrl-overview`
  — WSRL phases and replay behavior.

- :doc:`/guides/wsrl-reproduction`
  — Existing dataset-to-fine-tuning workflow.

- :doc:`/guides/d4rl-legacy-expansion`
  — Assumptions behind the legacy task presets.
