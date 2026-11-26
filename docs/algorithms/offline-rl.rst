Offline reinforcement learning
==============================

When to use
-----------

Use this family to improve a policy from a fixed dataset with rewards.
IQL fits an expectile value function and an advantage-weighted actor.
CQL and Cal-QL add conservative value learning; other entries use support constraints,
ensembles, generative actors or offline skill discovery.

What the policy must provide
----------------------------

There is no single interchangeable actor contract for every offline algorithm.
IQL's policy contains an actor, critics and a value network.
Flow, diffusion and skill methods own their specialized policy construction.
Check the selected module before replacing one of those components.

For CQL/Cal-QL and OfflineSAC descendants, read :ref:`extending-sac-seams`.
Their seam rows name policy_class and _policy_kwargs explicitly.
Use :doc:`/tutorials/08-bring-your-own-policy` for the SAC actor capability checks.
CQL importance sampling needs actual action densities, not a flow path likelihood.

Minimal command and construction
--------------------------------

The command below inspects IQL on an H5 placeholder; it does not read the file.
Prepare the data with :doc:`/tutorials/04-offline-pretraining` before running updates.
That chapter documents flat and Dict observations, terminal flags and action bounds.
The CPU construction uses OfflineEnvSpec and does not require an environment rollout.

Keep the offline and off2on entries distinct when a name exists in both registries.
The variants below list only the offline entries assigned to this family.
Offline WSRL checkpoint preparation belongs to :doc:`offline-to-online`.
Offline world-model pretraining belongs to :doc:`model-based`.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-offline iql --env-backend gymnasium --env-id Pendulum-v1 \
     --offline-dataset demos/example.h5 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start iql
   :end-before: # docs: end iql

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``offline/iql`` by the command above.
They are not constructor defaults or recommended settings for every task.
Inspect again after selecting a preset; it can override these values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--actor-lr``
     - ``0.0003``
     - Actor optimizer step size.
   * - ``--critic-value-lr``
     - ``0.0003``
     - Critic and value optimizer step size.
   * - ``--expectile``
     - ``0.7``
     - Asymmetric value-regression weight.
   * - ``--temperature``
     - ``3.0``
     - Advantage exponential weighting factor.
   * - ``--adv-clip-max``
     - ``100.0``
     - Maximum actor advantage weight.
   * - ``--gamma``
     - ``0.99``
     - Discount in the critic target.

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
   * - ``awac`` (offline)
     - Advantage-weighted actor-critic.
     - AWAC
   * - ``bcq`` (offline)
     - Batch-constrained action selection.
     - BCQ
   * - ``bppo`` (offline)
     - Offline critic fitting followed by behavior-proximal updates.
     - BPPO
   * - ``calql`` (offline)
     - Calibrated conservative Q-learning.
     - CalQL
   * - ``cql`` (offline)
     - Conservative Q regularization on offline data.
     - CQL
   * - ``edac`` (offline)
     - Offline critic ensemble with gradient-diversity penalty.
     - EDAC
   * - ``fac`` (offline)
     - Flow actor-critic with a staged training schedule.
     - FAC
   * - ``fino`` (offline)
     - FQL with scheduled noise in the BC-flow branch.
     - FINO
   * - ``floq`` (offline)
     - FQL actor with a flow-matching critic.
     - FloQ
   * - ``fql`` (offline)
     - Flow BC teacher and one-step Q-optimized student.
     - FQL
   * - ``hilp`` (offline)
     - Hilbert representations and unsupervised skill policies.
     - HILP
   * - ``idql`` (offline)
     - IQL-style value fitting with a diffusion actor.
     - IDQL
   * - ``iql`` (offline)
     - Expectile value regression and advantage-weighted actor fitting.
     - IQL
   * - ``opal`` (offline)
     - Skill VAE over action chunks for a downstream decoder.
     - OPAL
   * - ``plas`` (offline)
     - Offline policy optimization in a learned latent action space.
     - PLAS
   * - ``qam`` (offline)
     - Flow actor trained using adjoint Q-gradient targets; with
       ``--horizon-length`` above 1, evaluation executes each chunk one
       action per environment step.
     - QAM
   * - ``qgf`` (offline)
     - BC flow actor with Q-gradient guidance at inference; chunks are
       evaluated as for QAM.
     - QGF
   * - ``rebrac`` (offline)
     - TD3-BC with a cloning penalty on the critic bootstrap.
     - ReBRAC
   * - ``spot`` (offline)
     - VAE support constraint in place of the TD3-BC cloning term.
     - SPOT
   * - ``td3_bc`` (offline)
     - TD3 with a normalized cloning term in the actor objective.
     - TD3BC
   * - ``unio4`` (offline)
     - BPPO-style actor ensemble sharing a critic.
     - UniO4
   * - ``unio4_ope`` (offline)
     - Uni-O4 with dynamics-model evaluation gating.
     - UniO4OPE
   * - ``value_flows`` (offline)
     - FQL-style actor with a distributional flow critic.
     - ValueFlows

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/design/iql-overview`
  — IQL losses and the offline/online split.

- :doc:`/guides/offline-acceleration`
  — CQL-family compilation and throughput considerations.

- :doc:`/guides/d4rl-legacy-expansion`
  — Dataset and environment assumptions for legacy benchmarks.

- :doc:`/tutorials/04-offline-pretraining`
  — Dataset preparation, BC/IQL training and evaluation.
