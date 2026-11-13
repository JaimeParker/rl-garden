World models and planning
=========================

When to use
-----------

Use this family when the learning method includes an explicit learned dynamics model.
TD-MPC2 plans through an implicit world model; DreamerV3 learns through RSSM imagination.
PWM learns on a frozen multitask model, while NCRL fine-tunes a pretrained RSSM.
SuperMimic differentiates imagined returns through an explicit-state world model.

What the policy must provide
----------------------------

These methods own world models, actors or planners with method-specific interfaces.
Do not substitute a PPOActor or SACActor merely because the model has an action head.
Use :doc:`/tutorials/08-bring-your-own-policy` to understand that narrower actor boundary.
The :ref:`extending-ppo-seams` and :ref:`extending-sac-seams` cover those families only.

The model-based deep dives below describe the components each method actually builds.
PWMPolicy and SuperMimicPolicy are outside the BasePolicy extractor contract.
Their inputs and optimizer ownership follow the world-model design in their modules.
Start from the selected algorithm rather than transplanting a SAC policy subclass.

Minimal command and construction
--------------------------------

The TD-MPC2 command inspects a single Gymnasium environment and a single evaluation environment.
This port restricts online TD-MPC2 to that single-environment setting.
It also rejects episodic mode; the example leaves that mode disabled.
The small CPU construction reduces planner and network sizes without performing planning or training.

Offline multitask TD-MPC2 has its own dataset and does not run live CEM planning.
RSSM pretraining uses image sequences and does not train its actor or critic.
A generic H5 transition file is not a replacement for those methods' specialized datasets.
Read the relevant deep dive before preparing a large pretraining run.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-online tdmpc2 --env-backend gymnasium --env-id Pendulum-v1 \
     --num-envs 1 --num-eval-envs 1 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start tdmpc2
   :end-before: # docs: end tdmpc2

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``online/tdmpc2`` by the command above.
They are not constructor defaults or recommended settings for every task.
Inspect again after selecting a preset; it can override these values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--horizon``
     - ``3``
     - Planning horizon.
   * - ``--num-samples``
     - ``512``
     - Candidate planner trajectories.
   * - ``--num-elites``
     - ``64``
     - Elite trajectories retained by the planner.
   * - ``--num-pi-trajs``
     - ``24``
     - Policy-proposed trajectories.
   * - ``--iterations``
     - ``6``
     - Planner refinement iterations.
   * - ``--latent-dim``
     - ``512``
     - World-model latent width.
   * - ``--mlp-dim``
     - ``512``
     - World-model MLP width.
   * - ``--episodic``
     - ``False``
     - Termination modeling mode; unsupported here when enabled.

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
   * - ``dreamer_v3`` (online)
     - RSSM learning and imagination-based actor-critic.
     - DreamerV3
   * - ``ncrl`` (online)
     - Pretrained RSSM fine-tuning with rehearsal and guidance.
     - NCRL
   * - ``supermimic`` (online)
     - Actor learning through an explicit-state world model.
     - SuperMimic
   * - ``tdmpc2`` (online)
     - Implicit world model with CEM/MPPI action planning.
     - TDMPC2
   * - ``pwm`` (offline)
     - Actor-critic learning on a frozen multitask world model.
     - PWM
   * - ``rssm_pretrain`` (offline)
     - Reward-free RSSM pretraining from image sequences.
     - RSSMPretrain
   * - ``tdmpc2_multitask`` (offline)
     - Offline multitask world-model training without live planning.
     - TDMPC2 (multitask)

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/guides/dreamer-v3`
  — RSSM configuration and imagination learning.

- :doc:`/guides/ncrl`
  — Reward-free pretraining, retrieval and guidance.

- :doc:`/guides/supermimic`
  — Explicit-state model components and imagined-return updates.

- :doc:`/guides/extending`
  — General registration and checkpoint extension contract.
