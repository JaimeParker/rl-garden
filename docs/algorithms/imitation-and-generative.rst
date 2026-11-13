Imitation and generative policies
=================================

When to use
-----------

Use BC when your dataset supplies actions you want the policy to imitate.
Use diffusion or flow variants for their corresponding generative imitation objectives.
Choose DAgger for expert-labeled aggregation, or policy distillation for a frozen teacher.
GAIL combines a PPO generator with an imitation discriminator.
Online means live rollout, not necessarily a reward objective: DAgger and policy distillation learn supervised actions from encountered states.
Policy distillation trains no critic and uses no reward-based objective.

What the policy must provide
----------------------------

BC fits action likelihood with an actor and no learned Q-function.
Its actor-role encoder handles state or Dict observations.
Diffusion, flow and consistency methods build specialized policies and training objectives.
A generative policy name alone does not imply support for the SAC actor factory.

Use :doc:`/tutorials/08-bring-your-own-policy` to distinguish an actor head from its extractor.
The :ref:`extending-ppo-seams` apply to GAIL's PPO generator path.
The :ref:`extending-sac-seams` describe a different, Q-learning policy boundary.
Use the selected algorithm's builder when the method owns its full policy.

Minimal command and construction
--------------------------------

The command below inspects Gaussian BC on a placeholder H5 dataset.
For a tiny CPU dataset and actual updates, follow :doc:`/tutorials/04-offline-pretraining`.
The construction example supplies observation and action spaces through OfflineEnvSpec.
No expert, teacher checkpoint or simulator is needed to construct this BC example.

Diffusion BC produces the EMA weights used by the DPPO fine-tuning path.
Consistency distillation instead uses a frozen diffusion teacher and a student.
Online diffusion consistency distillation is listed here by its imitation mechanism,
even though its registry phase is online and its update builds on DPPO.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-offline bc --env-backend gymnasium --env-id Pendulum-v1 \
     --offline-dataset demos/example.h5 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start bc
   :end-before: # docs: end bc

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``offline/bc`` by the command above.
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
   * - ``--batch-size``
     - ``256``
     - Dataset samples per update batch.
   * - ``--tanh-squash``
     - ``True``
     - Whether the Gaussian action distribution is squashed.
   * - ``--std-parameterization``
     - ``exp``
     - Standard-deviation parameterization.
   * - ``--weight-decay``
     - ``0.0``
     - Optimizer weight decay.
   * - ``--lr-schedule``
     - ``constant``
     - Learning-rate schedule.

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
   * - ``dagger`` (online)
     - Expert-labeled dataset aggregation and cloning updates.
     - DAgger
   * - ``diffusion_cm_distill_online`` (online)
     - DPPO fine-tuning with online consistency distillation.
     - DiffusionCMDistillOnline
   * - ``gail`` (online)
     - PPO generator trained against an imitation discriminator.
     - GAIL
   * - ``policy_distillation`` (online)
     - On-policy action regression from a frozen teacher.
     - PolicyDistillation
   * - ``a2a_bc`` (offline)
     - Action-to-action flow-matching imitation.
     - A2ABC
   * - ``bc`` (offline)
     - Gaussian behavioral cloning by action likelihood.
     - BC
   * - ``consistency_distill_bc`` (offline)
     - Offline consistency student of a frozen diffusion teacher.
     - ConsistencyDistillBC
   * - ``diffusion_bc`` (offline)
     - Diffusion-policy pretraining with EMA weights.
     - DiffusionBC
   * - ``flow_bc`` (offline)
     - Conditional flow-matching imitation without a Q objective.
     - FlowBC
   * - ``mean_flow_bc`` (offline)
     - Imitation with the MeanFlow identity loss.
     - MeanFlowBC

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/tutorials/04-offline-pretraining`
  — H5 contract and the first behavioral-cloning run.

- :doc:`/tutorials/03-observations-and-encoders`
  — Image/state encoding and observation roles.

- :doc:`/algorithms/ppo-family`
  — Diffusion and flow PPO fine-tuning.

- :doc:`/REFERENCES`
  — Paper and reference implementation attribution.
