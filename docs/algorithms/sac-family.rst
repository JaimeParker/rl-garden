SAC family
==========

When to use
-----------

Use SAC for off-policy actor-critic learning with Box or Dict observations.
Use RLPD variants when online learning also samples a fixed offline prior.
Choose a sequence variant for a recurrent or transformer latent stage.
The family also groups deterministic DDPG/TD3 and specialized FlashSAC updates.

What the policy must provide
----------------------------

The standard SAC actor provides a reparameterized sample and its log-probability.
Read :doc:`/tutorials/08-bring-your-own-policy` for the SACActor methods and capability flags.
Read :ref:`extending-sac-seams` for actor construction, critic losses and optimizer ownership.
The flag checks belong to construction, before any rollout starts.

A deterministic actor needs entropy tuning disabled.
A flow path likelihood does not become an action density by using it in SAC.
SequenceSAC and SACFlow build their own actors and reject a supplied actor factory.
Do not assume every algorithm grouped here exposes the standard SAC factory seam.

Minimal command and construction
--------------------------------

RLPD mixes its prior data with online replay from the start.
ACRLPD adds action chunks and uses a different critic action space.
It executes each chunk one action per environment step, in rollout and in evaluation.
With ``--q-mc-diagnostics`` its Q-values are scored on the steps where a chunk was chosen.
It rejects ``--critic-only-steps`` above 0: Q-chunking has no critic-only phase.
A chunk that runs past its episode's end has zero weight in its critic loss, as in Q-chunking.
RLPDHybrid adds a discrete gripper head beside the continuous actor.
With ``--nstep`` above 1 that head bootstraps from the same n-step return as the critic.
These are separate choices from replacing the continuous actor head.

The DDPG class is exposed through the drqv2 registry entry, not a ddpg subcommand.
TD3 is a separate algorithm even when SAC uses entropy-off and target-noise controls.
SAC's entropy-off mode does not add a target actor copy.
See the tutorial's deterministic scenario before comparing those configurations.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-online sac --env-backend gymnasium --env-id Pendulum-v1 \
     --num-envs 1 --num-eval-envs 1 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start sac
   :end-before: # docs: end sac

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``online/sac`` by the command above.
They are not constructor defaults or recommended settings for every task.
Inspect again after selecting a preset; it can override these values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--policy-lr``
     - ``0.0003``
     - Actor optimizer step size.
   * - ``--q-lr``
     - ``0.0003``
     - Critic optimizer step size.
   * - ``--gamma``
     - ``0.8``
     - Discount in the critic target.
   * - ``--tau``
     - ``0.01``
     - Target-network interpolation rate.
   * - ``--ent-coef``
     - ``auto``
     - Automatic temperature tuning or a fixed coefficient.
   * - ``--utd``
     - ``0.25``
     - Update-to-data ratio.
   * - ``--batch-size``
     - ``512``
     - Replay samples per update batch.
   * - ``--target-policy-noise``
     - ``0.0``
     - Noise added to the target action.

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
   * - ``acrlpd`` (online)
     - RLPD with action-chunked critics and rollout.
     - ACRLPD
   * - ``drqv2`` (online)
     - DDPG with augmentation, n-step returns and scheduled exploration noise.
     - DDPG (DrQ-v2)
   * - ``flash_sac`` (online)
     - Distributional critic and unit-norm linear weights.
     - FlashSAC
   * - ``recurrent_sac`` (online)
     - Recurrent latent state with burn-in replay.
     - RecurrentSAC / TransformerSAC / RecurrentPPO / TransformerPPO
   * - ``rlpd`` (online)
     - SAC with a static offline prior mixed into online batches.
     - RLPD
   * - ``rlpd_hybrid`` (online)
     - RLPD with a discrete gripper head.
     - RLPDHybrid
   * - ``sac`` (online)
     - Entropy-regularized off-policy actor-critic.
     - SAC
   * - ``sac_flow`` (online)
     - SAC update with a flow-matching actor.
     - SACFlow
   * - ``td3`` (online)
     - Delayed deterministic policy updates and target-action smoothing.
     - TD3
   * - ``transformer_sac`` (online)
     - GTrXL latent state with burn-in-from-zero replay.
     - RecurrentSAC / TransformerSAC / RecurrentPPO / TransformerPPO

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/tutorials/02-online-rl`
  — Online SAC rollout, replay and logging.

- :doc:`/tutorials/03-observations-and-encoders`
  — Observation composition and actor/critic encoders.

- :doc:`/tutorials/08-bring-your-own-policy`
  — Flow actors, extractors and entropy-off construction.

- :doc:`/guides/extending`
  — Stable off-policy replay and loss hooks.
