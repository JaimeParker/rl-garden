PPO family
==========

When to use
-----------

Use this family for PPO updates on newly collected rollouts with Box or Dict observations.
Choose a recurrent or transformer variant when your policy needs a latent sequence stage.
Diffusion and flow fine-tuning keep their own chain-policy updates in this family.
Start with the standard actor before introducing a graph or custom head.

What the policy must provide
----------------------------

The standard policy samples actions and scores those stored actions again during an update.
Use :doc:`/tutorials/08-bring-your-own-policy` for the PPOActor interface and validators.
Use :ref:`extending-ppo-seams` for policy construction, auxiliary losses and rollout extras.
A custom actor is a smaller change than a new algorithm when the objective stays PPO.

A positive entropy coefficient requires an actor that reports entropy.
Adaptive-KL scheduling additionally needs the validated Gaussian capability.
Decision-level updates require per-dimension action scoring.
The tutorial describes these checks and the factory/checkpoint round trip.

Minimal command and construction
--------------------------------

Decision-level PPO and policy graphs are options within this family, not new registry names.
Use :doc:`/guides/decision-level-ppo` when actions take effect only at environment decisions.
Use :doc:`/guides/policy-graphs` to change the network between observations and PPO heads.
A graph cannot be combined with an actor factory.

The recurrent policy passes the sequence latent width to the actor factory.
Do not size that head from the raw observation space.
DPPO and FlowPPO train chains through their specialized policies; the plain PPOActor
example is not a replacement for their chain bookkeeping.

Run this inspection from the repository root. It prints configuration and exits.
It does not construct a simulator or prove that a dataset or checkpoint is valid.

.. code-block:: bash

   rl-garden-train-online ppo --env-backend gymnasium --env-id Pendulum-v1 \
     --num-envs 1 --num-eval-envs 1 --print-config

The Python example constructs a small CPU agent and never calls learn().
Its explicit sizes are a construction test, not benchmark hyperparameters.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start ppo
   :end-before: # docs: end ppo

The shared harness creates the spaces, builds all six representatives and closes the environment.
Use the complete source file when you want to execute the construction example.

.. literalinclude:: ../../examples/tutorials/ch_algorithms.py
   :language: python
   :start-after: # docs: start setup
   :end-before: # docs: end setup

Key parameters
--------------

These are the CLI defaults printed for ``online/ppo`` by the command above.
They are not constructor defaults or recommended settings for every task.
Inspect again after selecting a preset; it can override these values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--learning-rate``
     - ``0.0003``
     - Optimizer step size.
   * - ``--num-steps``
     - ``50``
     - Rollout steps per environment.
   * - ``--num-minibatches``
     - ``32``
     - Minibatches formed from a rollout.
   * - ``--update-epochs``
     - ``4``
     - Passes over the rollout.
   * - ``--gamma``
     - ``0.8``
     - Discount per environment step.
   * - ``--gae-lambda``
     - ``0.9``
     - GAE trace weighting.
   * - ``--clip-coef``
     - ``0.2``
     - Policy-ratio clipping width.
   * - ``--ent-coef``
     - ``0.0``
     - Entropy-bonus coefficient.

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
   * - ``dppo`` (online)
     - PPO fine-tuning of a pretrained diffusion policy.
     - DPPO
   * - ``flow_ppo`` (online)
     - PPO fine-tuning of a stochastic flow chain.
     - FlowPPO
   * - ``ppo`` (online)
     - Clipped policy updates with a value head.
     - PPO
   * - ``recurrent_ppo`` (online)
     - LSTM/GRU latent stage before the actor and value heads.
     - RecurrentSAC / TransformerSAC / RecurrentPPO / TransformerPPO
   * - ``transformer_ppo`` (online)
     - GTrXL latent stage before the actor and value heads.
     - RecurrentSAC / TransformerSAC / RecurrentPPO / TransformerPPO

The complete :doc:`/reference/_generated/algorithms` table reports the class resolved
by each registration, including entries whose optional class thunk is absent.

Deep dives
----------

- :doc:`/guides/decision-level-ppo`
  — Decision masks, return calculation and incompatible combinations.

- :doc:`/guides/policy-graphs`
  — Graph nodes, heads, losses and checkpoint behavior.

- :doc:`/design/rlinf-integration`
  — The RLinf PPO adapter contract.

- :doc:`/tutorials/02-online-rl`
  — A complete rollout-and-update training example.
