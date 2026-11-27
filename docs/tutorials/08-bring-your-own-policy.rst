Bring your own policy
=========================

Goal
--------

You train your own actor or feature extractor with the existing PPO and SAC-family loops.
You validate its interface, then save and evaluate a model using an importable factory.
The four examples cover a transformer Gaussian head, a flow actor, an image extractor,
and a deterministic actor with SAC's entropy term disabled.

Prerequisites
-----------------

Complete :doc:`02-online-rl` and read :doc:`03-observations-and-encoders`.
Use the core installation from the repository root. The executable examples run on CPU.
The Cal-QL example reuses the synthetic dataset writer from :doc:`04-offline-pretraining`.
The RGB CLI inspections assume ManiSkill PushCube; the Python extractor smoke uses synthetic images.

Steps
---------

Choose the component boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The extractor maps observations to features; the actor maps those features to actions.
The algorithm still owns its critic/value head, optimizers, rollout/replay and checkpoint lifecycle.
For an algorithm-level change, use :doc:`09-extending-from-your-package` and :doc:`/guides/extending`.

.. list-table::
   :header-rows: 1

   * - What
     - Python
     - CLI
     - Called as
   * - the actor head
     - ``actor_factory=fn``
     - ``--actor-factory pkg.mod:fn``
     - ``fn(features_dim, action_space) -> nn.Module``
   * - the actor's feature extractor
     - ``policy_kwargs={"actor_extractor_class": Cls}``
     - ``--actor-extractor pkg.mod:Cls``
     - ``Cls(observation_space=...)``
   * - the critic's feature extractor
     - ``policy_kwargs={"critic_extractor_class": Cls}``
     - ``--critic-extractor pkg.mod:Cls``
     - ``Cls(observation_space=...)``

The feature width is the extractor output, a sequence PPO latent, or a SAC actor compressor output.
The action space is Box: the continuous part for RLPDHybrid and the flattened chunk for ACRLPD.
A factory receives neither the standard actor's hidden widths nor its norm, dropout or Gaussian options.
Your factory owns its architecture. An extractor receives a Dict observation space, even for flat state.
Images use NHWC tensors under modality keys; the example converts uint8 RGB to floating-point NCHW.
Extractor kwargs exist in Python but are not CLI fields or checkpoint metadata; use constructor defaults.

The flags come from ``PolicyModuleArgs`` and builders resolve them with ``policy_module_kwargs(args)``.
Online users include PPO, recurrent/transformer PPO, GAIL, SAC, RLPD, RLPDHybrid and ACRLPD.
Offline users include CQL, Cal-QL, WSRL and EDAC; off2on users include Cal-QL and WSRL.

Keep the module importable for training and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``pkg.mod:attr`` (a dotted attribute path also resolves).
The commands below put the tutorial directory on PYTHONPATH; your installed package needs no such prefix.
Static inspection prints the requested paths; it does not prove the factory can construct an actor.
Construction resolves imports and raises ``ConfigError`` naming an unimportable flag.
A dry run constructs the agent and records canonical import paths in the constructor description.

Checkpoint hyperparameters record ``actor_factory``, ``actor_extractor`` and ``critic_extractor`` when set.
Evaluation restores those paths before building the agent, including without the run's config.json.
An off2on continuation must construct a compatible actor too; pass the same factory explicitly.
A lambda, local function or __main__ definition warns when saving and records an unrecorded sentinel.
Automatic reconstruction then raises ``ConfigError``; you must rebuild the matching agent by hand.

Implement the PPO actor contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PPO stores a sampled action and later scores that same action under the updated policy.
Its stored-action log-probability must be differentiable in the actor parameters.
Return ``(action, log_prob, entropy)`` from ``action_log_prob(features, deterministic=False, *, sum_dims=True)``.
Return ``(log_prob, entropy)`` from ``evaluate_action_log_prob(features, actions, *, sum_dims=True)``.
Also supply ``deterministic_action(features)`` and ``clamp_action(actions)``.
Clamping is for the environment; PPO scores the unclamped stored action.

Actions have shape (B, A). Summed log-probabilities and entropies have shape (B, 1).
With ``sum_dims=False``, both have shape (B, A); non-factoring actors raise ``NotImplementedError``.
Both log-probability methods must agree about per-dimension support and entropy availability.
An actor with no tractable entropy returns None and requires ``ent_coef=0``; a positive coefficient is rejected.
Its entropy metric is omitted.

Gaussian capability additionally requires a log_std tensor and a forward method returning Normal.
The validator checks that scoring a sample of this Normal agrees with the actor's scoring method.
A pre-squash Normal alone does not qualify a tanh-squashed actor for adaptive-KL.
Read ``actor_capabilities.is_gaussian`` rather than relying on a runtime Protocol isinstance check.

Implement the SAC actor contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SAC differentiates through the sampled action. Its sampling path must be reparameterized.
Subclass ``SACActorBase`` to inherit the optional-method defaults; a plain module supplies all methods and flags:

.. list-table::
   :header-rows: 1

   * - Method / attribute
     - Shape
     - ``SACActorBase`` default
   * - ``action_log_prob(features)``
     - ``((B, A), (B, 1))``, reparameterized
     - none (implement it)
   * - ``deterministic_action(features)``
     - ``(B, A)``
     - none (implement it)
   * - ``sample_actions_with_log_prob(features, n)``
     - ``((B, n, A), (B, n))``
     - one ``action_log_prob`` call on the features repeated ``n`` times
   * - ``evaluate_action_log_prob(features, actions)``
     - ``(B, 1)``
     - raises ``NotImplementedError``
   * - ``diagnostics(features)``
     - ``{name: scalar tensor}``
     - ``log_prob_mean``, ``log_prob_std``, ``action_abs_mean`` of one sample
   * - ``stochastic`` (class attribute)
     - ``bool``
     - ``True``; ``False`` for a deterministic (TD3-shaped) actor whose log-prob is a zero placeholder
   * - ``log_prob_is_density``
     - ``bool``
     - ``True``; ``False`` when the log-prob is not the action's density (a flow actor's path likelihood, a deterministic actor's zero)
   * - ``supports_evaluate_log_prob``
     - ``bool``
     - ``False``; ``True`` when ``evaluate_action_log_prob`` scores external (dataset) actions

The conservative regularizer uses the multi-sample method; actor diagnostics are logged with the update.
A path likelihood can serve an experimental SAC entropy surrogate but is not an action density.
Disabling CQL importance sampling does not turn it into one.

What is checked at construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validators check presence, shapes and return structure before rollout.
They use random batches of two under no_grad and fork_rng, with submodules temporarily in eval mode.
CPU and the selected CUDA torch RNG are restored, as are each submodule's training flags.
They do not check gradients or isolate NumPy/Python random state.

.. list-table::
   :header-rows: 1

   * - Problem
     - Error
   * - the factory returns something that is not an ``nn.Module``
     - ``TypeError: A PPO actor must be a torch.nn.Module, got <type>.``
   * - a method or SAC flag is missing
     - ``TypeError: <Cls> does not implement the PPO actor contract (rl_garden.policies.contracts.PPOActor); missing: clamp_action.`` (``SAC`` / ``SACActor`` for the SAC family)
   * - a returned tensor has the wrong shape
     - ``ValueError: <Cls>.<method>(...): <what> has shape (..), expected (..).``
   * - a wrong return arity or type
     - ``TypeError`` naming the method
   * - PPO: only one of the two log-prob methods returns an entropy, or only one accepts ``sum_dims=False``
     - ``ValueError``
   * - PPO ``--lr-schedule adaptive_kl`` with a non-Gaussian actor
     - ``ValueError: lr_schedule='adaptive_kl' (--lr-schedule adaptive_kl) needs a diagonal Gaussian actor ...``
   * - PPO ``--decision-level`` with an actor without per-dimension log-probs
     - ``ValueError: decision_level=True needs an actor whose action_log_prob and evaluate_action_log_prob accept sum_dims=False ...``
   * - PPO ``--actor-factory`` together with ``--policy-graph``
     - ``ValueError: actor_factory (--actor-factory) cannot be combined with policy_graph (--policy-graph) ...``
   * - SAC ``ent_coef="auto"`` (the default) with ``stochastic=False``
     - ``ValueError: ent_coef='auto' tunes the temperature toward a target entropy, which needs a stochastic actor, but the actor declares stochastic=False; pass ent_coef=0.0.``
   * - CQL / Cal-QL / WSRL ``cql_importance_sample=True`` (the default) with ``log_prob_is_density=False``
     - ``ValueError: cql_importance_sample=True subtracts the sampled actions' log-densities from their Q-values, but the actor declares log_prob_is_density=False; pass cql_importance_sample=False.``
   * - ACRLPD ``bc_alpha > 0`` with ``supports_evaluate_log_prob=False``
     - ``ValueError: bc_alpha=... scores the dataset actions with the actor's evaluate_action_log_prob, ...``
   * - a factory for ``sac_flow``, ``recurrent_sac``, ``transformer_sac``
     - ``ValueError`` (their policies build their own actor; those Args have no ``--actor-factory``)

Current checks additionally reject non-boolean SAC flags and inconsistent sample/self-scoring log-probabilities.
Self-scoring is checked for PPO and SAC actors declaring external-action scoring support.
The comparison uses 1e-3 tolerances, excludes near-bound saturated rows, retries once, then skips if all remain saturated.
PPO also rejects a positive entropy coefficient when the actor reports None.
Each example calls the appropriate validator and then trains; tests check finite parameters and actor gradients.

A transformer Gaussian actor for PPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This plain nn.Module makes tokens from the features, applies a small transformer and predicts a Normal.
Its log_std is a parameter. Dropout is zero so rollout and update scoring use the same forward map.

.. literalinclude:: ../../examples/tutorials/ch08_transformer_ppo_actor.py
   :language: python
   :start-after: # docs: start component
   :end-before: # docs: end component

.. literalinclude:: ../../examples/tutorials/ch08_transformer_ppo_actor.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

Train the 200-step CPU example through the console script:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online ppo --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 --num-eval-envs 1 \
     --total-timesteps 200 --eval-freq 0 --log-freq 100 --no-capture-video --num-steps 8 --num-minibatches 2 \
     --update-epochs 1 --actor-factory ch08_transformer_ppo_actor:make_transformer_actor --exp-name \
     chapter08-ppo

The same factory can read a recurrent PPO latent; inspect that wiring without constructing an environment:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online recurrent_ppo --actor-factory ch08_transformer_ppo_actor:make_transformer_actor \
     --print-config

For a tanh head with no entropy, see the tested fixture's ``make_tanh_gaussian_ppo_actor``.

A flow actor for SAC and Cal-QL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Euler integration adds reparameterized Gaussian noise at each step.
The accumulated log-probability is a path likelihood, so the actor declares ``log_prob_is_density=False``.

.. literalinclude:: ../../examples/tutorials/ch08_flow_sac_actor.py
   :language: python
   :start-after: # docs: start component
   :end-before: # docs: end component

.. literalinclude:: ../../examples/tutorials/ch08_flow_sac_actor.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

The Python smoke performs online SAC updates and one Cal-QL update on the chapter-04 synthetic H5.
The temporary dataset is removed afterward. Run the matching SAC CLI for 200 steps:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online sac --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 --num-eval-envs 1 \
     --total-timesteps 200 --eval-freq 0 --log-freq 100 --no-capture-video --buffer-device cpu --buffer-size \
     1000 --batch-size 16 --hidden-dim 32 --learning-starts 16 --training-freq 4 --actor-factory \
     ch08_flow_sac_actor:make_flow_actor --exp-name chapter08-sac

For Cal-QL, disable importance sampling because its density correction is invalid for this actor.
These two commands inspect the offline and off2on forms; they do not train:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-offline calql --offline-dataset runs/chapter04-data/pendulum.h5 --actor-factory \
     ch08_flow_sac_actor:make_flow_actor --no-cql-importance-sample --print-config

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-off2on calql --env-backend gymnasium --env-id Pendulum-v1 --offline-dataset \
     runs/chapter04-data/pendulum.h5 --num-offline-steps 4 --actor-factory ch08_flow_sac_actor:make_flow_actor \
     --no-cql-importance-sample --print-config

The second command's live backend is Gymnasium Pendulum; both refer to the chapter-04 synthetic file.

An external image backbone with standard heads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This BaseFeaturesExtractor reads RGB and state keys and returns one feature vector per batch row.
Its small convolutional backbone is a frozen stand-in, not pretrained VLA weights.
The projection remains trainable. The stand-in has no running statistics or dropout to freeze separately.

.. literalinclude:: ../../examples/tutorials/ch08_backbone_extractor.py
   :language: python
   :start-after: # docs: start component
   :end-before: # docs: end component

.. literalinclude:: ../../examples/tutorials/ch08_backbone_extractor.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

Under shared_critic_grad, the critic/value path trains the shared extractor and the actor reads detached features.
The shared mode allows both paths to train it. A separate critic extractor requires separate sharing.
An external extractor replaces schema-driven construction; encoder options do not configure its architecture.
Inspect the image forms and the asymmetric wiring:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online sac --env-id PushCube-v1 --obs.rgb base_camera --actor-extractor \
     ch08_backbone_extractor:BackboneExtractor --print-config

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online ppo --env-id PushCube-v1 --obs.rgb base_camera --actor-extractor \
     ch08_backbone_extractor:BackboneExtractor --print-config

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online sac --env-id PushCube-v1 --obs.rgb base_camera --actor-extractor \
     ch08_backbone_extractor:BackboneExtractor --critic-extractor ch08_backbone_extractor:BackboneExtractor \
     --encoder-sharing separate --print-config


TD3-shaped SAC with entropy disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the existing deterministic tanh actor rather than implementing a second one:

.. literalinclude:: ../../examples/tutorials/ch08_td3_shaped_sac.py
   :language: python
   :start-after: # docs: start component
   :end-before: # docs: end component

.. literalinclude:: ../../examples/tutorials/ch08_td3_shaped_sac.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

Inspect the entropy-off controls:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     rl-garden-train-online sac --env-backend gymnasium --env-id Pendulum-v1 --actor-factory \
     ch08_td3_shaped_sac:make_deterministic_actor --ent-coef 0 --target-policy-noise 0.2 --target-noise-clip \
     0.5 --actor-q-agg first --print-config

Zero entropy coefficient disables the entropy term; automatic temperature tuning rejects deterministic actors.
Target-policy noise is Gaussian noise in action units, clipped before the action is clamped to its bounds.
Zero target noise draws no random numbers. Actor-Q aggregation chooses min, mean or the first critic.
ACRLPD rejects a nondefault actor_q_agg: its actor uses the ensemble mean and its q_agg controls targets.
The actor follows SAC's update schedule. The in-tree TD3, DDPG and TD3-BC algorithms remain separate; do not claim bit-identical parity.

Limits
^^^^^^^^^^

- Both actor contracts and validators require Box action spaces.
- Decision-level PPO needs per-dimension log-probabilities even if the environment reports only decision.
- Adaptive-KL needs Gaussian capabilities; RLinf PPO also requires per-dimension scoring.
- SACFlow, recurrent SAC and transformer SAC build their own actors and reject actor factories.
- A policy subclass overriding _build_actor bypasses the factory; its constructed actor is still validated.
- Entropy-off SAC uses the online actor for the critic target, with no target actor copy.

What you ran
----------------

.. note::

   The PPO and SAC commands above each train for 200 steps and save final checkpoints.
   Watch PPO's ``losses/policy_loss`` and SAC's ``losses/actor_loss`` and ``losses/critic_loss``.
   Static commands only inspect wiring. The four Python examples are executed in the CPU tutorial tests.

Evaluate both saved models with the tutorial modules still importable:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-eval \
     --checkpoint-path runs/chapter08-ppo/checkpoints/final.pt --device cpu --num-eval-envs 1 \
     --num-eval-episodes 2

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-eval \
     --checkpoint-path runs/chapter08-sac/checkpoints/final.pt --device cpu --num-eval-envs 1 \
     --num-eval-episodes 2

.. code-block:: text

   runs/chapter08-ppo/config.json
   runs/chapter08-ppo/checkpoints/final.pt
   runs/chapter08-sac/config.json
   runs/chapter08-sac/checkpoints/final.pt


Next
--------

Use :doc:`09-extending-from-your-package` when you need a new algorithm or loss.
Keep :doc:`/guides/extending` as the authoritative extension contract.
