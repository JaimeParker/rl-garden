Python API index
================

This is a hand-written directory of importable construction and extension
entry points. It does not expand docstrings or promise a common constructor
signature across algorithms. Use the linked family pages and tutorials for
combinations exercised by the examples.
The :doc:`/guides/extending` contract names the supported subclass hooks.

rl_garden.algorithms
--------------------

Import the following exported classes from ``rl_garden.algorithms``.
Family links identify the relevant learning path; base classes and result
containers are infrastructure, not additional CLI registrations.
The :doc:`_generated/algorithms` table separately lists registered phase names.

.. list-table::
   :header-rows: 1

   * - Class
     - Purpose
     - Family
   * - ``PPO``
     - Clipped on-policy optimization.
     - :doc:`/algorithms/ppo-family`
   * - ``RecurrentPPO``
     - PPO with a persistent LSTM/GRU latent state.
     - :doc:`/algorithms/ppo-family`
   * - ``TransformerPPO``
     - PPO with GTrXL segment memory.
     - :doc:`/algorithms/ppo-family`
   * - ``SequencePPO``
     - Shared sequence-latent PPO base.
     - :doc:`/algorithms/ppo-family`
   * - ``DPPO``
     - PPO over a diffusion denoising chain.
     - :doc:`/algorithms/ppo-family`
   * - ``FlowPPO``
     - PPO over stochastic flow steps.
     - :doc:`/algorithms/ppo-family`
   * - ``OnPolicyAlgorithm``
     - Shared rollout and update lifecycle for on-policy methods.
     - :doc:`/algorithms/ppo-family`
   * - ``SAC``
     - Soft actor-critic.
     - :doc:`/algorithms/sac-family`
   * - ``DDPG``
     - DrQ-v2-style deterministic actor-critic with augmentation.
     - :doc:`/algorithms/sac-family`
   * - ``TD3``
     - Twin-delayed deterministic policy gradients.
     - :doc:`/algorithms/sac-family`
   * - ``FlashSAC``
     - FlashSAC for Box observations.
     - :doc:`/algorithms/sac-family`
   * - ``SACFlow``
     - SAC with a flow-matching actor.
     - :doc:`/algorithms/sac-family`
   * - ``RecurrentSAC``
     - SAC with recurrent replay and a recurrent latent stage.
     - :doc:`/algorithms/sac-family`
   * - ``TransformerSAC``
     - SAC with GTrXL segment memory.
     - :doc:`/algorithms/sac-family`
   * - ``SequenceSAC``
     - Shared sequence-latent SAC base.
     - :doc:`/algorithms/sac-family`
   * - ``RLPD``
     - SAC extended with prior-data mixing.
     - :doc:`/algorithms/sac-family`
   * - ``RLPDHybrid``
     - RLPD with discrete gripper-action heads.
     - :doc:`/algorithms/sac-family`
   * - ``ACRLPD``
     - Action-chunked RLPD.
     - :doc:`/algorithms/sac-family`
   * - ``OffPolicyAlgorithm``
     - Shared environment rollout and replay-update lifecycle.
     - :doc:`/algorithms/sac-family`
   * - ``BaseAlgorithm``
     - Shared seeding, logging, device and checkpoint lifecycle.
     - :doc:`/algorithms/sac-family`
   * - ``AWAC``
     - Advantage-weighted actor regression with Q-learning.
     - :doc:`/algorithms/offline-rl`
   * - ``BCQ``
     - VAE-constrained action perturbation and Q-learning.
     - :doc:`/algorithms/offline-rl`
   * - ``BPPO``
     - Behavior proximal policy optimization.
     - :doc:`/algorithms/offline-rl`
   * - ``CQL``
     - Conservative Q-learning over static replay.
     - :doc:`/algorithms/offline-rl`
   * - ``CalQL``
     - CQL with calibrated lower bounds.
     - :doc:`/algorithms/offline-rl`
   * - ``EDAC``
     - Critic ensembles with gradient-diversity regularization.
     - :doc:`/algorithms/offline-rl`
   * - ``FAC``
     - Conservative flow actor-critic.
     - :doc:`/algorithms/offline-rl`
   * - ``FINO``
     - Flow matching with injected noise.
     - :doc:`/algorithms/offline-rl`
   * - ``FloQ``
     - Flow-matching and distilled critics with a flow actor.
     - :doc:`/algorithms/offline-rl`
   * - ``FQL``
     - Q-learning with a two-network flow actor.
     - :doc:`/algorithms/offline-rl`
   * - ``HILP``
     - Hilbert representations and skill-policy pretraining.
     - :doc:`/algorithms/offline-rl`
   * - ``IDQL``
     - IQL-style values with diffusion actor extraction.
     - :doc:`/algorithms/offline-rl`
   * - ``IQL``
     - Expectile value fitting and advantage-weighted actor fitting.
     - :doc:`/algorithms/offline-rl`
   * - ``OfflineSAC``
     - Offline SAC scaffold over static replay.
     - :doc:`/algorithms/offline-rl`
   * - ``OfflineRLAlgorithm``
     - Base lifecycle for dataset-driven training.
     - :doc:`/algorithms/offline-rl`
   * - ``OfflineEnvSpec``
     - Space-only environment stand-in without reset or step.
     - :doc:`/algorithms/offline-rl`
   * - ``OfflinePretrainResult``
     - Offline run counters, final metrics and checkpoint path.
     - :doc:`/algorithms/offline-rl`
   * - ``OPAL``
     - Skill-space VAE pretraining.
     - :doc:`/algorithms/offline-rl`
   * - ``PLAS``
     - Latent policy over a frozen pretrained VAE.
     - :doc:`/algorithms/offline-rl`
   * - ``QAM``
     - Adjoint matching for a flow actor.
     - :doc:`/algorithms/offline-rl`
   * - ``QGF``
     - Q-guided flow-policy extraction.
     - :doc:`/algorithms/offline-rl`
   * - ``ReBRAC``
     - TD3-BC with critic-side behavior regularization.
     - :doc:`/algorithms/offline-rl`
   * - ``SPOT``
     - VAE support constraints for offline actor-critic.
     - :doc:`/algorithms/offline-rl`
   * - ``TD3BC``
     - TD3 with actor behavior-cloning regularization.
     - :doc:`/algorithms/offline-rl`
   * - ``UniO4``
     - Behavior-cloning ensemble and shared-critic policy improvement.
     - :doc:`/algorithms/offline-rl`
   * - ``UniO4OPE``
     - Uni-O4 with dynamics-model evaluation gating.
     - :doc:`/algorithms/offline-rl`
   * - ``ValueFlows``
     - Flow-matching critics over return distributions.
     - :doc:`/algorithms/offline-rl`
   * - ``ACFQL``
     - Action-chunked offline-to-online FQL.
     - :doc:`/algorithms/offline-to-online`
   * - ``ExPLORe``
     - RLPD extension for exploration with prior data.
     - :doc:`/algorithms/offline-to-online`
   * - ``JSRL``
     - SAC exploration jump-started by a frozen guide.
     - :doc:`/algorithms/offline-to-online`
   * - ``SUPE``
     - Skill-based extension of ExPLORe.
     - :doc:`/algorithms/offline-to-online`
   * - ``SO2``
     - Offline multi-critic SAC with target-action smoothing.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnSO2``
     - SO2 continuation with offline-data mixing.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnAWAC``
     - AWAC offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnCalQL``
     - Cal-QL offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnFAC``
     - FAC offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnFINO``
     - FINO offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnFloQ``
     - FloQ offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnFQL``
     - FQL with uninterrupted online updates after pretraining.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnIQL``
     - IQL offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnSPOT``
     - SPOT offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``Off2OnValueFlows``
     - Value Flows offline-to-online host.
     - :doc:`/algorithms/offline-to-online`
   * - ``WSRL``
     - Cal-QL pretraining, warmup and online SAC.
     - :doc:`/algorithms/offline-to-online`
   * - ``A2ABC``
     - Action-to-action flow-matching imitation.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``BC``
     - Expert-action log-likelihood training.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``ConsistencyDistillBC``
     - Offline consistency distillation.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``DAgger``
     - Expert-labelled data aggregation during rollout.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``DiffusionBC``
     - Diffusion-policy pretraining.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``DiffusionCMDistillOnline``
     - Online diffusion-policy fine-tuning with consistency distillation.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``FlowBC``
     - Conditional flow-matching imitation.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``GAIL``
     - PPO generator with adversarial imitation rewards.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``MeanFlowBC``
     - One-step-sampleable mean-flow imitation.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``PolicyDistillation``
     - On-policy teacher-to-student action regression.
     - :doc:`/algorithms/imitation-and-generative`
   * - ``DreamerV3``
     - RSSM learning with imagined actor-critic updates.
     - :doc:`/algorithms/model-based`
   * - ``NCRL``
     - Pretrained-RSSM online fine-tuning with rehearsal and guidance.
     - :doc:`/algorithms/model-based`
   * - ``OfflineModelBasedAlgorithm``
     - Dataset-driven host for world-model methods.
     - :doc:`/algorithms/model-based`
   * - ``PWM``
     - Policy learning on a frozen multitask world model.
     - :doc:`/algorithms/model-based`
   * - ``PWMCore``
     - PWM losses and optimizer mixin.
     - :doc:`/algorithms/model-based`
   * - ``RSSMPretrain``
     - Reward-free RSSM pretraining.
     - :doc:`/algorithms/model-based`
   * - ``SuperMimic``
     - Policy gradients through an explicit-state world model.
     - :doc:`/algorithms/model-based`
   * - ``SuperMimicPolicy``
     - Goal-conditioned history actor and state-value critic container.
     - :doc:`/algorithms/model-based`
   * - ``TDMPC2``
     - Latent world-model learning with decision-time planning.
     - :doc:`/algorithms/model-based`
   * - ``TDMPC2Core``
     - Shared TD-MPC2 losses and optimizer mixin.
     - :doc:`/algorithms/model-based`
   * - ``TDMPC2Multitask``
     - Offline multitask TD-MPC2 training.
     - :doc:`/algorithms/model-based`

``OfflineEnvSpec`` supplies observation/action spaces and the environment
count without live interaction. Use it to construct an offline agent as in
:doc:`/tutorials/04-offline-pretraining`; it has no reset or step method.
``OfflinePretrainResult`` records the outcome of the offline runner.

rl_garden.envs
--------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``GymnasiumEnvConfig``
     - Configuration for vectorizing a registered Gymnasium environment.
   * - ``make_gymnasium_env``
     - Build that environment with explicit torch output conversion.

See :doc:`/tutorials/07-bring-your-own-environment`.

rl_garden.envs.vector_env
-------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``TorchVectorEnvAdapter``
     - Adapt NumPy vector environments and terminal information to torch.

See :doc:`/backends/index`.

rl_garden.envs.backend_registry
-------------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``EnvRequest``
     - Backend-neutral environment and observation request.
   * - ``EnvBackend``
     - Version-2 backend interface.
   * - ``register_env_backend``
     - Register a backend class under a unique name.
   * - ``discover_env_backends``
     - Discover in-tree and installed-package backend registrations.
   * - ``resolve_backend_config``
     - Read the selected backend configuration from Args.
   * - ``materialize_backend_configs``
     - Resolve train/eval configs without constructing simulator resources.
   * - ``make_training_envs``
     - Build train and optional evaluation environments and validate keys.
   * - ``make_evaluation_env``
     - Build an evaluation environment and validate its observation contract.

See :doc:`/guides/extending`.

rl_garden.encoders
------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``EncoderConfig``
     - Configuration for observation feature extraction.
   * - ``BaseFeaturesExtractor``
     - Base class carrying the extractor output width.
   * - ``build_observation_encoder``
     - Build the schema-driven observation feature extractor.

See :doc:`/tutorials/03-observations-and-encoders`.

rl_garden.policies.contracts
----------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``PPOActor``
     - Protocol for sampled and stored-action PPO scores.
   * - ``GaussianParamsActor``
     - Optional Gaussian-parameter protocol for PPO actors.
   * - ``SACActor``
     - Protocol for reparameterized SAC sampling and capability flags.
   * - ``PPOActorCapabilities``
     - Validated entropy, per-dimension and Gaussian capabilities.
   * - ``SACActorCapabilities``
     - Validated stochastic, density and external-scoring flags.
   * - ``validate_ppo_actor``
     - Probe PPO actor structure and return its capabilities.
   * - ``validate_sac_actor``
     - Probe SAC actor structure and return its capabilities.

See :doc:`/tutorials/08-bring-your-own-policy`.

rl_garden.networks
------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``SACActorBase``
     - Base actor with optional SAC-method defaults.
   * - ``SquashedGaussianActor``
     - Tanh-squashed Gaussian actor for SAC-family policies.
   * - ``DeterministicTanhActor``
     - Deterministic bounded actor with placeholder log scores.
   * - ``UnsquashedGaussianActor``
     - Unsquashed Gaussian actor for advantage-weighted policies.
   * - ``DiagGaussianActor``
     - Diagonal Gaussian PPO actor.
   * - ``SplitDiagGaussianActor``
     - PPO actor with separate mean heads for action slices.

See :doc:`/tutorials/08-bring-your-own-policy`.

rl_garden.training.algorithm_registry
-------------------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``AlgorithmEntry``
     - Args type, runner, optional class thunk and optional builder.
   * - ``BaseAlgorithmRegistry``
     - Phase registry providing register, discover, entries and run_cli methods.

See :doc:`/tutorials/09-extending-from-your-package`.

Call ``register`` on the online, offline or off2on registry instance;
there is no module-level register function in algorithm_registry.
The class thunk is optional and delays the algorithm import.
Evaluation uses the entry's builder to reconstruct the trained agent.

rl_garden.buffers.h5_dataset
----------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``infer_specs_from_h5``
     - Infer Dict observation and bounded action spaces from trajectory H5.
   * - ``infer_box_specs_from_h5``
     - Infer state-only Box spaces; reject visual observations.
   * - ``load_h5_dataset_to_replay_buffer``
     - Load trajectory transitions into an existing replay buffer.

See :doc:`/tutorials/04-offline-pretraining`.

rl_garden.common.checkpoint
---------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``checkpoint_dict``
     - Assemble the checkpoint metadata and state dictionary.
   * - ``save_checkpoint_file``
     - Write a checkpoint dictionary as a torch file.
   * - ``load_checkpoint_file``
     - Read a checkpoint dictionary onto the requested device.
   * - ``validate_checkpoint_metadata``
     - Validate algorithm, space and policy-graph compatibility.
   * - ``save_replay_buffer_file``
     - Write a separate replay-buffer snapshot.
   * - ``load_replay_buffer_file``
     - Restore a replay-buffer snapshot into an existing buffer.

See :doc:`/tutorials/06-checkpoints-and-evaluation`.

Python network composition
--------------------------

Pass ``net_arch`` as shared MLP widths or, for SAC/WSRL, a mapping with ``pi`` and ``qf`` widths; IQL also accepts ``vf`` widths.
Pass ``encoder_config``, ``obs_groups``, ``critic_encoder_config`` and ``encoder_sharing`` directly when composing an algorithm in Python.
The internal ``ObservationEncoderMixin`` calls ``build_observation_encoder`` to construct a state ``FlattenExtractor`` or multimodal ``CombinedExtractor``.
See :doc:`/tutorials/03-observations-and-encoders` for the tested construction example.

rl_garden.common.pretrained_paths
---------------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
   * - ``default_pretrained_root``
     - Resolve the XDG cache root for externally sourced weights.
   * - ``pretrained_dir``
     - Resolve a kind-specific cache path or the explicit environment override.

See :doc:`/tutorials/03-observations-and-encoders`.

The explicit ``RL_GARDEN_PRETRAINED_DIR`` override is one flat directory.
Otherwise, weights live below the XDG cache root by kind, defaulting to
``~/.cache/rl_garden/pretrained/<kind>/``. These helpers do not search the
repository or download weights.
