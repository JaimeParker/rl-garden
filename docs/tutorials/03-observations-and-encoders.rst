Observations and encoders
=========================

Goal
----

You select camera observations, configure their encoder, and choose what
the actor and critic can see.
You then exercise the encoder with a synthetic CPU environment.
No renderer is needed for that example.

Prerequisites
-------------

Complete :doc:`02-online-rl`.
Read the observation section of :doc:`/guides/configuration`.
The inspection commands below need no simulator.
Actually training their camera configurations requires ManiSkill.

Keep observation selection separate from feature extraction.
A camera name belongs to the environment request.
An encoder backbone belongs to the learning architecture.

Steps
-----

Start from the observation contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A state-only run uses the ``state`` key.
Camera observations use keys such as ``rgb_base_camera``.
Images arrive with their channel dimension last.
The observation schema checks the supported key vocabulary.

Requesting a camera does not imply dropping state.
The state selection remains active unless you turn it off.
Inspect the resolved observation configuration before training.

Use :ref:`configuration-observation-and-encoder-configuration`
for the supported fields.
Keep the guide as the field reference rather than copying its table.

Request RGB and a convolutional encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspect a SAC camera configuration:

.. code-block:: bash

   rl-garden-train-online sac \
     --env-id PushCube-v1 --obs.rgb base_camera --encoder.backbone plain_conv \
     --print-config

The camera selection requests ``base_camera`` from the backend.
The encoder selection chooses how the resulting image is processed.
The command stops after preflight.

A non-default image encoder option without an image observation can be
rejected as inactive.
Selecting a backbone alone does not cause the simulator to render.
Inspect the observation request before debugging the network.

For PPO, start from its checked-in RGB preset:

.. code-block:: bash

   rl-garden-train-online ppo \
     --config configs/online/ppo_rgb.yaml --obs.rgb base_camera \
     --encoder.backbone plain_conv --print-config

Read the image size and camera selection from the resolved preset.
The displayed values belong to this invocation.
Do not assume every backend has the same camera names or image dimensions.

Exercise the encoder without a renderer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tutorial fixture emits synthetic state and RGB arrays.
It provides the same key and shape vocabulary expected by the encoder.
Its reward is only a wiring check.

.. literalinclude:: ../../examples/tutorials/ch03_encoder_config.py
   :language: python
   :start-after: # docs: start environment
   :end-before: # docs: end environment

The vector adapter converts these CPU arrays to tensors.
That conversion is explicit at the environment boundary.
The example keeps the algorithm and replay on CPU.

Build the encoder and perform a small SAC update:

.. literalinclude:: ../../examples/tutorials/ch03_encoder_config.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

The image branch produces features from ``rgb_camera``.
The state branch processes the low-dimensional vector.
The combined extractor supplies the policy with both.

The test executes this example and checks that updates occurred.
It also checks finite learned parameters.
It does not claim that random synthetic observations form a useful task.

Give actor and critic different inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An actor can consume camera features while a critic also receives state.
Inspect that asymmetric selection:

.. code-block:: bash

   rl-garden-train-online sac \
     --env-id PushCube-v1 --obs.rgb base_camera --obs-groups.actor rgb_base_camera \
     --obs-groups.critic rgb_base_camera state --encoder-sharing separate \
     --print-config

The actor group contains only the image key.
The critic group contains the image key and state.
The separate-sharing choice gives them distinct extractors.

For supported algorithms, asymmetric groups imply separate encoders.
An explicit incompatible sharing choice raises an error.
The restrictions for recurrent architectures are in
:ref:`configuration-actorcritic-encoder-asymmetry`.

Do not infer privileged state from an arbitrary observation name.
The backend must actually produce every requested key.
Preflight cannot establish every simulator-specific fact.

Choose a distinct critic encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different inputs are not the only reason to separate extractors.
You can also choose a different architecture for the critic:

.. code-block:: bash

   rl-garden-train-online sac \
     --env-id PushCube-v1 --obs.rgb base_camera --encoder.backbone plain_conv \
     --critic-encoder.backbone resnet18 --encoder-sharing separate \
     --print-config

The actor uses the primary encoder configuration.
The critic override supplies its own backbone.
Separate extractors make that difference possible.

A shared extractor cannot simultaneously have two architectures.
Keep the sharing decision visible when comparing runs.
See the same configuration-guide anchor for inference and validation rules.

Understand gradient sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

With shared critic-gradient encoding, the actor reads detached features.
The critic's loss trains the shared extractor.
With fully shared encoding, both losses can train it.
Separate encoding gives each side its own extractor.

Those are learning choices, not just memory settings.
They change which loss can update a feature representation.
Use the effective configuration to record the choice.

The observation groups also apply to state-only experiments.
For extra state and privileged inputs, use
:ref:`configuration-low-dim-observations-and-privileged-state`.

ResNet freezing and image fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When loading pretrained weights, use ``--encoder.freeze-resnet-backbone`` to freeze the stem and residual blocks while training the pooling/bottleneck head.
With pretrained weights, use ``--encoder.freeze-resnet-encoder`` to freeze the full visual extractor.
The ``stack_channels`` fusion concatenates images before encoding; ``per_key`` encodes each image key independently and concatenates features.
Backbones include ``plain_conv``, ``resnet10``, ``resnet18``, ``drqv2_conv``, ``cnn3d``, ``dreamer_conv`` and ``vit``; see :ref:`configuration-observation-and-encoder-configuration` for their options.

What you ran
------------

.. note::

   The SAC RGB, PPO RGB, asymmetric-input and critic-encoder commands above
   print configurations without creating run directories.
   The Python example performs a CPU update without saving a checkpoint.
   When you launch a real run, expect ``runs/<run_name>/config.json``
   and ``runs/<run_name>/checkpoints/final.pt``.
   For SAC, track ``losses/actor_loss``, ``losses/critic_loss``
   and ``q/predicted``.

Next
----

Continue to :doc:`06-checkpoints-and-evaluation` to preserve a trained
encoder together with its policy.
The tutorial :doc:`08-bring-your-own-policy` covers external extractors.

Pretrained backbone weights
---------------------------

Externally sourced backbone weights are separate from training checkpoints.
The lookup never searches the repository tree.
A source checkout does not need a ``pretrained/`` directory.

If ``RL_GARDEN_PRETRAINED_DIR`` is set to a nonempty path, that flat
directory is used for every weight kind.
It replaces the cache lookup; it does not add another fallback search.
Otherwise, the root is ``$XDG_CACHE_HOME/rl_garden/pretrained/``.
When that variable is unset, it is ``~/.cache/rl_garden/pretrained/``.

ResNet names resolve beneath the ``resnet/`` kind directory.
For example, ``resnet10-imagenet`` resolves to
``~/.cache/rl_garden/pretrained/resnet/resnet10-imagenet.pt``
when neither environment override is set.

Inspect a named-weight configuration:

.. code-block:: bash

   rl-garden-train-online sac \
     --env-id PushCube-v1 --obs.rgb base_camera --encoder.backbone resnet10 \
     --encoder.pretrained-weights resnet10-imagenet \
     --print-config

Printing the configuration does not load the weights.
The file must exist when the encoder is constructed.
The name is supplied without the ``.pt`` suffix.

A ResNet file may contain a raw state dictionary or a dictionary whose
``state_dict`` entry contains it.
Keys must match the rl-garden ResNet encoder.
Loading allows designated pooling or bottleneck heads to remain fresh.
Missing required backbone keys still raise an error despite non-strict loading.

Convert torchvision-style keys with
``tools/conversion/convert_resnet_checkpoint.py``.
Its input option selects your source file, its architecture option selects
the ResNet variant, and its output option selects the converted file.
Use the cache path above as that output destination.
The converter is a source-checkout utility, not a training registry command.

ACT names use the ``act/`` kind directory under the same cache root.
The accepted containers include ``ema_agent`` or ``agent`` state
dictionaries with optional ``norm_stats``, or a raw ACT state dictionary.
These are ACT base-policy weights, not interchangeable ResNet weights.
