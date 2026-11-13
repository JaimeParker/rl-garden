Bring your own environment
==============================

Goal
--------

You register a one-dimensional Gymnasium environment from your own module.
You train SAC on it through the existing Gymnasium backend.
You then run the custom PointReach backend and identify when that extra layer helps.

Prerequisites
-----------------

Complete the CPU run in :doc:`02-online-rl`.
Use the core installation and run from the repository root.
No renderer, simulator asset, or GPU is needed here.
The example module must be importable in the training process.

Read :doc:`/guides/gymnasium-custom-envs` for the complete backend settings.
The tutorial uses its ordinary single-instance environment path.
It does not require a new backend registration or a fork of rl-garden.
Keep your actual task in your own installable package when leaving the tutorial.

Steps
---------

Write the task's single-instance interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example's observation is one position value.
Its action is one bounded displacement control.
The reward is negative distance from zero.
An episode terminates when the point gets close enough to zero.
A registered time limit handles episodes that take too long.

.. literalinclude:: ../../examples/tutorials/ch07_my_env.py
   :language: python
   :start-after: # docs: start environment
   :end-before: # docs: end environment

The reset method calls its parent to initialize seeded randomness.
It returns an observation and an information dictionary.
The step method returns the five Gymnasium results.
Task termination and time-limit truncation stay separate.
This lets the adapter preserve the episode boundary information used by replay.

Match every returned array to its declared space.
The example returns float32 arrays with a stable shape.
It copies the observation so later mutation does not change an earlier result.
It clips controls and positions to the spaces it advertises.
These details matter before any neural-network choice enters the picture.

The task itself uses NumPy because it is a CPU Gymnasium environment.
The backend handles vectorization and conversion to the algorithm's torch interface.
You do not add an environment dimension inside the task.
You do not return a batch of independent episodes from this single-instance API.

Register it on import
^^^^^^^^^^^^^^^^^^^^^^^^^

Registration connects a stable environment ID to the implementation.
Place it at module scope so importing your module performs the registration.
Keep training inside the script's main guard so imports do not launch a run.

.. literalinclude:: ../../examples/tutorials/ch07_my_env.py
   :language: python
   :start-after: # docs: start registration
   :end-before: # docs: end registration

The guard makes repeated imports safe within the tutorial test process.
The example supplies a class as its registration entry point.
In an installed package, a module-and-class string can also locate the implementation.
Give your task a distinct ID so it does not collide with an existing task.

Registration sets a 32-step time limit for this exercise.
The task's own step method does not count that limit.
This separates success termination from the wrapper's truncation.
Keep the same separation when adapting a real task.

Check the Python composition path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script creates a vectorized wrapper through the existing factory.
It then constructs SAC with CPU model and replay devices.
A finally block closes the environment even if training raises an error.

.. literalinclude:: ../../examples/tutorials/ch07_my_env.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

Run the small example:

.. code-block:: bash

   python examples/tutorials/ch07_my_env.py --smoke

The smoke budget is 64 transitions.
It reaches optimizer updates after initial collection.
The example returns the agent for tests and does not save a checkpoint.
The CLI run below demonstrates artifact creation separately.

Import the task through the CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tell the Gymnasium backend which module performs registration:

.. code-block:: bash

   PYTHONPATH=examples/tutorials CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online \
     sac --env-backend gymnasium --env-id TutorialLineReach-v0 \
     --gymnasium.import-modules ch07_my_env --num-envs 1 --num-eval-envs 1 \
     --buffer-device cpu --buffer-size 512 --batch-size 16 --hidden-dim 32 \
     --learning-starts 16 --training-freq 4 --total-timesteps 64 --eval-freq 0 \
     --log-freq 32 --no-capture-video --exp-name chapter07-gymnasium

The module path is a string resolved by Python's import machinery.
It points at the source module, made importable by the command's PYTHONPATH.
The backend validates every dotted module component as a Python identifier.
For your package, replace it with your package's registration module.
Install that package in the same environment as the training command.

The backend imports registration modules in the worker before calling Gymnasium.
That matters when workers start fresh interpreters.
An import performed only in a separate interactive shell does not register tasks
inside a new training process.
Keep the module name in the run configuration so the task can be reconstructed.

Static configuration inspection checks the flag spelling and selected values.
It does not prove that the module imports or that a reset succeeds.
The live CPU smoke exercises import, construction, rollout, update, and save.
Use those separate checks when debugging your own environment.

Keep observations compatible with the encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task starts with a flat Box observation.
The algorithm-facing wrapper gives state observations a dictionary representation.
For richer observations, choose canonical modality keys deliberately.
Do not flatten images into the state vector as an accidental workaround.

See :doc:`03-observations-and-encoders` for state and image composition.
See :ref:`configuration-observation-and-encoder-configuration` for the shared contract.
The Gymnasium guide describes observation-key mapping when your task has native names.
Use that guide's settings reference instead of copying a second option table here.

Understand the custom backend example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The directory ``rl_garden/envs/custom/`` contains a small dedicated backend example.
Its config defines task settings and the wrapper's device.
Its PointReach task is a two-dimensional point mass moving toward a target.
Its factory builds Gymnasium vector environments and the torch adapter.

The factory also records episode statistics.
Those statistics feed the training and evaluation paths.
When writing a backend, preserve that behavior alongside reset and step conversion.
Returning tensors alone is not the entire environment contract.

Run the provided task through its registered backend:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online \
     sac --env-backend custom --env-id PointReach-v0 --num-envs 1 --num-eval-envs 1 \
     --buffer-device cpu --buffer-size 512 --batch-size 16 --hidden-dim 32 \
     --learning-starts 16 --training-freq 4 --total-timesteps 64 --eval-freq 0 \
     --log-freq 32 --no-capture-video --exp-name chapter07-custom

The task exposes state containing position, velocity, and target coordinates.
It uses two continuous action components.
The factory currently constructs PointReach directly.
Changing the environment ID alone does not turn this factory into a general task loader.
Use the generic Gymnasium backend for arbitrary registered Gymnasium tasks.

Choose whether a new backend is warranted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stay with Gymnasium for ordinary single-instance tasks from an importable module.
Consider a dedicated backend for a simulator that already batches environments on GPU.
Also consider one when runtime initialization is process-global or shared across tasks.
Named simulator-wide configuration can be another reason for that boundary.

A dedicated backend owns construction and adaptation, not the learning algorithm.
Keep optional simulator imports lazy so configuration inspection remains usable.
Keep device transfers explicit when adapting CPU data for CUDA algorithms.
See :doc:`/guides/extending` for the backend registration contract.

The custom example illustrates the adapter boundary with a small CPU task.
It is not a requirement to copy that directory for every environment.
Start from the smallest integration that your task actually needs.
Keep task reward semantics separate from training hyperparameters.

What you ran
----------------

.. note::

   You ran the Python smoke and the two 64-transition SAC commands above.
   The Gymnasium command loads your module; the custom command loads PointReach.
   Watch ``losses/actor_loss`` and ``losses/critic_loss`` once updates start.
   Short runs test the interface and do not establish that either task is solved.

.. code-block:: text

   runs/
     chapter07-gymnasium/config.json
     chapter07-gymnasium/checkpoints/final.pt
     chapter07-custom/config.json
     chapter07-custom/checkpoints/final.pt

Preserve the importable task module with the checkpoint's environment configuration.
A saved model cannot recreate code that is absent from the evaluation environment.
Use a new experiment name when comparing a task implementation change.

Next
--------

Continue to :doc:`09-extending-from-your-package` for package-level registration.
The tutorial :doc:`08-bring-your-own-policy` covers changing model components
while retaining an existing algorithm.
