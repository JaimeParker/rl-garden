Checkpoints and evaluation
==========================

Goal
----

You save a SAC run, resume with its replay buffer, and evaluate the result.
You also verify a Python save/load round trip by comparing deterministic actions.
A checkpoint is useful only when you can reconstruct a compatible agent.

Prerequisites
-------------

Complete the CPU portion of :doc:`02-online-rl`.
Use the core package and Pendulum.
Run from the repository root.
The commands explicitly use CPU and short tutorial budgets.

Read :doc:`/guides/checkpoint` for the checkpoint format.
Keep the original run directory available while resuming.
Use a distinct experiment name for the continuation.

Steps
-----

Decide whether replay belongs in the handoff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A SAC checkpoint contains learned state and training counters.
Replay transitions are stored in a separate sibling file when requested.
Saving model weights alone does not preserve the replay distribution.

For this continuation exercise, save replay explicitly.
The command opts in rather than relying on a default.
See :ref:`checkpoint-replay-buffer`
for the scenario table.

A replay snapshot can be much larger than the model.
The small buffer here keeps the exercise bounded.
Choose storage deliberately for a real training job.

Save a short Pendulum run
^^^^^^^^^^^^^^^^^^^^^^^^^

Train to a total budget of 200 transitions:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online sac \
     --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 \
     --num-eval-envs 1 --buffer-device cpu --buffer-size 1000 \
     --batch-size 32 --hidden-dim 32 --learning-starts 32 \
     --training-freq 4 --gamma 0.99 --eval-freq 0 --log-freq 100 \
     --no-capture-video --total-timesteps 200 --save-replay-buffer \
     --exp-name chapter06-save

The experiment writes its configuration under
``runs/chapter06-save/config.json``.
The final model is
``runs/chapter06-save/checkpoints/final.pt``.
The replay snapshot is a separate file referenced by model metadata.

Keep both checkpoint files together.
Moving the model alone breaks a referenced replay load.
The checkpoint embeds the run configuration, so evaluation needs no other file.

The command disables periodic evaluation for this short exercise.
It still collects experience and performs updates before the final save.
A checkpoint produced before learning would not test that path.

Resume to a larger total budget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reconstruct the same task and architecture, then load the saved state:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online sac \
     --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 \
     --num-eval-envs 1 --buffer-device cpu --buffer-size 1000 \
     --batch-size 32 --hidden-dim 32 --learning-starts 32 \
     --training-freq 4 --gamma 0.99 --eval-freq 0 --log-freq 100 \
     --no-capture-video --total-timesteps 400 --save-replay-buffer \
     --load-checkpoint runs/chapter06-save/checkpoints/final.pt \
     --load-replay-buffer --exp-name chapter06-resume

The second budget is a total target of 400 transitions.
It is not a request for 400 additional transitions.
The saved global step is restored before the learning loop compares it
with the new total target.

Use a target greater than the saved step to continue training.
Keep the model architecture compatible.
Changing a hidden width is not a harmless logging override.

The CLI replay-loading default is False.
This command opts in explicitly.
The Python ``load`` method has its own default, so use an explicit
argument there too when replay state matters.

The SAC builder forwards the CLI choice to the load call.
It does not infer replay loading merely from a sibling file's presence.
See :ref:`checkpoint-loading` for the loading contract.

Verify deterministic actions in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tested example constructs matching CPU agents.
It evaluates both on the same observation tensor.
It does not compare observations from two different resets.

.. literalinclude:: ../../examples/tutorials/ch06_save_resume.py
   :language: python
   :start-after: # docs: start roundtrip
   :end-before: # docs: end roundtrip

The comparison uses deterministic actions with both policies in evaluation mode.
It checks exact equality before the restored agent learns again.
The example then verifies that a continuation performs more updates.

Its helper constructs the same architecture for both agents.
The smoke mode trains 50 steps before saving.
By default the example uses a temporary directory and removes it afterward.
An output-directory argument retains the files when you need to inspect them.

Matching actions is evidence that this model state round-trips.
It is not a guarantee of an identical future training trajectory.
The learning loop resets the environment when a new learn call begins.
Simulator state and every external random source are not restored by this check.

Inspect the metadata boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model metadata identifies the algorithm class and observation/action spaces.
It also records hyperparameters used to reconstruct the agent.
The saved training state contains step and update counters.

For the format inventory, use
:ref:`checkpoint-what-gets-saved`.
Keep the distinction between metadata and model tensors clear.
The embedded run configuration provides the surrounding run settings.

Importable actor-factory paths can be recorded in hyperparameters.
Evaluation must be able to import the package providing those paths.
A local function or lambda is not a portable reconstruction recipe.
See :doc:`/guides/extending` for downstream package contracts.

Do not replace a saved actor factory with another callable merely because
it returns actions of the same shape.
The checkpoint parameters still belong to the original architecture.

Evaluate with automatic algorithm selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the resumed checkpoint:

.. code-block:: bash

   rl-garden-eval --checkpoint-path runs/chapter06-resume/checkpoints/final.pt \
     --algorithm auto \
     --device cpu \
     --num-eval-envs 1 \
     --num-eval-episodes 5

The automatic selector matches the checkpoint algorithm class to a registry entry.
The evaluator uses that entry's builder to reconstruct the agent.
It then loads weights for inference without replay or optimizer state.
It creates one evaluation environment and no replay storage.
The datasets and other files the run was trained with are not needed.
``--device cpu`` puts the policy on the CPU.

The run configuration is embedded in the checkpoint, so the environment and
observation settings travel with the file.
An explicit ``--config-path`` replaces it.
A checkpoint without one falls back to ``config.json`` in its run directory.
See :ref:`checkpoint-evaluating-a-checkpoint`
for precedence and override details.

Keep the task unchanged when checking a resumed policy.
A five-episode result is a short functionality check.
It is not a statistically established improvement over the saved policy.

Handle a failed load deliberately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strict loading checks compatibility rather than reshaping arbitrary weights.
A mismatched algorithm or observation space is a reason to inspect the run.
Do not disable checks to force an unrelated checkpoint into an agent.

If a checkpoint explicitly references a missing replay file, loading it
with replay enabled raises an error.
Recover that file or deliberately choose a weights-only continuation.
Those choices produce different training conditions.

Old observation-format checkpoints are unsupported.
See :ref:`checkpoint-compatibility`.
This tutorial does not provide a migration path.

What you ran
------------

.. note::

   You ran the save command, the resume command, and the evaluation command
   shown above, in that order.
   Expect configuration and checkpoint directories under both
   ``runs/chapter06-save/`` and ``runs/chapter06-resume/``.
   Keep each referenced replay snapshot beside its model.
   Watch ``losses/actor_loss``, ``losses/critic_loss``
   and the console ``return`` during the training portions.

The original run remains available for comparison.
The resumed run has its own configuration and final checkpoint.
Your Python round-trip check separately verifies deterministic action equality.

Next
----

Use :doc:`03-observations-and-encoders` before resuming a visual policy
with a different observation request.
The upcoming :doc:`05-offline-to-online` explains a different workflow:
starting online learning from an offline-trained agent.
