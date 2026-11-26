Offline-to-online training
==============================

Goal
--------

You pretrain a WSRL-compatible checkpoint and continue it on live Pendulum.
You distinguish the saved model from the new run's configuration and replay policy.
You also inspect the checked-in WSRL and Cal-QL recipes before using real data.

Prerequisites
-----------------

Complete :doc:`04-offline-pretraining` and retain its synthetic H5 file.
Use the core package from the repository root.
The small handoff exercise runs entirely on CPU.
Read :doc:`06-checkpoints-and-evaluation` for the general save/load contract.

Use a WSRL checkpoint for the WSRL continuation below.
The BC and IQL checkpoints from chapter 04 are not interchangeable with it.
Matching action dimensions alone does not make two algorithms compatible.
The synthetic fixture checks mechanics and cannot establish learning performance.

Steps
---------

Inspect a recipe before constructing it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start with the checked-in WSRL preset:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-off2on \
     wsrl --config configs/off2on/wsrl.yaml --print-config

This is static inspection; it does not create a simulator or train a policy.
The preset targets state-based ManiSkill PickCube with multiple environments.
Its offline update budget is zero.
It supplies an online recipe, not a trained checkpoint or demonstration file.
Provide a compatible checkpoint before expecting a meaningful warm start.

The printed configuration separates selected inputs from derived values.
Keep it beside the data collection settings when planning a real run.
See :ref:`configuration-resolution-order` for how preset and CLI values combine.
A checkpoint does not replace this configuration-resolution step.

Inspect the Cal-QL recipe separately:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-off2on \
     calql --config configs/off2on/calql_halfcheetah_medium_v2_paper.yaml \
     --print-config

This preset assumes the legacy D4RL environment and dataset backends.
Its task and dataset identifier are ``halfcheetah-medium-v2``.
It is not a Pendulum H5 recipe and is not part of the CPU smoke below.
Install and configure that stack using :doc:`/guides/d4rl-legacy-expansion`.
The preset also selects external experiment logging, so review that before training.

Its comments document a locomotion adaptation rather than an exact paper recipe.
It enables the optional SARSA reference for the continuing task.
The preset retains offline data through mixed replay and keeps conservative loss online.
Do not transfer those choices to a new task without checking their assumptions.

Choose what must match across phases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The offline process constructs an agent from a dataset-derived specification.
The online process constructs a new agent around the live environment.
It then loads the checkpoint into that agent.
The builder does not infer your new CLI flags from the old command line.

Keep observation keys and feature dimensions compatible.
Keep action dimensions, bounds, network widths, and layer counts compatible.
Keep the critic ensemble shape compatible too.
A state-dict shape error is evidence that reconstruction differs.
See :ref:`checkpoint-compatibility` for the full compatibility contract.

Carry reward transformations and discounting deliberately.
Changing reward scale changes the meaning of the pretrained values.
Changing observation selection can change the network input entirely.
The commands below use the same state task and discount in both phases.
They also spell out the shared architecture instead of relying on phase defaults.

Select new run controls explicitly.
The experiment name, log location, live environment count, and online budget
belong to the continuation you are launching.
The model load restores learned state and counters.
It does not mean that every flag from the original invocation becomes active.

Pretrain the matching checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the WSRL entry in the offline registry on the chapter 04 file:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-offline \
     wsrl --offline-dataset runs/chapter04-data/pendulum.h5 --dataset-backend h5 \
     --env-backend gymnasium --env-id Pendulum-v1 --buffer-device cpu --buffer-size \
     256 --batch-size 16 --action-low -2 --action-high 2 --hidden-dim 32 \
     --actor-hidden-layers 2 --critic-hidden-layers 2 --n-critics 2 \
     --critic-subsample-size 2 --gamma 0.99 --utd 1 --no-use-compile \
     --num-offline-steps 4 --eval-freq 0 --log-freq 2 --exp-name chapter05-offline

This command runs four offline updates and writes
``runs/chapter05-offline/checkpoints/wsrl_offline_pretrained.pt``.
The small widths and critic ensemble keep the smoke inexpensive.
Compilation is disabled for this short CPU exercise.

The offline WSRL entry builds the compatible Cal-QL pretraining path.
The H5 trajectory boundaries supply the replay information used for return estimates.
That is why the synthetic writer marks episode ends explicitly.
For algorithm details, see :doc:`/design/wsrl-overview`.

The final checkpoint is saved without a replay snapshot in this exercise.
The online phase starts with empty replay, so it does not need that snapshot.
See :ref:`checkpoint-replay-buffer` for cases where saving replay is appropriate.
Keep this choice separate from whether weights are loaded.

Continue with live experience
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Override the simulator recipe with Gymnasium Pendulum and the matching architecture:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-off2on \
     wsrl --config configs/off2on/wsrl.yaml --env-backend gymnasium --env-id \
     Pendulum-v1 --num-envs 1 --num-eval-envs 1 --buffer-device cpu --buffer-size 512 \
     --batch-size 16 --hidden-dim 32 --actor-hidden-layers 2 --critic-hidden-layers 2 \
     --n-critics 2 --critic-subsample-size 2 --gamma 0.99 --utd 1 --training-freq 4 \
     --no-use-compile --num-offline-steps 0 --num-online-steps 64 --warmup-steps 16 \
     --learning-starts 16 --online-replay-mode empty --load-checkpoint \
     runs/chapter05-offline/checkpoints/wsrl_offline_pretrained.pt --eval-freq 0 \
     --log-freq 16 --no-capture-video --exp-name chapter05-online

The continuation loads the checkpoint before switching into online mode.
It uses a short warmup with the loaded policy.
The warmup configuration disables actor, critic, and encoder updates.
It sets random-action probability to zero for that initial phase.
After the phase ends, the normal online update path can run.

The online step budget is additional to the loaded global counter.
The off2on runner computes its target from the current counter plus the online budget.
This differs from an online-only total-timesteps resume target.
Do not compare raw step axes across the phases without noting their units.
Offline steps count updates; online steps count collected transitions.

This command requests zero further offline updates.
It therefore does not need the source H5 during continuation with empty replay.
Keep the file for reproducibility even though this particular run does not reload it.
A mixed-replay continuation has different data requirements.

Understand the replay boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With empty replay, the switch clears transitions while retaining the loaded model.
With append replay, existing transitions stay in the active buffer.
With mixed replay, the switch keeps an offline buffer and builds a new online buffer.
The requested offline ratio controls mixed-batch sampling.
See :doc:`/design/wsrl-overview` for the detailed choices.

The off2on runner can reload an offline dataset when continuing a checkpoint
with zero offline updates and an explicit dataset path.
That supports retaining source data without requiring a replay snapshot.
A dataset given to a fresh run with zero offline updates in the ``append`` or
``mixed`` mode is loaded too, with the setup the offline stage does on it
(observation statistics, SPOT's VAE, FAC's BC flow), as the reference
implementations keep their dataset for the online phase; ``empty`` ignores it.
A checkpoint saved after the switch continues in its saved replay mode,
which then decides where the dataset goes; see :ref:`checkpoint-loading`.
It is unnecessary for this empty-replay exercise.
Choose the mode before deciding which large artifacts to copy.

WSRL and Cal-QL share useful components but represent different online recipes.
The WSRL preset disables online CQL loss and discards offline replay.
The Cal-QL preset inspected above keeps conservative loss and mixes offline data.
These are configuration choices with different training distributions.
They are not alternate names for the same continuation command.

Check the handoff before scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First check that the load succeeds with the expected architecture.
Then check that the online phase starts and collects new transitions.
Finally check that updates occur after warmup and that the final checkpoint exists.
A command that only prints configuration proves none of those runtime properties.

Keep the offline and online effective configurations together.
Keep the source dataset's collection settings too.
For a full simulator workflow, use :doc:`/guides/wsrl-reproduction`.
Its dataset selection discussion is useful when replacing this fixture with real demonstrations.
The machine-specific paths in that guide are examples, not prerequisites here.

What you ran
----------------

.. note::

   You inspected both preset commands, then ran the offline and online WSRL pair above.
   The offline run performs four updates; the continuation adds 64 live transitions.
   Watch ``losses/actor_loss`` and ``losses/td_loss`` during updates.
   ``off2on/online_start_step`` records the phase boundary in the logger summary.
   Warmup intentionally has no optimizer updates.

.. code-block:: text

   runs/
     chapter05-offline/config.json
     chapter05-offline/checkpoints/wsrl_offline_pretrained.pt
     chapter05-online/config.json
     chapter05-online/checkpoints/final.pt

The continuation has a separate run directory so its configuration is inspectable.
The two checkpoints serve different points in the experiment history.
The preset inspection commands create neither directory.

Next
--------

Use :doc:`06-checkpoints-and-evaluation` to inspect and evaluate saved state.
Continue to :doc:`07-bring-your-own-environment` when you need a different live task.
