Offline pretraining
=======================

Goal
--------

You write a small H5 dataset, train BC and IQL, and evaluate a saved policy.
You learn the transition layout before choosing a larger dataset.
The complete exercise runs on CPU without a robotics simulator.

Prerequisites
-----------------

Use the core installation from :doc:`/start/installation`.
It includes H5 support through h5py.
Run from the repository root with the package installed.
Read :doc:`03-observations-and-encoders` for observation names.
Read :doc:`06-checkpoints-and-evaluation` if checkpoint reconstruction is new to you.

This exercise uses synthetic Pendulum-shaped arrays.
They have the correct dimensions but do not follow Pendulum dynamics.
The actions are random, not expert demonstrations.
A successful run checks the data and training plumbing, not policy quality.

Steps
---------

Understand one trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The writer in ``rl_garden/datasets/wsrl_generation.py`` defines the producer format.
Its ``WSRLTrajectoryWriter.write_episode`` writes one complete episode per group.
The generic reader is ``rl_garden/buffers/h5_dataset.py``.
Its specification inference and transition loader enforce the consumer contract.

Use top-level groups named ``traj_0``, ``traj_1``, and so on.
For an episode with T transitions, store observations for T + 1 instants.
Store one action and reward for each transition.
The final observation supplies the last next observation.

.. code-block:: text

   pendulum.h5
     traj_0/
       obs          float32 (65, 3)
       actions      float32 (64, 1)
       rewards      float32 (64,)
       terminated   bool    (64,)
       truncated    bool    (64,)
     traj_1/
       ... same fields and shapes ...

The example has two trajectories with 64 transitions each.
The three state values have the shape of cosine, sine, and angular velocity.
Its final transition is a truncation, not a task termination.
Keep these flags separate so time limits do not become terminal targets.

The reader also accepts an explicit ``next_obs`` array alongside observations.
In that form both arrays contain T rows.
The accepted aliases include ``observations`` and ``next_observations``.
The writer convention above avoids needing those alternatives.
Keep all transition fields aligned; do not rely on the loader shortening arrays.

Represent dictionary observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For state-only data, ``obs`` can be a single H5 dataset.
The loader exposes that source as a dictionary containing ``state``.
For multimodal data, make ``obs`` an H5 group instead.
Each child is an array with the same leading time dimension.

Child keys follow the observation vocabulary:
``state``, ``state_<name>``, ``rgb_<cam>``, and ``depth_<cam>``.
Nested observation groups are rejected by the generic H5 specification reader.
Arbitrary camera names without a modality prefix are rejected too.
Rename producer fields deliberately or use a producer-specific dataset backend.

For example, a dictionary can contain ``state`` and ``rgb_front`` arrays.
Keep image shapes consistent with the encoder you choose.
See :ref:`configuration-observation-and-encoder-configuration`
for observation selection and encoder configuration.
This chapter keeps flat state data so no renderer or image model is needed.

Write the tiny dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

The example writes exactly the layout described above.
It also records that the file is a synthetic format fixture.
Run it in a scratch output directory; rerunning replaces ``pendulum.h5`` there.

.. literalinclude:: ../../examples/tutorials/ch04_offline_bc.py
   :language: python
   :start-after: # docs: start dataset
   :end-before: # docs: end dataset

.. code-block:: bash

   python examples/tutorials/ch04_offline_bc.py --smoke --output-dir runs/chapter04-data

The script writes the file and performs four small BC updates.
It leaves the H5 file available for the CLI commands below.
No checkpoint is written by this Python smoke example.
The CLI runs create their own saved models.

Connect data to the Python algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Python path infers spaces before constructing an offline environment spec.
That spec describes observations and actions without a live rollout environment.
The loader fills the agent's replay buffer from the file.
Learning then samples the stored transitions.

.. literalinclude:: ../../examples/tutorials/ch04_offline_bc.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

Action dimensions come from the file, but action bounds are explicit inputs.
Pendulum uses bounds of -2 and 2.
Use those bounds even though the synthetic actions occupy a smaller interval.
The CLI below passes them explicitly for both algorithms.
Do not infer the task's legal bounds from the extrema of a small sample.

Train a behavior-cloning baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BC fits the recorded actions with an actor log-likelihood objective.
It does not learn a critic from these rewards.
Run a short CLI baseline on the same file:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-offline \
     bc --offline-dataset runs/chapter04-data/pendulum.h5 --dataset-backend h5 \
     --env-backend gymnasium --env-id Pendulum-v1 --device cpu --buffer-device cpu \
     --buffer-size 256 --batch-size 16 --action-low -2 --action-high 2 \
     --num-offline-steps 20 --eval-freq 0 --log-freq 10 --exp-name chapter04-bc

The budget counts offline update steps.
There are no environment interactions in this training loop.
The environment identity is retained for checkpoint evaluation.
Periodic evaluation is disabled for the short training run.

The run records its effective configuration before and after construction.
The final filename identifies the offline algorithm.
Keep the run directory when you move the checkpoint.
See :ref:`checkpoint-what-gets-saved` for the checkpoint contents.

Train IQL on the same transitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IQL uses rewards and next observations in addition to the recorded actions.
Its update includes a critic, an expectile value objective, and weighted actor fitting.
Use the same dataset and action bounds to isolate the algorithm change:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-offline \
     iql --offline-dataset runs/chapter04-data/pendulum.h5 --dataset-backend h5 \
     --env-backend gymnasium --env-id Pendulum-v1 --device cpu --buffer-device cpu \
     --buffer-size 256 --batch-size 16 --action-low -2 --action-high 2 --n-critics 2 \
     --critic-subsample-size 2 --num-offline-steps 20 --eval-freq 0 --log-freq 10 \
     --exp-name chapter04-iql

The critic count is deliberately small for this CPU exercise.
Neither this update budget nor the synthetic data is a benchmark recipe.
Look for finite losses and a written checkpoint before scaling up.
See :doc:`/guides/offline-acceleration` when throughput becomes the next concern.

Inspect a conservative alternative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cal-QL is also registered in the offline phase.
Inspect its configuration against the same file without starting another training run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-offline \
     calql --offline-dataset runs/chapter04-data/pendulum.h5 --buffer-device cpu \
     --action-low -2 --action-high 2 --print-config

This only checks configuration; it does not load transitions or estimate policy quality.
Continue to the next chapter for a Cal-QL preset and the WSRL handoff.

Evaluate the saved IQL policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation creates a real Pendulum environment.
It tests checkpoint reconstruction and an episode rollout.
It does not turn the synthetic data into a useful control dataset.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-eval \
     --checkpoint-path runs/chapter04-iql/checkpoints/iql_offline_pretrained.pt \
     --algorithm auto --device cpu --num-eval-envs 1 --num-eval-episodes 2

Keep the package version and environment dependencies compatible with training.
See :ref:`checkpoint-evaluating-a-checkpoint` for evaluation options.
A low return here is expected and is not a dataset-format failure.

Replace synthetic data with a real source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simulator demonstrations, ``examples/generate.py`` provides ``wsrl_dataset``
and ``rgb_demos`` collection subcommands.
The former collects from SAC checkpoint sources in ManiSkill.
The latter collects RGB demonstrations with a state-policy oracle.
Follow :doc:`/guides/wsrl-reproduction` for the collection workflow.
Those commands need simulator dependencies and compatible trained checkpoints.

For public data, use the source's matching dataset backend.
See :doc:`/guides/ogbench-integration` and :doc:`/guides/d4rl-legacy-expansion`.
See :doc:`/backends/minari` for dataset IDs and recovered live environments.
The Minari install extra is listed in :doc:`/start/installation`.
A dataset identifier for one backend is not an H5 filesystem path.
Do not rename an unrelated H5 file and assume it satisfies this contract.

What you ran
----------------

.. note::

   You ran the dataset command, the BC and IQL commands, and IQL evaluation above.
   These exact argv lists are covered by the tutorial command/example tests.
   The training runs use 20 updates each; the example's smoke mode uses four.
   Watch ``losses/actor_loss`` and ``losses/critic_loss`` in the IQL logs.
   BC has an actor loss and no critic loss.
   Small stochastic batches need not produce monotonically falling losses.

.. code-block:: text

   runs/
     chapter04-data/pendulum.h5
     chapter04-bc/config.json
     chapter04-bc/checkpoints/bc_offline_pretrained.pt
     chapter04-iql/config.json
     chapter04-iql/checkpoints/iql_offline_pretrained.pt

TensorBoard event files use the logger's separate group/run directory.
The paths above identify configuration and model artifacts.

Next
--------

Continue to :doc:`05-offline-to-online` to carry a compatible checkpoint
into a live environment and choose how replay changes at the phase boundary.
