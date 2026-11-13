Online reinforcement learning
=============================

Goal
----

You train SAC and PPO on Pendulum, then run SAC on a GPU simulator.
You can locate the effective configuration, metrics and final checkpoint.
The short budgets check the training pipeline rather than convergence.

Prerequisites
-------------

Complete :doc:`01-configuration-and-cli`.
Install the core package for the CPU runs.
Install ManiSkill and configure a working CUDA simulator for PushCube.
Run commands from the repository root.

Keep the CPU and GPU device choices explicit.
The CPU examples hide CUDA from their process.
The GPU command needs CUDA visible in its shell.

Steps
-----

Collect experience with SAC
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the small Pendulum experiment:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online sac \
     --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 \
     --num-eval-envs 1 --buffer-device cpu --total-timesteps 2000 \
     --learning-starts 100 --training-freq 4 --batch-size 64 \
     --buffer-size 10000 --hidden-dim 64 --gamma 0.99 \
     --eval-freq 0 --log-freq 500 --no-capture-video \
     --exp-name chapter02-sac

SAC collects transitions and stores them in replay.
After the warm-up budget, it samples batches for gradient updates.
The actor chooses actions and the critics estimate their values.

The command reduces the network and replay sizes for a short run.
Those settings are tutorial choices, not recommended benchmark settings.
The discount factor is also set explicitly.

The environment runs on CPU.
The replay storage is explicitly on CPU.
Hiding CUDA makes automatic agent-device selection choose CPU too.
An environment's device setting alone does not select the agent device.

Read the training output
^^^^^^^^^^^^^^^^^^^^^^^^

The console reports progress and rollout statistics.
A completed episode contributes an episode return.
An interval without a completed episode can show an unavailable return.
Pendulum does not provide a task-success metric.

For SAC, inspect ``losses/actor_loss`` and ``losses/critic_loss``.
These are TensorBoard metric names from the logger's namespace mapping.
The console's ``return`` summarizes completed episodes.

A changing loss confirms that updates are being logged.
It does not establish that a policy solves the task.
Use evaluation over complete episodes for that question.

Compare PPO's rollout structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PPO collects a rollout and updates from that rollout.
Its minibatches revisit the collected policy data during the update.
This differs from SAC's persistent replay storage.

Run a small PPO experiment on the same CPU environment:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online ppo \
     --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 \
     --num-eval-envs 1 --num-steps 32 --num-minibatches 2 \
     --update-epochs 2 --total-timesteps 2048 --gamma 0.99 \
     --eval-freq 0 --no-capture-video --exp-name chapter02-ppo

Here the rollout length, minibatch count and update epochs are explicit.
The budget fits complete rollouts for this one-environment example.
Do not compare raw update counts between PPO and SAC as equal work.

The public Python example exposes the same construction pattern:

.. literalinclude:: ../../examples/tutorials/ch02_ppo_pendulum.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

The example closes its environment even if training raises.
Its smoke mode reduces the budget while still performing updates.
The tests check finite policy parameters after learning.

For the SAC counterpart, see the tested example in
:doc:`/start/quickstart`.
Both examples use explicit CPU devices in Python.

Move the environment to PushCube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the checked-in SAC preset and override the task and smoke budget:

.. code-block:: bash

   rl-garden-train-online sac \
     --config configs/online/sac_state.yaml --env-id PushCube-v1 \
     --num-envs 16 --num-eval-envs 1 --total-timesteps 2000 \
     --learning-starts 100 --training-freq 4 --buffer-size 10000 \
     --batch-size 64 --hidden-dim 64 --buffer-device cuda \
     --eval-freq 0 --log-freq 500 --no-capture-video \
     --exp-name chapter02-pushcube

The preset supplies ManiSkill configuration.
The command selects PushCube and keeps replay on CUDA.
Normal rollout tensors, samples and learning stay on the GPU path.

Use a GPU with capacity available for your run.
Do not hide CUDA as you did for Pendulum.
The same algorithm interface does not imply identical simulator setup.

The small transition budget may finish before a useful policy emerges.
A successful run verifies construction, collection, updates and saving.
Increase budgets only after those operations work.

To select PPO on the same GPU task, use its rollout settings explicitly:

.. code-block:: bash

   rl-garden-train-online ppo --env-backend maniskill --env-id PushCube-v1 \
     --num-envs 16 --num-eval-envs 1 --num-steps 32 --num-minibatches 2 \
     --update-epochs 2 --total-timesteps 2048 --eval-freq 0 \
     --no-capture-video --exp-name chapter02-pushcube-ppo

This command has its own experiment directory.
It uses the PPO rollout path rather than a SAC replay buffer.

Find the artifacts
^^^^^^^^^^^^^^^^^^

The explicit experiment name makes the run directory predictable.
The SAC CPU configuration is
``runs/chapter02-sac/config.json``.
Its final checkpoint is
``runs/chapter02-sac/checkpoints/final.pt``.

TensorBoard events use an environment grouping directory:
``runs/Pendulum-v1/chapter02-sac/``.
Point TensorBoard's log-directory setting at ``runs``
to browse these grouped events.
The configuration and checkpoint remain under the experiment directory.

PushCube uses ``runs/chapter02-pushcube/`` for its configuration
and checkpoints.
Its events are grouped under ``runs/PushCube-v1/``.
Changing the log directory moves both sets of artifacts.

Evaluate a saved CPU policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the SAC checkpoint produced above:

.. code-block:: bash

   rl-garden-eval --checkpoint-path runs/chapter02-sac/checkpoints/final.pt \
     --algorithm auto \
     --device cpu \
     --num-eval-envs 1 \
     --num-eval-episodes 5

The evaluator reads the run configuration and checkpoint metadata.
Automatic algorithm selection matches the saved algorithm class.
It rebuilds the agent before loading the weights.

The evaluation device is explicit here.
The saved environment configuration keeps Pendulum as the backend task.
The evaluator reports completed-episode returns.

See :ref:`checkpoint-evaluating-a-checkpoint`
for the configuration lookup and override rules.
Do not substitute another environment merely because its action shape matches.

Keep comparisons interpretable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Record seeds together with the effective configuration.
The same seed does not guarantee identical floating-point results
across devices or critic implementations.
See :doc:`/design/rng-numerical-determinism` for the numerical caveat.

Compare matching tasks, budgets and evaluation settings.
Keep the simulator and encoder choices with your result.
A fast smoke run and a learning-quality experiment answer different questions.

Additional presets include ``configs/online/sac_rgb_resnet.yaml``, ``configs/online/ppo_state.yaml``, ``configs/online/ppo_rgb.yaml`` and ``configs/online/drqv2_rgbd.yaml``.
Inspect a preset before selecting its simulator and camera configuration.

What you ran
------------

.. note::

   In one terminal sequence you ran the SAC and PPO Pendulum commands,
   the PushCube commands, and the saved-policy evaluation above.
   Each training run writes ``runs/<run_name>/config.json`` and
   ``runs/<run_name>/checkpoints/final.pt``.
   Watch ``losses/actor_loss``, ``losses/critic_loss`` and
   the console ``return`` for the SAC runs.

Keep the checkpoint together with the saved configuration.
A readable final checkpoint is the handoff to evaluation and resuming.

Next
----

Continue with :doc:`03-observations-and-encoders` to replace state-only
observations with camera inputs.
Use :doc:`06-checkpoints-and-evaluation` when you need to resume a run.
