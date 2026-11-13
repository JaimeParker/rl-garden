Quickstart
==========

Install the source checkout as described in :doc:`installation`.
This short run checks the training pipeline; it does not establish policy quality.

Train SAC on Pendulum
---------------------

Run from the repository root. Hiding CUDA makes the agent use CPU as well
as the environment and replay buffer. Thread limits keep this small run inexpensive.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rl-garden-train-online sac \
     --env-backend gymnasium --env-id Pendulum-v1 --num-envs 1 --num-eval-envs 1 \
     --gymnasium.device cpu --buffer-device cpu --total-timesteps 2000 \
     --learning-starts 100 --training-freq 4 --batch-size 64 --buffer-size 10000 \
     --hidden-dim 64 --gamma 0.99 --eval-freq 0 --log-freq 500 \
     --no-capture-video --exp-name sac-pendulum

Evaluate the saved policy:

.. code-block:: bash

   rl-garden-eval --checkpoint-path runs/sac-pendulum/checkpoints/final.pt \
     --device cpu --num-eval-envs 1 --num-eval-episodes 5

Try a GPU simulator
-------------------

With the ManiSkill extra and a working CUDA simulator installed, select
PushCube. Run this in a shell where CUDA is visible.

.. code-block:: bash

   rl-garden-train-online sac --env-backend maniskill --env-id PushCube-v1 \
     --num-envs 16 --num-eval-envs 1 --buffer-device cuda --total-timesteps 2000 \
     --learning-starts 100 --eval-freq 0 --no-capture-video --exp-name sac-pushcube

Use the Python API
------------------

This tested example explicitly selects CPU and closes the environment after training.
Its ``main(argv)`` accepts ``--smoke`` for a 128-step check.

.. literalinclude:: ../../examples/tutorials/ch02_sac_pendulum.py
   :language: python
   :start-after: # docs: start train
   :end-before: # docs: end train

What you ran
------------

The commands above train SAC, evaluate its checkpoint, and select a GPU backend.
For Pendulum, expect ``runs/sac-pendulum/config.json`` and
``runs/sac-pendulum/checkpoints/final.pt``; TensorBoard events live under
``runs/Pendulum-v1/sac-pendulum/``.
Watch ``losses/actor_loss``, ``losses/critic_loss`` and the console's
``return`` as training advances.
A short run is a functionality check, so returns need not improve.

Continue with :doc:`concepts` and :doc:`/tutorials/02-online-rl`.
