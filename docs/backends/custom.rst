Custom template
===============

Install and select
------------------

The custom template uses the core installation; no simulator extra is needed.
It provides a small Gymnasium point-reaching task and a backend adapter.
Select ``--env-backend custom`` and use ``PointReach-v0`` as the task label.

For your own single-instance Gymnasium environment, prefer registration
through the existing Gymnasium backend.
Follow :doc:`/tutorials/07-bring-your-own-environment` for that workflow.
The custom package demonstrates the pieces of a dedicated backend.

Namespace flags
---------------

These defaults come from the SAC preflight configuration with this backend
selected and no backend overrides. The command may override a default explicitly.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--custom.device``
     - ``cpu``
     - Device for output tensors from the vector adapter.

Validated command
-----------------

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= rl-garden-train-online sac \
     --env-backend custom --env-id PointReach-v0 \
     --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
     --buffer-size 512 --batch-size 16 --hidden-dim 32 \
     --learning-starts 16 --training-freq 4 --total-timesteps 64 \
     --eval-freq 0 --log-freq 32 --no-capture-video \
     --exp-name backend-custom

Observations
------------

PointReach emits a Dict with one float32 ``state`` vector:
position x/y, velocity x/y and target x/y.
Its two action components apply force to the point mass.
The task terminates on reaching the target; Gymnasium's TimeLimit bounds
the episode. There are no image observations or render modes.

The backend requires state-only observations and rejects requests for
cameras, extra state, image sizing or image frame stacking.
Read :doc:`/tutorials/03-observations-and-encoders` for the shared vocabulary.

Devices
-------

The task computes with NumPy on CPU.
A synchronous vector environment batches instances, and
``TorchVectorEnvAdapter`` converts actions and outputs explicitly.
The command hides CUDA so the algorithm and replay both stay on CPU.
The output device setting does not change where the task physics runs.
SAME_STEP autoreset preserves terminal observations and episode statistics.

Known limits
------------

The factory currently always constructs PointReach; changing the task
label does not dispatch to another Gymnasium registration.
It also does not forward its config seed into reset; do not rely on that
field alone for repeatable task resets through this template.
These are template limitations, not features of the Gymnasium backend.

The command is a short execution check, not a trained-policy recipe.
It writes a run configuration and final checkpoint beneath the log directory.
See :doc:`/tutorials/06-checkpoints-and-evaluation` for checkpoint handling.
Use :doc:`/guides/extending` for downstream backend registration.
