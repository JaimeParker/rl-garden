MuJoCo Warp
===========

Install and select
------------------

There is no dedicated ``mujoco_warp`` extra in rl-garden.
Install MuJoCo, MuJoCo Warp and its Warp runtime externally in a compatible
CUDA environment. The implementation imports ``mujoco``, ``mujoco_warp``
and ``warp`` when a task is constructed.
The core package alone does not supply these simulator dependencies.

Select ``--env-backend mujoco_warp``.
The ID below belongs to rl-garden's task registry, not Gymnasium's registry.
The registered camera variant is ``RlGarden-InvertedPendulum-Camera-Warp-v0``.

Namespace flags
---------------

These defaults come from the SAC preflight configuration with this backend
selected and no backend overrides. The command may override a default explicitly.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--mujoco-warp.device``
     - ``cuda:0``
     - Torch/Warp task device.
   * - ``--mujoco-warp.env-kwargs-json``
     - ``{}``
     - Additional keyword arguments for the task constructor.

Validated command
-----------------

.. code-block:: bash

   rl-garden-train-online sac --env-backend mujoco_warp \
     --env-id RlGarden-InvertedPendulum-Warp-v0 --print-config

Observations
------------

The state-only pendulum emits a ``state`` vector containing positions
and velocities. The camera task adds the requested ``rgb_main`` and
``depth_main`` keys. RGB is uint8 with three channels; depth has one
floating-point channel and represents planar distance in model units.
Zero depth denotes background with no geometry hit.

The shared observation request controls which camera buffers are enabled.
Image size is interpreted as height then width.
Frame stacking wraps the requested image keys after task construction.
See :doc:`/tutorials/03-observations-and-encoders` for the consumer contract.

Devices
-------

The task is already batched over worlds and returns torch tensors.
Physics uses MuJoCo Warp directly; Warp arrays are exposed to torch through
``wp.to_torch``. There is no Gymnasium vectorization or CPU NumPy adapter.
An ended world saves its true terminal observation, resets in place and
returns the new episode's observation from that same step.

Known limits
------------

Only tasks registered with ``register_mujoco_warp_task`` are available.
Gymnasium benchmark IDs cannot be substituted for these task IDs.
The current backend supports only the camera name ``main``.
State is required, and extra state sources are rejected.
A camera task needs an explicit camera in its model; there is no free-camera
fallback. The CPU scaffold's raw depth convention is different.

This command validates configuration only; it does not exercise CUDA physics.
Read :doc:`/design/autoreset-modes` when implementing another native task.
