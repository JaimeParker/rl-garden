MuJoCo
======

Install and select
------------------

There is no dedicated ``mujoco`` extra in rl-garden.
Install Gymnasium's MuJoCo dependencies externally, for example with
``pip install "gymnasium[mujoco]"``, alongside the core package.
Rendering also needs an appropriate OpenGL runtime.

Select ``--env-backend mujoco`` and a registered Gymnasium task ID.
The factory also imports rl-garden's MuJoCo task registrations inside each
worker, so custom task registration survives spawned worker processes.

Namespace flags
---------------

These defaults come from the SAC preflight configuration with this backend
selected and no backend overrides. The command may override a default explicitly.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--mujoco.device``
     - ``cpu``
     - Device for the adapter output tensors.
   * - ``--mujoco.env-kwargs-json``
     - ``{}``
     - JSON keyword arguments forwarded to gym.make.
   * - ``--mujoco.vectorization``
     - ``sync``
     - Use sync or async Gymnasium vectorization.

Validated command
-----------------

.. code-block:: bash

   rl-garden-train-online sac --env-backend mujoco \
     --env-id InvertedPendulum-v4 --print-config

Observations
------------

The registered CLI backend accepts state-only observation requests.
Benchmark tasks with a flat Box observation are wrapped under ``state``.
In-tree custom tasks may already produce that Dict layout.
The vector adapter converts floating-point observations to float32.

Requests for RGB, depth, extra state, image size, or image frame stacking
are rejected by backend configuration resolution.
The lower-level custom-task scaffold has camera code, but this does not
make camera requests available through the current registered backend.

Devices
-------

Physics runs in CPU Gymnasium environments.
The adapter explicitly sends actions to NumPy on CPU and converts results
to tensors on the requested device. A CUDA adapter device changes transfer
placement, not the simulator implementation.
Sync vectorization runs instances in one process; async uses worker processes.
SAME_STEP autoreset preserves terminal observations for bootstrapping.

Known limits
------------

Task IDs and kwargs must be supported by the installed Gymnasium version.
The factory defaults ``MUJOCO_GL`` to ``egl`` when the variable is absent.
Set the appropriate renderer environment before starting a process if your
host requires another OpenGL backend.
Static inspection does not import MuJoCo or test rendering.

Use :doc:`mujoco-warp` for the separate native GPU task implementation.
Use :doc:`/guides/gymnasium-custom-envs` to register an ordinary environment.
See :doc:`index` for the common device and episode-boundary contract.
