ManiSkill
=========

Install and select
------------------

Install the ``maniskill`` extra with ``pip install -e ".[maniskill]"``.
You also need a working simulator/rendering runtime and the task's assets.
The package imports ManiSkill only when you construct an environment.
See :doc:`/start/installation` for the core installation and CUDA setup.

Select ``--env-backend maniskill`` and a ManiSkill task ID.
The command below uses PushCube and checks configuration without loading assets.

Namespace flags
---------------

These defaults come from the SAC preflight configuration with this backend
selected and no backend overrides. The command may override a default explicitly.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--maniskill.sim-backend``
     - ``gpu``
     - Simulation backend passed to the task.
   * - ``--maniskill.render-backend``
     - ``gpu``
     - Renderer backend passed to the task.
   * - ``--maniskill.reward-mode``
     - ``null``
     - Leave the task reward mode unchanged.
   * - ``--maniskill.env-kwargs-json``
     - ``{}``
     - Additional JSON keyword arguments for task construction.

Validated command
-----------------

.. code-block:: bash

   rl-garden-train-online sac --env-backend maniskill \
     --env-id PushCube-v1 --print-config

Observations
------------

State-only tasks are wrapped as ``{"state": tensor}``.
For visual observations, request the task's sensor names with ``--obs.rgb``
and ``--obs.depth``. The per-camera wrapper emits ``rgb_<camera>`` as uint8
and ``depth_<camera>`` as float32, rather than concatenating cameras.
For example, the ``base_camera`` sensor maps to ``rgb_base_camera``.
The optional state vector is flattened from the task's agent and extra data.

Image sizing comes from the shared observation request.
Frame stacking applies to requested image keys; state stays single-frame.
See :doc:`/tutorials/03-observations-and-encoders` for encoder selection.

Devices
-------

The factory returns a ManiSkill vector environment, with torch tensors on
the simulator's device. The default GPU simulator path stays tensor-native.
It does not pass actions through the CPU vector adapter.
Reward scaling is applied after the vector environment records raw episode
metrics, so scaled training rewards do not replace raw episode returns.

Known limits
------------

Camera names must exist on the selected task.
This backend rejects extra state sources requested through the shared
observation namespace. A state-only request cannot stack image frames.
The task and installed ManiSkill version determine control modes and assets;
static inspection does not establish runtime compatibility.

For an actual training walkthrough, use :doc:`/tutorials/02-online-rl`.
For environment-authoring alternatives, return to :doc:`index`.
