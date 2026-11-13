Real-robot work
===============

Real-robot installation, hardware control and training orchestration live in
`real-world-rl <https://github.com/JaimeParker/real-world-rl>`_.
That downstream repository carries rl-garden as a submodule under
``rl-garden/`` and imports it as a library.
Start there when your task involves physical hardware.

- `Installation <https://github.com/JaimeParker/real-world-rl/blob/main/docs/tutorial/01-install.md>`_
- `rwrl CLI reference <https://github.com/JaimeParker/real-world-rl/blob/main/docs/cli.md>`_
- `Teleoperation and recording <https://github.com/JaimeParker/real-world-rl/blob/main/docs/teleop.md>`_
- `Device-layer roadmap <https://github.com/JaimeParker/real-world-rl/blob/main/docs/devices/robot-infra-roadmap.md>`_
- `HIL-SERL roadmap <https://github.com/JaimeParker/real-world-rl/blob/main/docs/hil-serl-roadmap.md>`_

rl-garden provides the reusable learning components and the
:doc:`extension contract <guides/extending>` used by that package.
The ``franka_real`` environment backend is registered downstream.
The success-classifier reward model lives here under
``rl_garden/models/reward/success``; its training module is
``rl_garden.models.reward.success.train``.

The downstream ``real_world_rl/`` package owns ActorLoop/LearnerLoop, ``franka_real``/``real_robot`` backends, SERL/HIL-SERL/residual/SiLRI loops and demonstration recording.
Its torch-free device layer lives in ``real_world_rl/{devices,server,discovery,setup,supervise,doctor}``; ``teleop/<device>/`` holds spacemouse, gello, pico, keyboard and leader sources with shared ``core/``, while ``legacy/`` retains EE twist/impedance controllers and the ROS Franka bridge.
Install the downstream ``[learning]`` extra for learning dependencies, or use ``bash scripts/install.sh --daemon-only`` on daemon hosts; the unified command is ``rwrl``.
Clone it with ``git clone --recurse-submodules https://github.com/JaimeParker/real-world-rl.git`` so its pinned rl-garden submodule is present; neither downstream hardware package nor runtime is needed for simulation-only work.
Run downstream tests from that repository's root, following its own current testing instructions.
