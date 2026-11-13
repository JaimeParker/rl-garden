Environment backends
====================

Select a backend
----------------

Choose a registered adapter with ``--env-backend``.
Its named configuration lives under ``--<backend>.*``;
for example, ``--maniskill.sim-backend`` selects the ManiSkill simulator mode.
A preset uses the same backend name as a nested mapping.
Explicit settings for an inactive backend are rejected during preflight.
See :ref:`configuration-resolution-order` for precedence and validation.

The meaning of ``--env-id`` belongs to the backend.
For Gymnasium and CPU MuJoCo it names a registered Gymnasium task.
For Minari it names a dataset whose environment is recovered.
For MuJoCo Warp it names a task in rl-garden's own task registry.
The custom template currently constructs PointReach regardless of this value.
Inspect the selected backend's page before reusing an ID from another backend.

``--print-config`` inspects configuration without starting a simulator.
It does not prove that a task ID exists or that its assets are installed.
Use the :doc:`/reference/cli` inspection workflow before a full run.

Devices and episode boundaries
------------------------------

Algorithms receive batched torch observations, rewards, terminations and
truncations. GPU-native simulators can produce these tensors directly;
ManiSkill's GPU mode and MuJoCo Warp avoid a NumPy vector-environment adapter.

CPU environments use ``TorchVectorEnvAdapter`` from
``rl_garden.envs.vector_env``. On each step it explicitly detaches tensor
actions, moves them to CPU and converts them to NumPy for the environment.
It converts returned arrays to torch tensors on the configured adapter device
and casts float64 observations to float32.
Choosing a CUDA output device does not turn CPU physics into GPU physics.
The replay storage device and algorithm device are separate choices.

The CPU factories use SAME_STEP autoreset. The adapter translates Gymnasium's
terminal observation into ``infos["final_observation"]`` and preserves
terminal episode statistics. GPU-native backends implement this contract
without that translation. ``tests/test_episode_boundary_contract.py`` is the
executable contract; run it before writing a wrapper that changes episode
boundaries.

Register an environment or write a backend
------------------------------------------

For an ordinary single-instance Gymnasium environment, register it in your
package and select the existing Gymnasium backend.
Follow :doc:`/tutorials/07-bring-your-own-environment` for the import-module path.
A dedicated backend is useful for an already-batched simulator, process-global
runtime ownership, or named configuration shared across a task family.
Use :doc:`/guides/extending` for downstream registration contracts.

Backend directory
-----------------

The :doc:`/reference/_generated/backends` table lists every registered backend,
its implementation class, declared installation extra and documentation page.
A dash in the extra column means there is no dedicated extra in this project;
the linked page explains core or external dependencies.
The backend guides in this section cover installation and operational limits.
