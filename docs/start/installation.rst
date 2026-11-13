Installation
============

Use Python 3.10 or newer in an isolated environment. From a source checkout,
install the core package:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

Core dependencies include PyTorch and Gymnasium. The Pendulum quickstart uses
the Gymnasium backend without a simulator extra.

Optional dependencies
---------------------

Install only the extras needed for your backend or development task, for example:

.. code-block:: bash

   pip install -e ".[maniskill]"
   pip install -e ".[docs]"

The following extras are declared in the repository's pyproject.toml.
Legacy D4RL dependencies belong in a separate environment because their pins
differ from modern simulator stacks.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Dependencies
   * - ``docs``
     - ``sphinx>=7``, ``furo``, ``myst-parser``, ``sphinx-copybutton``
   * - ``maniskill``
     - ``mani_skill``
   * - ``robotwin``
     - ``pillow``, ``pyyaml``
   * - ``minari``
     - ``minari[hf]``, ``gymnasium-robotics``
   * - ``d4rl-legacy``
     - ``gym==0.23.1``, ``mujoco-py==2.1.2.14``, ``d4rl @ git+https://github.com/nakamotoo/D4RL.git@d234792d5562b738b12b78e9074a2d6c7e23b01f``
   * - ``robomimic``
     - ``robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0``, ``robosuite``, ``mujoco>=3.3.0,<3.10``
   * - ``d4rl-legacy-manip``
     - ``gym==0.23.1``, ``numpy==1.23.5``, ``scipy==1.10.1``, ``Cython==0.29.37``, ``mujoco-py==2.1.2.14``, ``dm-control==1.0.14``, ``mujoco==2.3.7``, ``six==1.17.0``, ``termcolor==2.5.0``, ``mjrl @ git+https://github.com/aravindr93/mjrl.git@3871d93763d3b49c4741e6daeaebbc605fe140dc``, ``d4rl @ git+https://github.com/nakamotoo/D4RL.git@d234792d5562b738b12b78e9074a2d6c7e23b01f``
   * - ``ogbench``
     - ``ogbench``
   * - ``metaworld``
     - ``metaworld``
   * - ``dm_control``
     - ``dm_control>=1.0.15``, ``mujoco>=3.0``
   * - ``wandb``
     - ``wandb``
   * - ``dev``
     - ``pytest``, ``pytest-xdist``, ``pytest-timeout``, ``ruff``

CUDA and optional simulators
----------------------------

Normal GPU training keeps rollout tensors, replay samples and updates on CUDA.
Install a PyTorch build compatible with your GPU driver before adding simulator
dependencies. The CPU quickstart explicitly selects CPU; installing the core
package alone does not configure a GPU simulator.

Some backends need an external simulator installation beyond a Python extra.
Follow :doc:`/guides/isaaclab-custom-tasks`,
:doc:`/guides/rlbench-integration` or :doc:`/guides/robotwin` for those systems.
Optional backends are loaded when selected, so you can inspect the CLI before
installing every simulator.

Verify the commands
-------------------

.. code-block:: bash

   rl-garden-train-online sac --help
   rl-garden-eval --help

Continue with :doc:`quickstart`.
