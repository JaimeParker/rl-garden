rl-garden
=========

Train and evaluate robot-learning agents with a PyTorch framework for online
reinforcement learning, offline datasets and simulation.

Start
   :doc:`start/installation` gets your checkout ready.
   :doc:`start/quickstart` runs your first SAC experiment.

Tutorials
   Follow :doc:`tutorials/index` to move from configuration to extending the framework.

Algorithms
   Browse :doc:`algorithms/index` for learning methods and their existing deep dives.

Reference
   Use :doc:`reference/index` and :doc:`guides/configuration` when you need a contract
   or a configuration detail.

Backends
   Select a simulator or dataset interface with :doc:`backends/index`.

Design
   Read :doc:`design/index` for architecture and the extension contract.

Real robot
   Follow :doc:`real-robot` to the downstream hardware documentation.

.. toctree::
   :maxdepth: 2
   :caption: Start

   start/installation
   start/quickstart
   start/concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/01-configuration-and-cli
   tutorials/02-online-rl
   tutorials/03-observations-and-encoders
   tutorials/04-offline-pretraining
   tutorials/05-offline-to-online
   tutorials/06-checkpoints-and-evaluation
   tutorials/07-bring-your-own-environment
   tutorials/08-bring-your-own-policy
   tutorials/09-extending-from-your-package

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms/index
   algorithms/ppo-family
   algorithms/sac-family
   algorithms/offline-rl
   algorithms/offline-to-online
   algorithms/imitation-and-generative
   algorithms/model-based
   guides/decision-level-ppo
   guides/policy-graphs
   guides/dreamer-v3
   guides/ncrl
   guides/supermimic
   guides/wsrl-reproduction
   guides/offline-acceleration
   guides/d4rl-legacy-expansion

.. toctree::
   :maxdepth: 2
   :caption: Backends

   backends/index
   backends/maniskill
   backends/mujoco
   backends/mujoco-warp
   backends/minari
   backends/custom
   guides/gymnasium-custom-envs
   guides/isaaclab-custom-tasks
   guides/isaaclab-camera-stall
   guides/dm-control-integration
   guides/metaworld-integration
   guides/ogbench-integration
   guides/rlbench-integration
   guides/robomimic-integration
   guides/robotwin

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Design

   design/index

.. toctree::
   :maxdepth: 2
   :caption: Real robot

   real-robot
