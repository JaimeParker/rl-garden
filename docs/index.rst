.. _home:

rl-garden
=========

.. raw:: html

   <div class="rlg-hero">
     <img class="rlg-hero-logo" src="_static/logo.png" alt="rl-garden logo" />
     <h1 class="rlg-hero-title">rl-garden</h1>
     <p class="rlg-hero-subtitle">Train and evaluate robot-learning agents with a PyTorch framework for online reinforcement learning, offline datasets and simulation.</p>
   </div>

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Start
      :link: start/installation
      :link-type: doc

      Install the framework and run your first experiment.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Learn by building and extending agents.

   .. grid-item-card:: Algorithms
      :link: algorithms/index
      :link-type: doc

      Choose a family of learning methods.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Look up CLI, configuration and Python contracts.

   .. grid-item-card:: Backends
      :link: backends/index
      :link-type: doc

      Select a simulator or dataset interface.

   .. grid-item-card:: Design
      :link: design/index
      :link-type: doc

      Understand the architecture and extension contract.

   .. grid-item-card:: Real robot
      :link: real-robot
      :link-type: doc

      Find the downstream hardware documentation.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Start

   start/installation
   start/quickstart
   start/concepts

.. toctree::
   :maxdepth: 2
   :hidden:
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
   :hidden:
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
   :hidden:
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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Design

   design/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Real robot

   real-robot
