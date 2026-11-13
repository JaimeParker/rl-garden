Minari
======

Install and select
------------------

Install the ``minari`` extra with ``pip install -e ".[minari]"``.
It includes Minari's download support and Gymnasium Robotics.
The recovered task may need additional environment dependencies and assets.

Select ``--env-backend minari`` and use a Minari dataset ID as ``--env-id``.
The live environment is recovered from the dataset metadata.
This is distinct from selecting ``--dataset-backend minari`` for offline
transition loading; neither identifier is an H5 filesystem path.

Namespace flags
---------------

These defaults come from the SAC preflight configuration with this backend
selected and no backend overrides. The command may override a default explicitly.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--minari.device``
     - ``cpu``
     - Device for output tensors from the recovered environment.
   * - ``--minari.download``
     - ``true``
     - Allow Minari to fetch an absent dataset during construction.

Validated command
-----------------

.. code-block:: bash

   rl-garden-train-online sac --env-backend minari \
     --env-id metaworld/reach/expert-v0 --minari.no-download \
     --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
     --buffer-size 512 --batch-size 16 --hidden-dim 32 \
     --learning-starts 16 --training-freq 4 --total-timesteps 64 \
     --eval-freq 0 --log-freq 32 --no-capture-video \
     --exp-name backend-minari --print-config

Observations
------------

The factory wraps the recovered environment's Box observation as ``state``.
It does not rename arbitrary Dict observations or extract goal keys.
The registered backend accepts only state observation requests.
A dataset that recovers an incompatible observation space is not supported
by this adapter, even if Minari itself can load the dataset.

Offline dataset loading is a separate path.
Follow :doc:`/tutorials/04-offline-pretraining` for the distinction between
an environment source, a dataset locator and the transition format.

Devices
-------

The factory recovers one environment per requested instance and uses a
synchronous CPU Gymnasium vector environment with SAME_STEP autoreset.
The torch adapter transfers actions to CPU NumPy and outputs to its device.
Training recovery uses the dataset's ordinary environment; evaluation
recovery asks Minari for its evaluation environment.
Episode statistics are recorded before conversion to the torch contract.

Known limits
------------

The example disables downloads and performs static inspection only.
To run it, cache that dataset first and remove the final inspection flag.
Missing datasets or missing task dependencies then fail at construction.
An installed Minari package is required even when files are cached.
This backend does not provide camera requests, image stacking or extra state.

Use :doc:`/guides/ogbench-integration` or
:doc:`/guides/d4rl-legacy-expansion` for those separate dataset ecosystems.
