CLI reference
=============

Entry points
------------

.. list-table::
   :header-rows: 1

   * - Console script
     - Source-checkout equivalent
     - Purpose
   * - ``rl-garden-train-online``
     - ``examples/train_online.py``
     - Train with live environment interaction.
   * - ``rl-garden-train-offline``
     - ``examples/pretrain_offline.py``
     - Train from a dataset.
   * - ``rl-garden-train-off2on``
     - ``examples/train_off2on.py``
     - Pretrain or load, then continue with live interaction.
   * - ``rl-garden-eval``
     - ``examples/eval_checkpoint.py``
     - Rebuild and evaluate a saved checkpoint.

Run an examples/ equivalent with Python from a source checkout; it calls the
same console entry point. For each training command, the first positional
token selects an algorithm from that phase's registry.
The :doc:`_generated/algorithms` table lists the available phase/name pairs.
Use the selected command's help or preflight output for its own flags.
See :doc:`/tutorials/01-configuration-and-cli` for executable inspection examples.

Evaluation has no positional algorithm token: select with ``--algorithm``
and ``--phase``, or let checkpoint metadata identify them.
Its required input is ``--checkpoint-path``.
It has a separate ``EvalCheckpointArgs`` parser, not the training control
flags below. In particular, ``--config-path`` identifies a saved run config;
it is not a YAML preset. See :ref:`checkpoint-evaluating-a-checkpoint`.

Training control flags
----------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Action
   * - ``--config PATH``
     - Apply one strict, args-only YAML preset before explicit CLI values.
   * - ``--print-config``
     - Validate and print preflight JSON without constructing environments or agents.
   * - ``--dry-run``
     - Materialize the environment and agent, then exit without training.

The two inspection actions are mutually exclusive, including repeated
actions. A preset can accompany any one of them.
Dry-run may load simulator dependencies, dataset metadata and configured
checkpoint state; it is not a simulator-free check.
It stops before full dataset/replay loading, W&B initialization and checkpoint
writes. A registry can explicitly disallow dry-run.

Precedence is dataclass defaults, then the YAML preset, then explicit CLI
flags. An algorithm's Args subclass can override shared dataclass defaults.
Runtime normalization follows parsing and records derived changes, such as
a CPU replay fallback when CUDA is unavailable.
Unknown YAML fields, duplicate keys and wrong types are errors; presets do
not include other presets or select the phase/algorithm.
See :ref:`configuration-resolution-order` for the full contract.

Use dashes in option names. Tyro-managed argument names also accept
underscores; keep the registry control flags spelled as shown above.
Nested dataclass paths use dots between levels.
A false boolean default is enabled by its flag; a true boolean can be
disabled with its ``no-`` form, including ``--minari.no-download``.
The JSON uses Python field names with underscores.

Shared argument groups
----------------------

The defaults below are the effective **online SAC** preflight values with
ManiSkill PushCube selected and no preset, not a promise that every algorithm
inherits the same defaults. Null means unset. Offline and off2on Args may
resolve evaluation budgets differently; consult their own preflight output.
The configuration guide owns detailed observation/encoder semantics.

LoggingArgs
^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--log-dir``
     - ``runs``
     - Parent directory for run artifacts.
   * - ``--exp-name``
     - ``null``
     - Optional experiment name.
   * - ``--log-freq``
     - ``1000``
     - Logging cadence.
   * - ``--eval-freq``
     - ``10000``
     - Periodic evaluation cadence; zero disables it.
   * - ``--num-eval-steps``
     - ``50``
     - Step budget for each fixed-step evaluation round.
   * - ``--std-log``
     - ``true``
     - Write metrics to the terminal.
   * - ``--log-type``
     - ``tensorboard``
     - Metric sink: tensorboard, wandb or none.
   * - ``--log-keywords``
     - ``null``
     - Comma-separated W&B tags appended to the run name.
   * - ``--wandb-project``
     - ``rl-garden``
     - W&B project when that sink is selected.
   * - ``--wandb-entity``
     - ``null``
     - Optional W&B entity.
   * - ``--log-group``
     - ``null``
     - Group runs and nest TensorBoard output.

CheckpointArgs
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--checkpoint-dir``
     - ``null``
     - Derive the checkpoint directory under the run.
   * - ``--checkpoint-freq``
     - ``0``
     - Disable periodic checkpoint writes.
   * - ``--load-checkpoint``
     - ``null``
     - Optional checkpoint path to load.
   * - ``--save-replay-buffer``
     - ``false``
     - Opt in to a separate replay snapshot.
   * - ``--load-replay-buffer``
     - ``false``
     - Opt in to loading a saved replay snapshot.
   * - ``--save-final-checkpoint``
     - ``true``
     - Write the final checkpoint.

See :doc:`checkpoints` for paths, compatibility and resume behavior.

ObservationArgs
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Flag / namespace
     - Effective default
     - Meaning
   * - ``--obs.state``
     - ``true``
     - Request the main state vector.
   * - ``--obs.rgb``
     - ``[]``
     - Names of requested RGB cameras.
   * - ``--obs.depth``
     - ``[]``
     - Names of requested depth cameras.
   * - ``--obs.extra-state``
     - ``[]``
     - Names of additional state sources.
   * - ``--obs.image-size``
     - ``null``
     - Leave image resolution to the backend.
   * - ``--obs.frame-stack``
     - ``1``
     - Keep one image frame.
   * - ``--encoder.normalize-obs``
     - ``false``
     - Do not enable observation normalization.
   * - ``--obs-groups.actor``
     - ``null``
     - Use the default actor key selection.
   * - ``--obs-groups.critic``
     - ``null``
     - Use the default critic key selection.
   * - ``--encoder-sharing``
     - ``shared_critic_grad``
     - SAC-resolved encoder-sharing mode.
   * - ``--critic-encoder.*``
     - ``{}``
     - No separate critic encoder override in effective JSON.

The state-only preflight output contains only normalization under the encoder
namespace; inactive image options are omitted.
With ``--obs.rgb base_camera`` and otherwise unchanged inputs, the visual
preflight reports ``encoder.backbone=plain_conv``, ``encoder.features_dim=256``
and ``encoder.image_fusion_mode=stack_channels``.
The critic encoder's empty effective mapping means no override, not an
empty Python config object. The Args field uses the default EncoderConfig
instance as its sentinel.
See :ref:`configuration-observation-and-encoder-configuration` for the
observation contract and encoder option narrative.

PolicyModuleArgs
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--actor-factory``
     - ``null``
     - Import path of the actor factory.
   * - ``--actor-extractor``
     - ``null``
     - Import path of the actor feature-extractor class.
   * - ``--critic-extractor``
     - ``null``
     - Import path of the critic feature-extractor class.

These fields are exposed by Args classes that mix in PolicyModuleArgs;
they are not universal flags for every registered algorithm.
Builders resolve import paths when constructing policy modules.
Read :doc:`/tutorials/08-bring-your-own-policy` for the actor and extractor
contracts, checkpoint metadata and the limits of static inspection.

Errors and inspection boundaries
--------------------------------

``ConfigError`` from ``rl_garden.common.effective_config`` identifies an
invalid configuration: an unknown algorithm, invalid preset, conflicting
inspection actions, inactive override or missing checkpoint path, for example.
The training CLI catches it and exits with a ``Configuration error:`` message.
Tyro can also exit with its own parsing error for an invalid flag or value.

``ObservationContractError`` from ``rl_garden.observations.schema`` identifies
an incompatible observation request, space or encoder arrangement.
Backend materialization can raise it for unsupported modalities or key sets.
The registry converts some static encoder-sharing failures into ConfigError;
these classes therefore do not correspond to one universal pair of exit codes.

A successful preflight does not load dataset contents, instantiate a simulator,
validate task assets or prove that a custom actor meets its runtime contract.
Normal runs save preflight and materialized configuration snapshots as
explained in :ref:`configuration-effectiveconfig-v3`.
