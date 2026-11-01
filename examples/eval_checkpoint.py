"""Evaluate a saved rl-garden checkpoint in a live environment.

Thin wrapper over :mod:`rl_garden.training.evaluation` (also installed as the
``rl-garden-eval`` console script); see that module for the CLI and how the
training args are rebuilt.

Example:

    python examples/eval_checkpoint.py \
      --checkpoint-path runs/<run_name>/checkpoints/<checkpoint>.pt \
      --num-eval-episodes 50 --output-json /tmp/eval.json

The underscore-prefixed names below are re-exported for scripts that load this
file by path and call its helpers; new code imports them from
``rl_garden.training.evaluation``. Assigning to a name here does not change
the package module's behavior.
"""

from __future__ import annotations

from rl_garden.training.evaluation import (  # noqa: F401
    EvalCheckpointArgs,
    _algorithm_from_class,
    _apply_mapping_to_args,
    _build_agent,
    _builder_from_module,
    _config_args,
    _env_request,
    _headline_metrics,
    _load_training_args,
    _registry_for_phase,
    _resolve_phase_and_algorithm,
    _training_module,
    _write_json,
    evaluate_agent,
    evaluate_checkpoint,
    main,
)

if __name__ == "__main__":
    main()
