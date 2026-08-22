"""Orchestrator stub for the official JAX implicit_q_learning (IQL-jax)
baseline, against rl-garden's canonical environments.

Not yet implemented. Full hyperparameter/CLI design is deferred follow-up
work; this file exists so ``baselines.iql_jax`` is importable and
``baselines/baselines.yaml``'s ``iql_jax.integration_module`` entry
resolves.

Expected shape, once built (see ``baselines/baselines.yaml``'s
``iql_jax`` entry and its ``notes`` field for the full rationale):

- Mirrors ``baselines.cal_ql.run_offline``'s pattern closely: ``sys.path``
  prepended with the submodule directory (matching
  ``tools/diagnostics/probe_iql_jax_q_contrast.py``'s existing convention of
  pointing at ``3rd_party/implicit_q_learning``), then import ``Learner``
  from ``learner.py`` and ``D4RLDataset`` from ``dataset_utils.py``.
- Training samples only from the static in-memory ``D4RLDataset`` -- no
  online rollout. The environment is only exercised via
  ``baselines.core.env_bridge.GymnasiumEnvBridge`` +
  ``baselines.core.evaluation.evaluate_bridge_policy`` for periodic
  evaluation (matching ``evaluation.py``'s ``evaluate()`` call sites in the
  upstream ``train_offline.py``, cadence controlled by ``--eval_interval``),
  the same low-frequency cadence as Cal-QL's eval, not wsrl's every-step
  online cadence.
- Arguments it will need: ``--iql-source``, ``--dataset`` (or a D4RL env id,
  since IQL-jax builds its dataset via ``gym.make(env_name)`` +
  ``D4RLDataset(env)`` directly rather than a pre-exported npz),
  ``--env-python`` (bridge), ``--output-dir``, and IQL hyperparameters
  (``expectile``, ``temperature``, ``actor_lr``, etc., from
  ``configs/default.py`` under ``3rd_party/implicit_q_learning``).
- Its D4RL dependency (``rail-berkeley/d4rl@master``, currently unpinned in
  ``requirements.txt``) should be pinned to a specific commit as part of
  building this, matching the pinning convention already used for
  cal_ql's/wsrl's forks in the manifest.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "baselines.iql_jax.run_offline is not yet implemented -- see "
        "this module's docstring and baselines/baselines.yaml."
    )


if __name__ == "__main__":
    main()
