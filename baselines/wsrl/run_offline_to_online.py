"""Orchestrator stub for the official JAX wsrl baseline (offline pretraining
+ online fine-tuning), against rl-garden's canonical environments.

Not yet implemented. Full hyperparameter/CLI design is deferred follow-up
work; this file exists so ``baselines.wsrl`` is importable and
``baselines/baselines.yaml``'s ``wsrl.integration_module`` entry
resolves.

Expected shape, once built (see ``baselines/baselines.yaml``'s ``wsrl``
entry and its ``notes`` field for the full rationale):

- Run with ``cwd=3rd_party/wsrl`` (or prepend ``3rd_party/wsrl`` to
  ``sys.path``) -- ``wsrl.agents``/``wsrl.envs.env_common`` resolve relative
  to the submodule's own directory, unlike Cal-QL's ``sys.path.insert``
  style.
- Unlike ``baselines.cal_ql.run_offline`` (bridge used only for periodic
  evaluation), wsrl's online phase steps the environment through
  ``baselines.core.env_bridge.GymnasiumEnvBridge`` on *every* training
  step, not just eval -- expect this to be an IPC-overhead bottleneck worth
  profiling before trusting timing comparisons against rl-garden's own
  in-process rollout.
- Needs its own D4RL fork (``zhouzypaul/D4RL``, pinned in the manifest's
  ``d4rl_fork``), distinct from Cal-QL's/rl-garden's ``nakamotoo/D4RL``.
- Arguments it will need: ``--wsrl-source``, ``--agent`` (calql/cql/sac),
  an offline dataset locator (mirroring
  ``get_d4rl_dataset_with_mc_calculation``), ``--num-offline-steps``,
  ``--num-online-steps``, ``--env-python`` (bridge, exercised online).
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "baselines.wsrl.run_offline_to_online is not yet implemented -- "
        "see this module's docstring and baselines/baselines.yaml."
    )


if __name__ == "__main__":
    main()
