"""Entry script that launches ``RLGardenOfflineActor`` under RLinf.

Modeled on RLinf's own
``3rd_party/RLinf/examples/embodiment/train_offline_rl.py`` (read-only
reference, not edited), but does not call RLinf's ``validate_cfg`` -- that
function rejects any ``cfg.actor.model.model_type`` outside RLinf's own
``SupportedModel`` enum, and this adapter is not one of RLinf's own models
(see docs/design/rlinf-integration.md, "Launch-plane: ray.remote(cls)
accepts any class object" -- ``validate_cfg`` is only invoked from example
entry scripts, not library code, so a custom entry script skips it
entirely, by design).

Usage (once an RLinf environment is set up -- see
``.agents/local/9990.md``/``.agents/local/personal_config.md`` for this
repo's known-working venv):

    python -m rl_garden.integrations.rlinf.train_offline \\
        --config-path config --config-name d4rl_offline_antmaze
"""
from __future__ import annotations

import hydra

from rl_garden.integrations.rlinf import require_rlinf


@hydra.main(version_base="1.1", config_path="config", config_name="d4rl_offline_antmaze")
def main(cfg) -> None:
    require_rlinf()
    from rlinf.runners.offline_runner import OfflineRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    from rl_garden.integrations.rlinf.offline_actor import RLGardenOfflineActor

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor_group = RLGardenOfflineActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # eval disabled: no env/rollout worker groups (see "Offline-uniform
    # contract" in docs/design/rlinf-integration.md -- this adapter's
    # sync_model_to_rollout() is a stub for exactly this reason).
    runner = OfflineRunner(cfg=cfg, actor=actor_group, env=None, rollout=None)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
