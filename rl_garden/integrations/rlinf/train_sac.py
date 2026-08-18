"""Entry script that launches the RLinf async SAC/RLPD adapter.

Modeled on RLinf's own
``RLinf/examples/embodiment/train_async.py`` (read-only reference,
not edited), but does not call RLinf's ``validate_cfg`` -- same reason as
``rl_garden/integrations/rlinf/train_offline.py``: our adapter classes
aren't RLinf's own registered model/env types, and `validate_cfg` is only
invoked from example entry scripts, never library code (see
"Launch-plane: ray.remote(cls) accepts any class object" in
``docs/design/rlinf-integration.md``).

Usage (once an RLinf environment with a live ManiSkill install is set up --
see ``.agents/local/9990.md``):

    python -m rl_garden.integrations.rlinf.train_sac \\
        --config-path config --config-name maniskill_sac_online
"""
from __future__ import annotations

import hydra
import torch.multiprocessing as mp

from rl_garden.integrations.rlinf import require_rlinf

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="maniskill_sac_online")
def main(cfg) -> None:
    require_rlinf()
    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker

    from rl_garden.integrations.rlinf.sac_actor import RLGardenSACActor
    from rl_garden.integrations.rlinf.sac_rollout import RLGardenSACRollout

    # No distributed_log_dir: that comes from cfg.runner.per_worker_log_path,
    # which only validate_cfg populates (skipped here, see module docstring).
    # None (Cluster's own default) just disables split per-worker logging.
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group = RLGardenSACActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )
    rollout_group = RLGardenSACRollout.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    runner = AsyncEmbodiedRunner(
        cfg=cfg, actor=actor_group, rollout=rollout_group, env=env_group, reward=None
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
