"""Entry script that launches the RLinf sync PPO adapter.

Modeled on RLinf's own
``RLinf/examples/embodiment/train_embodied_agent.py`` (read-only
reference, not edited), but does not call RLinf's ``validate_cfg`` -- same
reason as ``rl_garden/integrations/rlinf/train_offline.py``/``train_sac.py``:
our adapter classes aren't RLinf's own registered model/env types, and
``validate_cfg`` is only invoked from example entry scripts, never library
code (see "Launch-plane: ray.remote(cls) accepts any class object" in
``docs/design/rlinf-integration.md``).

Usage (once an RLinf environment with a live ManiSkill install is set up --
see ``.agents/local/6017.md``):

    python -m rl_garden.integrations.rlinf.train_ppo \\
        --config-path config --config-name maniskill_ppo_online
"""
from __future__ import annotations

import hydra
import torch.multiprocessing as mp

from rl_garden.integrations.rlinf import require_rlinf

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="maniskill_ppo_online")
def main(cfg) -> None:
    require_rlinf()
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker

    from rl_garden.integrations.rlinf.ppo_actor import RLGardenPPOFSDPActor
    from rl_garden.integrations.rlinf.ppo_rollout import RLGardenPPORollout

    # No explicit register_rl_garden_ppo_model() call here: importing
    # ppo_model.py (transitively, via ppo_actor/ppo_rollout above) already
    # registers "rl_garden_ppo" as an import-time side effect -- required
    # so every Ray-spawned worker process (actor, rollout, each in its own
    # process, re-importing this module to reconstruct the remote actor
    # class) gets the registration too, not just this driver process. See
    # ppo_model.register_rl_garden_ppo_model's docstring for why a
    # driver-only explicit call (Phase 1/2's convention) doesn't work here.

    # No distributed_log_dir: that comes from cfg.runner.per_worker_log_path,
    # which only validate_cfg populates (skipped here, see module docstring).
    # None (Cluster's own default) just disables split per-worker logging.
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group = RLGardenPPOFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )
    rollout_group = RLGardenPPORollout.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    # eval disabled, no reward model: matches Phase 1/2's "smallest possible
    # foothold" bring-up precedent.
    runner = EmbodiedRunner(
        cfg=cfg, actor=actor_group, rollout=rollout_group, env=env_group, reward=None
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
