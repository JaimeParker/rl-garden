"""Shared, algorithm-agnostic Args mixin + real-robot-only backend config for
real-robot training entrypoints.

``RealWorldFrankaArgs`` fields are common to every real-robot training method
(SERL, HIL-SERL, ...): actor/learner role dispatch, sync-server network
topology, and the ``ActorLoop``/``LearnerLoop`` cadence knobs. Algorithm-
specific fields (e.g. SERL's RLPD hyperparameters, or HIL-SERL's
classifier/demo-dataset paths) belong in each method's own
``rl_garden/training/real_world/<method>.py``, not here.

``FrankaRealConfig`` lives here (not in ``rl_garden/common/env_args.py``,
where the other env backends' configs live) because it's exclusively used by
real-robot training methods -- online/offline/off2on never target
``env_backend="franka_real"``. Each method's Args class opts in to it as a
field explicitly (e.g. ``SerlArgs.franka_real``), the same way
``rl_garden/training/online/rlpd.py``'s ``RLPDArgs`` composes
``EnvBackendArgs`` itself rather than inheriting it implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RealWorldFrankaArgs:
    role: Literal["actor", "learner"] = "actor"

    # Actor: address of the learner's sync server. Learner: bind address for
    # that same server. Both processes must agree on the port. Cross-machine
    # by default (SERL's own topology) -- point this at the learner's real,
    # network-reachable address for a two-machine deployment.
    sync_host: str = "127.0.0.1"
    sync_port: int = 6000

    control_hz: float = 10.0
    deterministic_actor: bool = False
    train_freq: int = 1
    publish_freq: int = 100

    env_backend: str = "franka_real"


@dataclass
class FrankaRealConfig:
    """Real-Franka env settings. CLI prefix: ``--franka_real.<field>``"""

    bridge_url: str = "http://localhost:5000"
    action_scale_pos: float = 0.02
    action_scale_rot: float = 0.1
    gripper_threshold: float = 0.5
    max_episode_steps: int = 100
    # JSON-encoded dict forwarded verbatim to FrankaRealEnvConfig, taking
    # precedence over the named fields above. Escape hatch for cell-specific
    # settings (safety_box_low/high, camera_keys, device) that don't belong
    # as named CLI fields here (see adding-env-backend.md Step 3).
    env_kwargs_json: str = "{}"
