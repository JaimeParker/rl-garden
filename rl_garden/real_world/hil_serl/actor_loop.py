"""HIL-SERL's ActorLoop. HITL intervention and the hybrid continuous+discrete
action are handled by the env wrapper stack and the policy respectively, so
the only override needed is tagging pushed transitions with whether a human
was intervening -- ``HilSerlLearnerLoop`` uses this to route the transition
into the online replay buffer or the growing demo buffer.
"""
from __future__ import annotations

from rl_garden.real_world.actor_loop import ActorLoop


class HilSerlActorLoop(ActorLoop):
    def _extra_transition_fields(self, info: dict) -> dict:
        return {"intervened": "intervene_action" in info}
