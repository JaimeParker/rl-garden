"""RLinf sync PPO actor adapter.

Drives training via RLinf's own ``EmbodiedRunner`` (sync, collect-then-update
-- ``3rd_party/RLinf/rlinf/runners/embodied_runner.py``), not the async
``AsyncEmbodiedRunner``/``AsyncPPOEmbodiedRunner`` paths. See
``docs/design/rlinf-integration.md``, "PPO contract".

Unlike Phase 1/2 (``offline_actor.py``, ``sac_actor.py``), which subclass
RLinf's plain ``Worker`` to avoid inheriting FSDP machinery only to override
it away, this adapter deliberately subclasses ``EmbodiedFSDPActor`` -- a
choice made explicitly, not by default: PPO's `run_training`/
`train_micro_batch` (shuffle, multi-epoch minibatch passes, gradient
accumulation, distributed metric all-reduce) is substantial enough that
reimplementing it in a plain ``Worker`` would be reinventing exactly the
machinery ``EmbodiedFSDPActor`` already provides. Subclassing a concrete
RLinf worker base class is itself sanctioned by
``docs/design/rlinf-integration.md``'s "Launch-plane" section (its own
example: ``RLGardenIQLWorker(EmbodiedFSDPActor)``) -- not a violation of
the "adapters must not reference a concrete algorithm class by name" rule,
which targets rl-garden's own algorithm classes and is enforced here at the
``ppo_model._POLICIES`` registry level, not this actor's base class.

**What never gets constructed**: this design never instantiates
``rl_garden.algorithms.ppo.PPO`` -- only the bare ``PPOPolicy`` network
(via ``ppo_model.build_policy_from_cfg``, itself dispatched through RLinf's
own ``get_model()`` registry, see ``ppo_model.py``). RLinf's own generic
``compute_ppo_actor_loss``/``compute_ppo_critic_loss``
(``rlinf/algorithms/losses.py``) drive training through
``EmbodiedFSDPActor.run_training``/``train_micro_batch``, inherited
unchanged. ``PPO._optimizer_names()``/``_extra_checkpoint_state()`` (the
polymorphic checkpoint hooks Phase 1/2 relied on) are not used here --
optimizer construction and checkpointing both flow through
``FSDPModelManager`` instead. A checkpoint saved by this adapter is not
loadable by rl-garden's own ``PPO.load()``, and vice versa -- an accepted
consequence of never constructing a ``PPO`` instance, not a gap to close.

RLinf is optional: this module is importable without it (the class exists
with an ``object`` fallback base), but instantiating
``RLGardenPPOFSDPActor`` requires it.
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.integrations.rlinf import require_rlinf

# Import-time registration side effect (register_rl_garden_ppo_model, run
# at the bottom of ppo_model.py): Ray reconstructs RLGardenPPOFSDPActor in
# its own worker process by re-importing this module, and RLinf's
# _MODEL_REGISTRY is a plain in-process dict -- a registration made only
# in the driver process (train_ppo.py's old convention) never reaches this
# worker's own copy, so model_provider_func's get_model() call would
# silently return None here without this import. See
# ppo_model.register_rl_garden_ppo_model's docstring.
from rl_garden.integrations.rlinf import ppo_model  # noqa: F401
from rl_garden.integrations.rlinf.ppo_advantages import (
    compute_advantages_from_rollout_batch,
)

try:
    from rlinf.algorithms.utils import safe_normalize
    from rlinf.utils.metric_utils import compute_rollout_metrics
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    _RLINF_AVAILABLE = True
except ImportError:
    EmbodiedFSDPActor = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


class RLGardenPPOFSDPActor(EmbodiedFSDPActor):
    """PPO actor driven by RLinf's own FSDP training loop.

    Overrides only ``compute_advantages_and_returns``. Everything else --
    ``init_worker``/``setup_model_and_optimizer`` (model built via the
    ``"rl_garden_ppo"`` registry key, see ``ppo_model.py``),
    ``sync_model_to_rollout``, ``recv_rollout_trajectories``,
    ``run_training``/``train_micro_batch`` (the full FSDP training loop),
    ``set_global_step``, ``save_checkpoint``/``load_checkpoint`` -- is
    inherited unchanged from ``EmbodiedFSDPActor``.
    """

    def __init__(self, cfg: Any) -> None:
        require_rlinf()
        super().__init__(cfg)

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Auto-reset-correct GAE, replacing RLinf's own ``"gae"`` registry entry.

        The conversion itself (batch-shape handling, the auto-reset
        bootstrap fix) lives in ``ppo_advantages
        .compute_advantages_from_rollout_batch`` -- kept RLinf-independent
        and separately unit-tested (mirrors
        ``sac_actor.trajectory_batch_to_sample``'s role in Phase 2). This
        method is the thin RLinf-dependent glue: supplies the model's value
        head as a closure, applies ``algorithm.normalize_advantages``
        (matching RLinf's own stock behavior), and writes the result back
        into ``self.rollout_batch`` in the layout ``train_micro_batch``'s
        downstream reads expect.
        """

        def value_head_forward(states: torch.Tensor) -> torch.Tensor:
            # self.model(...) -- NOT self.model.default_forward(...)
            # directly. Under real FSDP sharding (sharding_strategy other
            # than no_shard), parameters only get all-gathered to full
            # size around an actual nn.Module.__call__ invocation (which
            # FSDP hooks into); calling default_forward as a plain method
            # bypasses those hooks and sees each rank's own (partial, or
            # under full_shard on a single-parameter-owning rank,
            # zero-sized) local shard instead. train_micro_batch already
            # calls self.model(...), matching this. Confirmed a real,
            # sharding-strategy-specific failure by a live 2-GPU
            # full_shard launch on 6017: RuntimeError: setStorage: ...
            # storage size of 0 -- invisible under no_shard (Tier 3),
            # since there every rank already holds the full parameter.
            return self.model(
                forward_inputs={"states": states},
                compute_logprobs=False,
                compute_entropy=False,
                compute_values=True,
            )["values"]

        advantages, returns = compute_advantages_from_rollout_batch(
            self.rollout_batch,
            value_head_forward,
            self.cfg.algorithm.get("gamma", 1.0),
            self.cfg.algorithm.get("gae_lambda", 1.0),
        )

        if self.cfg.algorithm.get("normalize_advantages", True):
            advantages = safe_normalize(
                advantages, loss_mask=self.rollout_batch.get("loss_mask", None)
            )

        self.rollout_batch["advantages"] = advantages
        self.rollout_batch["returns"] = returns

        return compute_rollout_metrics(self.rollout_batch)
