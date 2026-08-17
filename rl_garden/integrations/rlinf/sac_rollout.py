"""RLinf async SAC rollout adapter.

Selects actions for the live ManiSkill env using an rl-garden ``SACPolicy``
instead of RLinf's own ``get_model()``-dispatched models. See
``docs/design/rlinf-integration.md``, "SACCore contract (SAC, RLPD)".

Unlike the actor (``sac_actor.py``, a lightweight ``Worker`` subclass that
avoids RLinf's FSDP-actor base entirely), this **does** subclass RLinf's
``AsyncMultiStepRolloutWorker``: there is no FSDP anywhere in the rollout
path (it holds a plain ``nn.Module``), and its real content -- the channel
wire-protocol with the env worker (``recv_from``/``send_to``,
``PolicyOutput`` construction, pipeline-stage looping, weight-sync receive)
-- is exactly the machinery this adapter wants for free. Only two methods
are overridden: ``init_worker`` (build an rl-garden policy instead of
calling RLinf's model registry) and ``predict`` (call that policy directly
instead of going through ``SupportedModel``-dispatched
``predict_action_batch``, so ``cfg.actor.model.model_type`` never needs to
claim to be an RLinf model).

RLinf is optional: this module is importable without it (the class exists
with an ``object`` fallback base), but instantiating
``RLGardenSACRollout`` requires it.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from rl_garden.integrations.rlinf import require_rlinf
from rl_garden.integrations.rlinf.sac_actor import build_algo_from_cfg

try:
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )

    _RLINF_AVAILABLE = True
except ImportError:
    AsyncMultiStepRolloutWorker = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


class RLGardenSACRollout(AsyncMultiStepRolloutWorker):
    """RLinf rollout worker that selects actions with an rl-garden ``SACPolicy``.

    Construction mirrors ``RLGardenSACActor``'s own policy construction
    exactly (both go through ``build_algo_from_cfg``) because
    ``PatchWeightSyncer.init_sender`` enforces exact state-dict key-set
    equality between the actor (sender) and this rollout worker (receiver).
    """

    def __init__(self, cfg: Any) -> None:
        require_rlinf()
        super().__init__(cfg)

    def init_worker(self) -> None:
        algo = build_algo_from_cfg(self.cfg)
        self.hf_model = algo.policy.to(self.device)
        self.hf_model.eval()

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        states = env_obs["states"].to(self.device)
        deterministic = mode == "eval"
        with torch.no_grad():
            actions = self.hf_model.predict(states, deterministic=deterministic)
        num_action_chunks = int(self.model_cfg.num_action_chunks)
        if num_action_chunks != 1:
            raise NotImplementedError(
                "RLGardenSACRollout only supports num_action_chunks=1 "
                f"(got {num_action_chunks}) -- SAC/RLPD select one action "
                "per env step, not an action chunk."
            )
        actions = actions.unsqueeze(1)  # [B, action_dim] -> [B, 1, action_dim]

        # prev_logprobs is read unconditionally by the base class's
        # _build_policy_output (huggingface_worker.py:604-608, used as a
        # shape template for PolicyOutput.versions even when
        # collect_prev_infos=False) and its shape drives
        # TrajectoryReplayBuffer.add_trajectories' T/B accounting
        # (buffer.py:459-466) -- it must be present and correctly shaped.
        # Its value is otherwise unused: SAC/RLPD are off-policy and never
        # read prev_logprobs from a sampled batch (see
        # trajectory_batch_to_sample in sac_actor.py), so a zero
        # placeholder is correct, not just convenient.
        #
        # forward_inputs["action"] duplicates the top-level `actions`
        # return value: EnvWorker._run_interact_once builds
        # ChunkStepResult.actions from
        # `policy_output.forward_inputs.get("action", None)`, not from
        # PolicyOutput.actions itself (env_worker.py:1097) -- VLA models
        # populate this as part of their own forward-pass bookkeeping, but
        # for this adapter the two are identical. Leaving it out means
        # every trajectory's actions field stays None and gets silently
        # dropped by TrajectoryReplayBuffer._flatten_trajectory (only
        # tensor fields survive), which surfaces downstream as a bare
        # KeyError("actions") in the shim, not an error here.
        result = {
            "prev_logprobs": torch.zeros(
                actions.shape[0], num_action_chunks, device=actions.device
            ),
            "prev_values": None,
            "forward_inputs": {"action": actions},
            "intervene_flags": None,
            "expert_label_flag": False,
        }
        return actions, result
