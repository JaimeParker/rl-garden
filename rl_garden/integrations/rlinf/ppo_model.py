"""Wraps an rl-garden policy to satisfy RLinf's embodied model protocol.

Registers a new RLinf model type (``"rl_garden_ppo"``) via ``register_model``
(``RLinf/rlinf/models/__init__.py`` -- a real, public, mutation-
based extension point, not a source edit) so ``EmbodiedFSDPActor`` and
``MultiStepRolloutWorker`` can build this model through RLinf's own
``get_model()`` dispatch, exactly like RLinf's own ``MLPPolicy``
(``RLinf/rlinf/models/embodiment/mlp_policy/mlp_policy.py``, the
concrete reference this wrapper's ``default_forward``/``predict_action_batch``
shape was built against).

Unlike Phase 1/2 (``offline_actor.py``, ``sac_actor.py``, ``sac_rollout.py``),
which bypass RLinf's FSDP/model-registry machinery entirely, this Phase 3
adapter deliberately goes through it: ``RLGardenPPOFSDPActor``
(``ppo_actor.py``) subclasses ``EmbodiedFSDPActor`` to reuse its FSDP-aware
training loop rather than reimplementing it. The consequence, accepted
explicitly (see ``docs/design/rlinf-integration.md``, Phase 3): RLinf's own
generic ``compute_ppo_actor_loss``/``compute_ppo_critic_loss``
(``rlinf/algorithms/losses.py``) drive training, not
``rl_garden.algorithms.ppo.PPO``'s own ``train()``. This wrapper contributes
only network architecture -- an rl-garden policy's actor/value networks --
never the algorithm's update loop.

Never references ``PPOPolicy`` by name outside the ``_POLICIES`` registry
below (mirrors ``sac_actor.py``'s ``_ALGORITHMS``/``resolve_algorithm``
pattern): this wrapper is written against a duck-typed contract -- "the PPO
contract, network layer" -- requiring only
``act_with_value_and_logprob``/``evaluate_actions(..., sum_dims=False)``/
``predict``/``predict_values``, not any concrete policy class. A future
policy on the same contract (e.g. a ``SequencePPO``-family policy) needs
only a ``_POLICIES`` entry, not a wrapper change -- per
``docs/design/rlinf-integration.md``'s "adapters target the hook contract"
acceptance criterion, applied one layer below algorithm selection.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.integrations.rlinf import require_rlinf
from rl_garden.policies.ppo_policy import PPOPolicy

try:
    from rlinf.models import register_model
    from rlinf.models.embodiment.base_policy import BasePolicy as RLinfBasePolicy

    _RLINF_AVAILABLE = True
except ImportError:
    RLinfBasePolicy = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


# The PPO contract, network layer (docs/design/rlinf-integration.md, "PPO
# contract"). RecurrentPPO/TransformerPPO (the SequencePPO contract) are
# deliberately excluded -- see that document's "Known, unscheduled" table
# and the module docstring above.
_POLICIES: dict[str, type[PPOPolicy]] = {
    "ppo": PPOPolicy,
}


def resolve_policy(name: str) -> type[PPOPolicy]:
    """Look up a PPO-contract policy class by config name.

    Raises ``ValueError`` for unknown names. Kept RLinf-independent so this
    dispatch logic is testable without RLinf installed (mirrors
    ``rl_garden.integrations.rlinf.sac_actor.resolve_algorithm``).
    """
    policy_cls = _POLICIES.get(name)
    if policy_cls is None:
        valid = ", ".join(sorted(_POLICIES))
        raise ValueError(
            f"Unsupported cfg.actor.model.rlgarden_policy={name!r}. "
            f"Supported (PPO contract): {valid}. RecurrentPPO/TransformerPPO "
            "(the SequencePPO contract) are deliberately not supported by "
            "this wrapper -- see docs/design/rlinf-integration.md, "
            "'Known, unscheduled'."
        )
    return policy_cls


def build_policy_from_cfg(cfg: Any) -> PPOPolicy:
    """Build an rl-garden policy from ``cfg.actor.model``.

    Construction mirrors ``rl_garden.algorithms.ppo.PPO._setup_model``'s own
    recipe for the state-based (``Box`` observation) case: a
    ``FlattenExtractor`` feature extractor plus ``PPOPolicy`` built from
    plain ``gymnasium.spaces.Box`` observation/action spaces derived from
    ``obs_dim``/``action_dim`` -- no live env, matching the ``OfflineEnvSpec``
    construction pattern already established for Phase 1/2's algorithms.
    Called identically by both ``RLGardenPPOFSDPActor`` (via
    ``_build_rl_garden_ppo_model``, registered as the ``"rl_garden_ppo"``
    model builder) and, indirectly, ``RLGardenPPORollout`` (via the same
    registry dispatch) -- both sides must build an identical policy for
    ``PatchWeightSyncer``'s exact-key-set-equality requirement to hold.
    """
    model_cfg = cfg
    policy_cls = resolve_policy(model_cfg.get("rlgarden_policy", "ppo"))
    obs_dim = int(model_cfg.obs_dim)
    action_dim = int(model_cfg.action_dim)
    observation_space = spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype="float32"
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype="float32"
    )
    features_extractor = FlattenExtractor(observation_space=observation_space)
    policy_kwargs = dict(model_cfg.get("rlgarden_policy_kwargs", {}))
    return policy_cls(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor=features_extractor,
        **policy_kwargs,
    )


class RLGardenPPOModel(nn.Module, RLinfBasePolicy):
    """Adapts a PPO-contract rl-garden policy to RLinf's embodied model protocol.

    ``BasePolicy``'s abstract methods are ``default_forward`` and
    ``predict_action_batch`` (``RLinf/rlinf/models/embodiment/
    base_policy.py``); everything else (``enable_torch_compile``,
    ``capture_cuda_graph``) keeps ``BasePolicy``'s default
    ``NotImplementedError`` stubs -- out of scope for this pilot (see
    ``maniskill_ppo_online.yaml``'s ``enable_torch_compile: False``/
    ``enable_cuda_graph: False``).
    """

    def __init__(self, policy: PPOPolicy, num_action_chunks: int = 1) -> None:
        # No require_rlinf() here: this is a plain nn.Module, unit-testable
        # without RLinf installed (RLinfBasePolicy falls back to `object`
        # when RLinf is unavailable, a harmless no-op mixin). RLinf's own
        # machinery only enters the picture at registration time
        # (register_rl_garden_ppo_model) and via RLGardenPPOFSDPActor/
        # RLGardenPPORollout, both of which do call require_rlinf().
        super().__init__()
        if num_action_chunks != 1:
            raise NotImplementedError(
                "RLGardenPPOModel only supports num_action_chunks=1 "
                f"(got {num_action_chunks}) -- PPO selects one action per "
                "env step, not an action chunk."
            )
        # Assignment order matters and is load-bearing, not cosmetic:
        # value_head must be set *before* policy. named_parameters()
        # dedups by first-registration-order when the same parameter
        # tensor is reachable via two attribute paths (verified directly:
        # nn.Module's default remove_duplicate=True keeps whichever named
        # path was registered first, dropping the later alias entirely) --
        # policy.value_net and this top-level value_head alias are the
        # same tensors reachable both ways, so registering value_head
        # first is what makes "value_head" actually survive into
        # named_parameters()'s output. FSDPModelManager.build_optimizer
        # (fsdp_model_manager.py) buckets parameters into a separate LR
        # group by substring-matching "value_head"/"model.value_head"
        # against exactly that (deduped) output -- get the order wrong and
        # the value head silently trains at the actor's LR instead of its
        # own, no error raised. See
        # tests/test_rlinf_ppo_model.py::test_value_head_attribute_reachable_for_optimizer_split
        # for the regression test.
        self.value_head = policy.value_net
        self.policy = policy

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Explicit override, not relying on RLinfBasePolicy.forward's own
        # dispatch-to-default_forward (BasePolicy.forward(self,
        # forward_type=ForwardType.DEFAULT, **kwargs)): nn.Module defines
        # its own forward (the _forward_unimplemented stub), and since
        # nn.Module is listed first in this class's bases, its forward
        # shadows RLinfBasePolicy.forward in MRO resolution regardless of
        # what BasePolicy itself does -- confirmed by a real launch on
        # 6017 (TypeError: _forward_unimplemented() got an unexpected
        # keyword argument 'forward_inputs', from EmbodiedFSDPActor
        # .train_micro_batch's `self.model(forward_inputs=..., ...)` call).
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Explicit device move, matching MLPPolicy.preprocess_env_obs's own
        # pattern (mlp_policy.py) -- forward_inputs/env_obs are not
        # guaranteed to already be on the model's device (confirmed by a
        # real launch on 6017: RuntimeError: mat1 is on cpu, different
        # from other tensors on cuda:0, inside the rollout worker's
        # predict_action_batch call before this fix).
        states = forward_inputs["states"].to(self._device)
        if compute_logprobs or compute_entropy:
            actions = forward_inputs["action"].to(self._device)
            values, log_prob, entropy = self.policy.evaluate_actions(
                states, actions, sum_dims=False
            )
        else:
            values = self.policy.predict_values(states)
            log_prob = entropy = None

        output: dict[str, torch.Tensor] = {}
        if compute_logprobs:
            output["logprobs"] = log_prob
        if compute_entropy:
            output["entropy"] = entropy
        if compute_values:
            output["values"] = values
        return output

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, torch.Tensor],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        states = env_obs["states"].to(self._device)
        # sum_dims=False: prev_logprobs must match default_forward's
        # unsummed, per-action-dimension `logprobs` shape -- RLinf's own
        # preprocess_loss_inputs computes the PPO ratio from
        # exp(logprobs - prev_logprobs) elementwise before any reduction,
        # so a shape mismatch here would silently broadcast wrong (or
        # error) rather than producing a merely-different-but-valid loss.
        actions, values, log_prob, _entropy = self.policy.forward(
            states, deterministic=(mode == "eval"), sum_dims=False
        )
        chunk_actions = actions.unsqueeze(1)  # [B, action_dim] -> [B, 1, action_dim]
        forward_inputs = {"action": actions}
        if return_obs:
            forward_inputs["states"] = states
        result = {
            "prev_logprobs": log_prob,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


def _build_rl_garden_ppo_model(cfg: Any, torch_dtype: torch.dtype) -> RLGardenPPOModel:
    # PPOPolicy always builds a value_net unconditionally -- add_value_head
    # isn't actually read by build_policy_from_cfg, so a False value here
    # would be a config lie that silently trains fine anyway. Fail loudly
    # instead (docs/design/rlinf-integration.md, "Silent-degradation class
    # of failures").
    if not cfg.get("add_value_head", True):
        raise ValueError(
            "actor.model.add_value_head=False is not supported: PPOPolicy "
            "always builds a value head."
        )
    policy = build_policy_from_cfg(cfg)
    num_action_chunks = int(cfg.get("num_action_chunks", 1))
    return RLGardenPPOModel(policy, num_action_chunks=num_action_chunks)


def register_rl_garden_ppo_model() -> None:
    """Register the ``"rl_garden_ppo"`` model type with RLinf's own registry.

    A new key, not overwriting RLinf's existing ``"mlp_policy"`` -- both are
    zero-RLinf-source-edit, but a new key has a smaller behavioral
    footprint on the shared process (doesn't change what ``"mlp_policy"``
    means for anything else in the same interpreter). Costs one small
    override on the rollout worker side (``ppo_rollout.py``'s ``predict``)
    since RLinf's own dispatch hardcodes a handful of known model-type
    names.

    Called automatically at module import time (see bottom of this file),
    **not** only from ``train_ppo.py``'s ``main()``: RLinf's actor/rollout
    workers each run in their own Ray-spawned process, and
    ``_MODEL_REGISTRY`` (``rlinf/models/__init__.py``) is a plain
    in-process dict -- a driver-only call never reaches the worker
    processes' own copies. Ray does re-import this module when
    reconstructing the remote actor class in each worker process, which is
    what makes an import-time call reach every process that needs it. A
    driver-only explicit call (Phase 1/2's convention, safe there because
    neither ever registered anything with RLinf's registry) silently left
    every worker's ``get_model()`` returning ``None`` -- confirmed by a
    real launch on 6017: ``RLGardenPPORollout.init_worker``'s
    ``self.hf_model.eval()`` raised ``AttributeError: 'NoneType' object
    has no attribute 'eval'``.
    """
    require_rlinf()
    register_model(
        "rl_garden_ppo", _build_rl_garden_ppo_model, category="embodied", force=False
    )


if _RLINF_AVAILABLE:
    register_rl_garden_ppo_model()
