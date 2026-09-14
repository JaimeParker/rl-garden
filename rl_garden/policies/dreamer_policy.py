"""``DreamerPolicy(BasePolicy)``: the actor (policy prior) + critic (value
function) + slow (target) critic + return-normalization state that sits on
top of an ``RSSM`` world model (plan model-based-base Part 2, section E).

``actor_extractor = rssm`` -- same in-contract, stateful-world-model-as-
extractor pattern ``TDMPC2Policy`` establishes for TD-MPC2 (see
``.agents/rules/adding-algorithm.md`` part A.5's "Model-based algorithms"
section): ``RSSM`` duck-types the ``BaseFeaturesExtractor`` surface
(``features_dim``/``extract()``/``update_normalizer()``, see
``rl_garden.world_models.base.WorldModel``) without subclassing it.
``critic_extractor=None``, ``encoder_sharing="shared"`` -- there is exactly
one encoder (the RSSM's own), trained jointly by the world-model loss and
the (replay-based ``repval``) critic loss through one shared optimizer (see
``rl_garden.algorithms.dreamer_v3``'s module docstring), so there is no
separate actor/critic extractor split to stop-gradient between.

Deviation from ``BasePolicy``'s abstract ``predict(obs, deterministic) ->
Tensor`` contract (documented, plan-directed -- model-based-base plan
section E, DreamerV3 plan section F): ``predict()`` here takes and returns
explicit recurrent state (``state``, ``is_first`` in; ``(action, new_state)``
out), matching the RSSM's own ``observe()`` contract and following the same
"stateful rollout needs its own explicit-state API, `BasePolicy.predict`'s
single-Tensor-return contract does not fit" precedent as
``RecurrentSACPolicy.act_recurrent_step`` (though that class keeps a
separate method and leaves ``predict()`` alone; this one is directed to make
``predict()`` itself the stateful entry point). Python's ``ABC`` only checks
that the method NAME is overridden, not its signature, so this remains a
legal override -- but it means every caller of this policy's ``predict()``
must be aware of the different signature/return type; the DEFAULT
``BaseAlgorithm._eval_action``/``OffPolicyAlgorithm._policy_action`` paths
(which call ``policy.predict(obs, deterministic=...)`` expecting a bare
Tensor) are never reached for DreamerV3 -- ``DreamerV3`` overrides
``_rollout_action`` (per plan section F) and ``_eval_action_and_critic_action``/
``_eval_start_hook``/``_eval_step_hook`` (mirroring
``rl_garden.algorithms.sequence_sac.SequenceSAC``'s ``_rollout_hidden``/
``_eval_hidden`` split) so both the rollout and eval loops always call this
class's own stateful ``predict()`` directly, threading their OWN separate
state (never sharing one mutable state between training rollout and eval,
matching ``TDMPC2._rollout_action``'s documented reasoning for the same
hazard).
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.networks.dreamer_nets import MLPHead, ReturnEMA
from rl_garden.policies.base import BasePolicy
from rl_garden.world_models.base import State
from rl_garden.world_models.rssm import RSSM


class DreamerPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: "spaces.Box | spaces.Dict",
        action_space: spaces.Space,
        world_model: RSSM,
        *,
        actor_layers: int = 3,
        critic_layers: int = 3,
        units: int = 256,
        reward_bins: int = 255,
        min_std: float = 0.1,
        max_std: float = 1.0,
        slow_target_fraction: float = 0.02,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=world_model,
            critic_extractor=None,
            encoder_sharing="shared",
        )
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        if self.is_discrete:
            action_dim = int(action_space.n)
        else:
            action_dim = int(np.prod(action_space.shape))
        self.action_dim = action_dim
        self.slow_target_fraction = slow_target_fraction

        latent_dim = world_model.latent_dim
        self.actor = MLPHead(
            latent_dim,
            action_dim,
            output="categorical" if self.is_discrete else "bounded_normal",
            layers=actor_layers,
            units=units,
            outscale=0.01,
            min_std=min_std,
            max_std=max_std,
        )
        self.critic = MLPHead(
            latent_dim,
            reward_bins,
            output="symexp_twohot",
            layers=critic_layers,
            units=units,
            outscale=0.0,
        )
        # Slow (target) critic: an EMA copy of self.critic, never trained by
        # gradient descent (excluded from the algorithm's optimizer param
        # list) and always kept in eval() mode -- see self.train() below and
        # rl_garden.algorithms.dreamer_v3's _update_targets.
        self.slow_critic = copy.deepcopy(self.critic)
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)
        self.slow_critic.eval()

        self.return_ema = ReturnEMA()

        if not self.is_discrete:
            low = torch.as_tensor(np.asarray(action_space.low), dtype=torch.float32)
            high = torch.as_tensor(np.asarray(action_space.high), dtype=torch.float32)
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)

    @property
    def world_model(self) -> RSSM:
        return self.actor_extractor

    def train(self, mode: bool = True) -> "DreamerPolicy":
        super().train(mode)
        # The slow critic is an EMA target network -- never itself trained,
        # always eval() regardless of the owning policy's mode (matches
        # r2dreamer's Dreamer.train(), dreamer.py:174-178).
        self.slow_critic.eval()
        return self

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        """``world_model.initial_state()`` plus a zeroed ``"prev_action"`` --
        convenience wrapper for callers that want a complete ``predict()``
        input state up front (``predict()`` itself tolerates a state missing
        ``"prev_action"``, defaulting to zeros, so this is not required)."""
        state = self.world_model.initial_state(batch_size, device)
        state["prev_action"] = torch.zeros(batch_size, self.action_dim, device=device)
        return state

    def predict(
        self,
        obs: Obs,
        state: Optional[State] = None,
        is_first: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, State]:
        """One rollout step: ``embed = rssm.encode(obs)``; fold it into
        ``state`` via ``rssm.observe(...)`` (using ``state["prev_action"]``,
        or zeros if the incoming ``state`` doesn't carry one -- e.g. the very
        first call after ``rssm.initial_state()``); sample (or take the
        mode of) the actor distribution on the resulting features.
        Continuous actions are clipped to the action-space bounds before
        being returned (decision 4 -- ``bounded_normal`` itself does not
        guarantee a sample stays in range, see that function's docstring).
        The action just taken is stored back into the returned state under
        ``"prev_action"`` so a caller threading this state through
        successive calls never needs to pass the action separately (plan
        section E)."""
        embed = self.world_model.encode(obs)
        batch_size = embed.shape[0]
        device = embed.device
        if state is None:
            state = self.world_model.initial_state(batch_size, device)
        prev_action = state.get("prev_action")
        if prev_action is None:
            prev_action = torch.zeros(batch_size, self.action_dim, device=device)
        if is_first is None:
            is_first = torch.zeros(batch_size, dtype=torch.bool, device=device)

        new_state = self.world_model.observe(
            {"deter": state["deter"], "stoch": state["stoch"]}, prev_action, embed, is_first
        )
        features = self.world_model.features(new_state)
        dist = self.actor(features)
        if deterministic:
            action = dist.mode
        elif self.is_discrete:
            action = dist.sample()
        else:
            action = dist.rsample()
        if not self.is_discrete:
            action = torch.clamp(action, self.action_low, self.action_high)

        new_state = {**new_state, "prev_action": action}
        return action, new_state
