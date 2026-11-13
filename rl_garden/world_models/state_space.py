"""``StateSpaceWorldModel``: a differentiable world model over an *explicit*
simulator state, after SuperMimic (arXiv 2610.28480, §3.2 / Eq. 1).

A deterministic MLP reads normalized state features, the action and the
per-episode context, predicts a normalized *local increment*, and an analytic
integrator (``StateCodec.integrate``) turns it into the next state. Training is
the k-step autoregressive loss of Eq. 1: only the window's first state is
real, every later state is the model's own prediction (full BPTT), and the
state error is expressed in the *real* state's local frame and divided by the
running std of the increments (floored per component).

Two points of variation are overridable. ``normalize_output`` (class
attribute, ``True`` here) decides whether the MLP output is de-normalized by
the running increment statistics ``out_norm``; ``_state_loss`` decides how
the predicted states are scored against the real ones.
``NominalErrorStateSpaceWorldModel`` (``nominal_error_state_space.py``)
predicts raw increments and scores the states in the codec's
``LossBlockStateCodec.loss_blocks`` with fixed nominal errors, as the official
implementation does.

Dynamics and learned heads use Linear -> LayerNorm -> GELU hidden layers
and PyTorch's default Linear initialization, as in the official implementation.
The dynamics output is not zero-initialized, so initial steps need not be identity.

Relation to the ``WorldModel`` contract. The model is an explicit-state
degenerate of it: ``encode`` picks a key out of the observation dict instead of
computing a learned embedding, ``observe`` just wraps that state, and
``initial_state`` is all zeros; ``decode`` stays ``None``. ``latent_dim`` is
the codec's ``feature_dim``.

State dict (leading dims ``(B,)`` or ``(T, B)``)::

    {"s": (.., S), "ctx": (.., C), "goal": (.., G), "t": (..,) long}

``state_from_obs`` reads it from an observation dict: ``state_key`` (falling
back to ``obs_key``), ``ctx_key``, ``goal_key``, ``time_key``; absent
context/goal become zero-width and absent time becomes zeros. ``goal`` is not a
dynamics input: it only feeds the heads. When a ``GoalSource`` is supplied
(defaulting to the ``AnalyticMDP``) and ``goal_dim > 0``, ``step`` recomputes
it with ``goal_source.goal(s_next, ctx, t + 1)``, retaining state gradients.

Heads. Reward, observation and termination are each independently

- ``"learned"``: a small MLP on the normalized state features, context and goal
  (reward also on the action), trained by the extra terms of ``model_loss``;
- ``"analytic"``: delegated to the differentiable, task-supplied
  ``AnalyticMDP`` (no parameters, no loss term);
- observation only: ``"identity"``, the observation is the state itself.

Learned heads are evaluated on the predicted states inside the k-step rollout,
so their losses also shape the dynamics; this is Eq. 1 as written.

The analytic reward needs the next state, so use ``transition_reward``;
``reward`` is the learned head only.

Precision. The dynamics/head MLPs may be run under the caller's autocast; the
increment is cast to fp32 and the integration always runs with autocast
disabled.
"""
from __future__ import annotations

import warnings
from typing import Iterator, Literal, Mapping, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.common.types import Obs
from rl_garden.networks.mlp import create_mlp
from rl_garden.networks.running_norm import RunningNorm
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.batch import SequenceBatch

HeadMode = Literal["learned", "analytic"]
ObservationHeadMode = Literal["learned", "analytic", "identity"]


class StateCodec(Protocol):
    """The model's dynamics representation: what the MLP sees (``features``),
    what it predicts (``increment``) and how a prediction is applied
    (``integrate``)."""

    state_dim: int
    feature_dim: int
    increment_dim: int
    #: ``(increment_dim,)`` per-component lower bound on the error scale, or ``None``.
    error_floor: Optional[torch.Tensor]

    def features(self, s: torch.Tensor) -> torch.Tensor:
        """``(..., S) -> (..., F)``."""

    def increment(self, s: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Local increment from ``s`` to ``s1``, ``(..., D)``. The error of a
        prediction ``s_hat`` against truth ``s1`` is ``increment(s1, s_hat)``."""

    def integrate(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """``s (+) d``, differentiable in both arguments."""


class LossBlockStateCodec(StateCodec, Protocol):
    """A ``StateCodec`` that also names the space the state error is scored in.

    ``increment`` decides what the dynamics MLP predicts (for example a
    velocity-level delta that ``integrate`` turns into a pose);
    ``loss_blocks`` decides in which space the prediction error is measured
    and ``nominal_errors`` with which nominal error each block is scored (for
    example the integrated position, scored at 1 cm). Channels left out of
    every block do not enter the loss.
    """

    #: Nominal error per block; the keys are exactly those of ``loss_blocks``.
    nominal_errors: Mapping[str, float]

    def loss_blocks(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """``(..., S) -> {name: (..., W_name)}``, differentiable in ``s``."""


class ResidualStateCodec:
    """Features are the state, the increment is ``s1 - s``, integration adds."""

    def __init__(self, state_dim: int, error_floor: Optional[torch.Tensor] = None) -> None:
        self.state_dim = state_dim
        self.feature_dim = state_dim
        self.increment_dim = state_dim
        self.error_floor = error_floor

    def features(self, s: torch.Tensor) -> torch.Tensor:
        return s

    def increment(self, s: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        return s1 - s

    def integrate(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return s + d


class GoalSource(Protocol):
    """Task-supplied, batched, pure-torch goal generation, differentiable in state."""

    goal_dim: int

    def goal(self, s: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``(B, goal_dim)``; ``(B, 0)`` when ``goal_dim == 0``.

        ``s`` and ``ctx`` are the state and context at time index ``t``.
        """


class AnalyticMDP(GoalSource, Protocol):
    """Task-supplied, batched, pure-torch and differentiable MDP functions.
    Also satisfies ``GoalSource``. ``t`` is ``(B,)`` long (or ``(T, B)``).

    In ``reward(..., t_next)``, ``t_next = t + 1`` indexes ``s_next``;
    ``termination_prob(s, t)`` receives the post-step state and its time index.
    Official SuperMimic scores reward/termination before advancing motion time,
    against ``ref(t_next - 1)``: subtract one when porting those heads.
    ``valid_start`` may return all True; clamping motion time at clip ends,
    as in the official implementation, is the MDP's responsibility.
    """

    obs_dim: int
    goal_dim: int
    #: ``(obs_dim,)`` half-width of uniform observation noise, or ``None``.
    obs_noise: Optional[torch.Tensor]

    def observation(
        self, s: torch.Tensor, prev_action: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Noise-free observation."""

    def reward(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
        s_next: torch.Tensor,
        prev_action: torch.Tensor,
        ctx: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """``(B,)``."""

    def termination_prob(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``(B,)`` in ``[0, 1]``, differentiable."""

    def goal(self, s: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``(B, goal_dim)``; ``(B, 0)`` when ``goal_dim == 0``."""

    def critic_features(self, s: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``(B, K)``, ``K`` may be 0."""

    def valid_start(self, t: torch.Tensor, horizon: int) -> torch.Tensor:
        """``(B,)`` bool: whether an imagination of ``horizon`` steps may start at ``t``."""


class StateSpaceWorldModel(WorldModel):
    """Explicit-state dynamics with GELU/LayerNorm MLPs and default initialization."""

    #: De-normalize the dynamics output with ``out_norm`` (and keep its
    #: statistics updated). ``out_norm`` exists either way.
    normalize_output: bool = True

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        obs_dim: int,
        ctx_dim: int = 0,
        goal_dim: int = 0,
        codec: Optional[StateCodec] = None,
        mdp: Optional[AnalyticMDP] = None,
        goal_source: Optional[GoalSource] = None,
        reward_head: HeadMode = "learned",
        observation_head: ObservationHeadMode = "identity",
        termination_head: HeadMode = "learned",
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        head_hidden_dims: tuple[int, ...] = (256, 256),
        state_key: str = "state_sim",
        obs_key: str = "state",
        ctx_key: str = "state_ctx",
        goal_key: str = "state_goal",
        time_key: str = "state_time",
        in_norm_eps: float = 1e-2,
        out_norm_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if mdp is None and "analytic" in (reward_head, observation_head, termination_head):
            raise ValueError("an 'analytic' head requires an AnalyticMDP (mdp=...).")
        if observation_head == "identity" and obs_dim != state_dim:
            raise ValueError(
                f"observation_head='identity' requires obs_dim == state_dim, got {obs_dim} != {state_dim}."
            )
        self.codec: StateCodec = codec if codec is not None else ResidualStateCodec(state_dim)
        self.mdp = mdp
        self.goal_source = goal_source if goal_source is not None else mdp
        if self.goal_source is not None and self.goal_source.goal_dim != goal_dim:
            raise ValueError(
                f"goal_source.goal_dim ({self.goal_source.goal_dim}) must match goal_dim ({goal_dim})."
            )
        if (
            goal_dim > 0
            and self.goal_source is None
            and "learned" in (reward_head, observation_head, termination_head)
        ):
            warnings.warn(
                "Learned heads read goals frozen at the imagination start while training uses "
                "stepwise advancing recorded goals; provide goal_source for time-varying goals.",
                stacklevel=2,
            )
        self.state_dim, self.action_dim, self.obs_dim = state_dim, action_dim, obs_dim
        self.ctx_dim, self.goal_dim = ctx_dim, goal_dim
        self.reward_mode, self.observation_mode, self.termination_mode = (
            reward_head,
            observation_head,
            termination_head,
        )
        self.state_key, self.obs_key = state_key, obs_key
        self.ctx_key, self.goal_key, self.time_key = ctx_key, goal_key, time_key
        self.out_norm_eps = out_norm_eps
        self.latent_dim = self.codec.feature_dim

        feature_dim = self.codec.feature_dim
        self._feature_slice = slice(0, feature_dim)
        self._action_slice = slice(feature_dim, feature_dim + action_dim)
        self._ctx_slice = slice(feature_dim + action_dim, feature_dim + action_dim + ctx_dim)
        dynamics_in_dim = feature_dim + action_dim + ctx_dim

        self.in_norm = RunningNorm((dynamics_in_dim,), eps=in_norm_eps)
        self.out_norm = RunningNorm((self.codec.increment_dim,), eps=out_norm_eps)
        self.goal_norm = RunningNorm((goal_dim,), eps=in_norm_eps)

        self.dynamics_mlp = create_mlp(
            dynamics_in_dim,
            self.codec.increment_dim,
            hidden_dims,
            activation_fn=nn.GELU,
            use_layer_norm=True,
        )

        state_context_dim = feature_dim + ctx_dim + goal_dim
        self.reward_mlp = (
            create_mlp(
                state_context_dim + action_dim, 1, head_hidden_dims, activation_fn=nn.GELU, use_layer_norm=True
            )
            if reward_head == "learned"
            else None
        )
        self.observation_mlp = (
            create_mlp(state_context_dim, obs_dim, head_hidden_dims, activation_fn=nn.GELU, use_layer_norm=True)
            if observation_head == "learned"
            else None
        )
        self.termination_mlp = (
            create_mlp(state_context_dim, 1, head_hidden_dims, activation_fn=nn.GELU, use_layer_norm=True)
            if termination_head == "learned"
            else None
        )

    # ------------------------------------------------------------------
    # Observation dict <-> State
    # ------------------------------------------------------------------

    def _read_state_tensor(self, obs: Obs) -> torch.Tensor:
        key = self.state_key if self.state_key in obs else self.obs_key
        return obs[key].float()

    def state_from_obs(self, obs: Obs) -> State:
        """Works for any leading dims (``(B,)`` or ``(T, B)``)."""
        s = self._read_state_tensor(obs)
        lead = s.shape[:-1]
        ctx = obs[self.ctx_key].float() if self.ctx_key in obs else s.new_zeros(*lead, 0)
        goal = obs[self.goal_key].float() if self.goal_key in obs else s.new_zeros(*lead, 0)
        if self.time_key in obs:
            t = obs[self.time_key]
            if t.ndim == s.ndim:  # stored as a (.., 1) observation column
                t = t.squeeze(-1)
            t = t.long()
        else:
            t = torch.zeros(lead, dtype=torch.long, device=s.device)
        return {"s": s, "ctx": ctx, "goal": goal, "t": t}

    def encode(self, obs: Obs) -> torch.Tensor:
        return self._read_state_tensor(obs)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return {
            "s": torch.zeros(batch_size, self.state_dim, device=device),
            "ctx": torch.zeros(batch_size, self.ctx_dim, device=device),
            "goal": torch.zeros(batch_size, self.goal_dim, device=device),
            "t": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def observe(
        self, state: State, action: Optional[torch.Tensor], embed: torch.Tensor, is_first: torch.Tensor
    ) -> State:
        del action, is_first  # explicit state: nothing recurrent to reset.
        return {**state, "s": embed}

    def update_normalizer(self, obs: Obs) -> None:
        """No observation encoder to update; see ``update_normalizers``."""

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _dynamics_input(self, state: State, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.codec.features(state["s"].float()), action.float(), state["ctx"].float()], dim=-1)

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        del sample  # deterministic.
        s = state["s"].float()
        increment = self.dynamics_mlp(self.in_norm(self._dynamics_input(state, action)))
        if self.normalize_output:
            increment = self.out_norm.denormalize(increment)
        with torch.autocast(device_type=s.device.type, enabled=False):
            s_next = self.codec.integrate(s, increment.float())
        t_next = state["t"] + 1
        if self.goal_source is not None and self.goal_dim > 0:
            goal_next = self.goal_source.goal(s_next, state["ctx"], t_next)
        else:
            goal_next = state["goal"]
        return {"s": s_next, "ctx": state["ctx"], "goal": goal_next, "t": t_next}

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    def _normalized_slice(self, x: torch.Tensor, sl: slice) -> torch.Tensor:
        mean, var = self.in_norm.mean[sl], self.in_norm.var[sl]
        return (x - mean) / torch.sqrt(var + self.in_norm.eps**2)

    def _head_input(self, state: State, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [self._normalized_slice(self.codec.features(state["s"].float()), self._feature_slice)]
        if action is not None:
            parts.append(self._normalized_slice(action.float(), self._action_slice))
        parts.append(self._normalized_slice(state["ctx"].float(), self._ctx_slice))
        parts.append(self.goal_norm(state["goal"].float()))
        return torch.cat(parts, dim=-1)

    def features(self, state: State) -> torch.Tensor:
        return self.codec.features(state["s"])

    def transition_reward(
        self, state: State, action: torch.Tensor, next_state: State, prev_action: torch.Tensor
    ) -> torch.Tensor:
        """``(..., 1)``. Learned: the reward head on ``(state, action)``.
        Analytic: ``mdp.reward`` on the transition."""
        if self.reward_mode == "learned":
            return self.reward(state, action)
        return self.mdp.reward(
            state["s"], action, next_state["s"], prev_action, state["ctx"], next_state["t"]
        ).unsqueeze(-1)

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        if self.reward_mode != "learned":
            raise NotImplementedError(
                "the analytic reward depends on the next state; use transition_reward()."
            )
        return self.reward_mlp(self._head_input(state, action))

    def reward_value(
        self, state: State, action: Optional[torch.Tensor], *, symlog_space: bool = False
    ) -> torch.Tensor:
        if symlog_space:
            raise ValueError("StateSpaceWorldModel rewards are not symlog-encoded.")
        return self.reward(state, action)

    def termination_logit(self, state: State) -> torch.Tensor:
        if self.termination_mode != "learned":
            raise NotImplementedError("termination_logit exists only for a learned termination head.")
        return self.termination_mlp(self._head_input(state))

    def termination_prob(self, state: State) -> torch.Tensor:
        """``(..., 1)``."""
        if self.termination_mode == "learned":
            return torch.sigmoid(self.termination_logit(state))
        return self.mdp.termination_prob(state["s"], state["t"]).unsqueeze(-1)

    def continue_(self, state: State) -> torch.Tensor:
        return 1.0 - self.termination_prob(state)

    def observation(self, state: State, prev_action: torch.Tensor) -> torch.Tensor:
        """``(..., obs_dim)``."""
        if self.observation_mode == "identity":
            return state["s"]
        if self.observation_mode == "learned":
            return self.observation_mlp(self._head_input(state))
        return self.mdp.observation(state["s"], prev_action, state["ctx"], state["t"])

    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        heads = [m for m in (self.reward_mlp, self.observation_mlp, self.termination_mlp) if m is not None]
        return {
            "encoder": iter(()),
            "model": self.dynamics_mlp.parameters(),
            "heads": (p for head in heads for p in head.parameters()),
        }

    @torch.no_grad()
    def update_normalizers(self, obs: Obs, action: torch.Tensor, next_obs: Obs) -> None:
        state, next_state = self.state_from_obs(obs), self.state_from_obs(next_obs)
        self.in_norm.update(self._dynamics_input(state, action))
        if self.normalize_output:
            self.out_norm.update(self.codec.increment(state["s"], next_state["s"]))
        if self.goal_dim > 0:
            self.goal_norm.update(state["goal"])

    # ------------------------------------------------------------------
    # Training loss (Eq. 1)
    # ------------------------------------------------------------------

    def _error_scale(self, like: torch.Tensor) -> torch.Tensor:
        scale = self.out_norm.std().clamp_min(self.out_norm_eps)
        floor = self.codec.error_floor
        if floor is not None:
            scale = torch.maximum(scale, floor.to(scale))
        return scale.to(like)

    def _rollout_window(self, batch: SequenceBatch) -> tuple[State, torch.Tensor, torch.Tensor]:
        """Roll the model from the window's real first state with full BPTT.

        Returns the real window ``State`` (leading dims ``(k + 1, B)``), the
        predicted states ``s_hat`` ``(k + 1, B, S)`` (``s_hat[0]`` is the real
        first state) and the window's actions ``(k, B, A)``.
        """
        window = self.state_from_obs(batch.obs)
        action = batch.action
        state: State = {"s": window["s"][0], "ctx": window["ctx"][0], "goal": window["goal"][0], "t": window["t"][0]}
        predicted = [state["s"]]
        for j in range(action.shape[0]):
            state = self.step(state, action[j])
            predicted.append(state["s"])
        return window, torch.stack(predicted, dim=0), action

    def _state_loss(
        self, s_real: torch.Tensor, s_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Score predicted against real states, both ``(k + 1, B, S)``.

        Returns the state loss, the per-step error ``(k, B)`` (its last step
        is logged as ``state_err_last``) and extra loss components to log,
        here none. The default is the codec increment error divided by the
        floored running increment std, averaged over components, summed over
        steps and averaged over ``B``.
        """
        increment_error = self.codec.increment(s_real[1:], s_hat[1:])
        per_step_error = (increment_error / self._error_scale(increment_error)).square().mean(-1)
        return per_step_error.sum(0).mean(), per_step_error, {}

    def model_loss(self, batch: SequenceBatch) -> tuple[dict[str, torch.Tensor], State]:
        """k-step autoregressive loss over a ``(k + 1, B)`` observation window.

        ``batch.obs`` is a dict of ``(k + 1, B, ...)``; ``batch.action``
        ``(k, B, A)``; ``batch.reward`` ``(k, B)`` or ``(k, B, 1)``;
        ``batch.terminated`` ``(k, B)`` (needed for a learned termination
        head). Predicted states are rolled from the real first state with full
        BPTT (``_rollout_window``) and scored by ``_state_loss``. Every head
        term is summed over the ``k`` steps and averaged over ``B``;
        learned heads see the *recorded* goal of each step, even with a
        state-dependent ``GoalSource``; analytic/identity heads add no term.
        The extra components of ``_state_loss`` are added to the returned
        dict, detached.
        """
        window, s_hat, action = self._rollout_window(batch)
        horizon, batch_size = action.shape[0], action.shape[1]
        s_real = window["s"]
        ctx = window["ctx"][0]
        t0 = window["t"][0]

        loss_state, per_step_error, extra = self._state_loss(s_real, s_hat)

        zero = s_hat.new_zeros(())
        loss_reward = loss_observation = loss_termination = zero
        times = t0.unsqueeze(0) + torch.arange(horizon + 1, device=t0.device).view(-1, *([1] * t0.ndim))
        ctx_all = ctx.unsqueeze(0).expand(horizon + 1, *ctx.shape)

        def window_state(sl: slice) -> State:
            return {"s": s_hat[sl], "ctx": ctx_all[sl], "goal": window["goal"][sl], "t": times[sl]}

        before, after = window_state(slice(0, horizon)), window_state(slice(1, None))
        if self.reward_mode == "learned":
            prev_action = torch.cat([torch.zeros_like(action[:1]), action[:-1]], dim=0)
            predicted_reward = self.transition_reward(before, action, after, prev_action).squeeze(-1)
            target_reward = batch.reward.reshape(horizon, batch_size).to(predicted_reward)
            loss_reward = (predicted_reward - target_reward).square().sum(0).mean()
        if self.observation_mode == "learned":
            predicted_obs = self.observation(after, action)
            target_obs = batch.obs[self.obs_key][1:].to(predicted_obs)
            loss_observation = (predicted_obs - target_obs).square().mean(-1).sum(0).mean()
        if self.termination_mode == "learned":
            if batch.terminated is None:
                raise ValueError("a learned termination head needs batch.terminated.")
            terminated = batch.terminated.reshape(horizon, batch_size).float()
            loss_termination = (
                F.binary_cross_entropy_with_logits(self.termination_logit(after).squeeze(-1), terminated, reduction="none")
                .sum(0)
                .mean()
            )

        losses = {
            "state": loss_state,
            "reward": loss_reward,
            "observation": loss_observation,
            "termination": loss_termination,
            "total": loss_state + loss_reward + loss_observation + loss_termination,
            "state_err_last": per_step_error[-1].mean().detach(),
            **{name: value.detach() for name, value in extra.items()},
        }
        posterior: State = {"s": s_real[0].detach(), "ctx": ctx, "goal": window["goal"][0], "t": t0}
        return losses, posterior
