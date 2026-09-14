"""``WorldModel``: the abstract base every model-based algorithm's learned
dynamics model implements, shared across decision-time-planning models
(TD-MPC2's latent-consistency model, ``rl_garden.world_models
.latent_consistency.LatentConsistencyModel``) and imagination-based models
(DreamerV3's RSSM, a future consumer -- see ``rl_garden.world_models.imagine``
and the model-based-base plan's Part 2).

Vocabulary, shared across the three concrete "what does the model predict"
families this base is designed to fit (see the model-based-base plan and
scratchpad ``dreamer-code-survey.md`` section 10, "interface fit"):

- **embedding vs. state.** ``encode(obs)`` produces an *observation
  embedding* -- for Dreamer's RSSM this is a token from the CNN/MLP encoder,
  *not* the full recurrent ``{deter, stoch}`` state (the embedding alone
  can't be rolled forward; it must be combined with the previous state and
  action via ``observe`` first). For TD-MPC2, whose latent has no separate
  recurrent carry, the embedding *is* numerically the state's only content
  (see ``LatentConsistencyModel``'s module docstring), but the two methods
  stay conceptually distinct so a Dreamer-style model can implement this
  base without the interface lying about what ``encode`` alone gives you.
- **``is_first``.** Every recurrent model (Dreamer's RSSM) needs a per-batch-
  element mask marking sequence starts so it zeroes the carried state at an
  episode boundary within a training window, instead of leaking the
  previous episode's dynamics across the seam. TD-MPC2 has no recurrent
  carry to reset (each ``observe`` call re-encodes from scratch, ignoring
  incoming state entirely), so it accepts and ignores this argument rather
  than dropping it from the shared signature.
- **termination_fn injection when ``continue_`` is ``None``.** A model that
  hasn't learned its own continuation/termination head (``continue_``
  returns ``None``) is not usable standalone inside ``imagine()``'s rollout
  without termination information from elsewhere -- the caller (a
  ``ModelBasedAlgorithm`` subclass, Part 1.5+) must supply a
  ``termination_fn(state) -> Tensor`` computed from a known ground truth
  (e.g. a task's known episode length, or a hand-written termination
  predicate) instead. This base does not implement that injection itself;
  it only documents the contract ``continue_``'s ``None`` return leaves for
  the caller to fill.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import torch
import torch.nn as nn

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor

#: Per-model state representation. Batch dimension first (or ``(B, ...)``
#: inside a ``(T, B, ...)`` rollout array, see ``rl_garden.world_models
#: .imagine.ImaginedTrajectory``). Contents are entirely up to the concrete
#: model -- e.g. ``{"z": Tensor}`` for TD-MPC2's single-tensor latent,
#: ``{"deter": Tensor, "stoch": Tensor}`` for a future Dreamer RSSM port.
State = dict[str, torch.Tensor]


class WorldModel(nn.Module, ABC):
    """Learned dynamics model: encode real observations into a latent state,
    roll that state forward under actions (with or without new observations),
    and predict reward/continuation/decoded-observation from it.

    Subclasses set ``self.encoder``/``self.latent_dim`` in ``__init__`` and
    implement every method below. ``continue_``/``decode`` have workable
    defaults (``None``, "not modeled") so a subclass that doesn't need them
    (TD-MPC2's non-episodic path, any model without a pixel decoder) doesn't
    have to override them.
    """

    #: Feature extractor turning raw ``obs`` into an embedding via
    #: ``encode()``. Concrete models that build their own bespoke encoder
    #: (e.g. a task-conditioned MLP, see ``MultitaskWorldModel``) may not
    #: literally hold a ``BaseFeaturesExtractor`` instance here -- the type
    #: hint documents the common case.
    encoder: BaseFeaturesExtractor
    #: Dimensionality of the tensor ``features(state)`` returns.
    latent_dim: int

    # ------------------------------------------------------------------
    # BaseFeaturesExtractor duck-typing.
    #
    # A WorldModel is used as a BasePolicy's ``actor_extractor`` (see
    # rl_garden.policies.tdmpc2_policy.TDMPC2Policy) -- it does NOT subclass
    # BaseFeaturesExtractor (that class's single encode-and-return-features
    # contract doesn't fit a stateful model with this much larger method
    # surface), so it instead provides the specific attributes/methods
    # BasePolicy actually touches on an extractor: ``features_dim``,
    # ``extract()``, ``update_normalizer()``.
    # ------------------------------------------------------------------

    @property
    def features_dim(self) -> int:
        return self.latent_dim

    def extract(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Returns this model's raw observation embedding (``encode(obs)``),
        detached if requested -- satisfies ``BasePolicy.extract_actor_features``'s
        call shape without this class subclassing ``BaseFeaturesExtractor``."""
        embed = self.encode(obs)
        return embed.detach() if stop_gradient else embed

    def update_normalizer(self, obs: Obs) -> None:
        """Delegates to ``self.encoder`` (the actual feature extractor whose
        running obs-normalization statistics, if any, need updating)."""
        self.encoder.update_normalizer(obs)

    @abstractmethod
    def encode(self, obs: Obs) -> torch.Tensor:
        """Observation -> embedding (not a full state, see module docstring).

        For a recurrent model this embedding must go through ``observe()``
        together with the previous state/action to become a usable state;
        for a non-recurrent model ``observe()`` may do nothing but wrap this
        embedding into a ``State`` dict.
        """

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> State:
        """The state to warm-start ``observe()``/``step()`` with before any
        real observation has been seen (e.g. all-zero ``deter``/``stoch`` for
        an RSSM; an empty placeholder for a stateless model)."""

    @abstractmethod
    def observe(
        self, state: State, action: Optional[torch.Tensor], embed: torch.Tensor, is_first: torch.Tensor
    ) -> State:
        """Posterior/filtering step: fold a new observation embedding (plus
        the action taken to reach it, and the previous state) into an
        updated state. ``is_first`` (bool tensor, one per batch element)
        marks sequence starts; a recurrent model must zero the incoming
        ``state`` wherever it's set before combining with ``embed``, instead
        of carrying a previous episode's dynamics across the seam."""

    @abstractmethod
    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        """Prior step: roll ``state`` forward under ``action`` with no new
        observation (imagination / planning rollout). ``sample`` selects
        between sampling a stochastic component (Dreamer's categorical
        stochastic state) and using its mode/mean; models with no stochastic
        component ignore it."""

    @abstractmethod
    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        """Predicted reward at ``state`` (some models also condition on
        ``action``; a model that doesn't may accept and ignore it). Returns
        whatever the concrete model's reward head natively outputs -- e.g.
        two-hot bin logits for TD-MPC2/DreamerV3 (decode with
        ``rl_garden.networks.twohot.two_hot_inv``), not a scalar."""

    def continue_(self, state: State) -> Optional[torch.Tensor]:
        """Probability of the episode continuing past ``state`` (Dreamer's
        Bernoulli continue head; TD-MPC2's ``1 - termination_prob`` when
        ``episodic=True``). ``None`` means this model has no learned
        continuation head -- the caller must inject a ground-truth
        ``termination_fn(state)`` instead (see module docstring)."""
        return None

    def decode(self, state: State) -> Optional[dict[str, "torch.distributions.Distribution"]]:
        """Per-observation-key reconstruction distributions from ``state``
        (Dreamer's pixel/state decoder). ``None`` means this model has no
        decoder (TD-MPC2 never reconstructs observations)."""
        return None

    @abstractmethod
    def features(self, state: State) -> torch.Tensor:
        """The flat tensor actor/critic heads consume (TD-MPC2: ``state["z"]``
        directly; Dreamer: ``cat([deter, stoch])``)."""

    @abstractmethod
    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        """Named parameter groups for the owning algorithm's optimizer,
        keyed by role rather than by this model's private submodule names --
        callers (``TDMPC2``/``TDMPC2Multitask``) build their optimizer's
        param groups from this instead of reaching into ``world_model._foo``
        directly. At minimum: ``"encoder"`` (the observation-embedding
        stack, typically trained at a scaled-down ``lr``) and ``"model"``
        (everything else this model owns -- dynamics/reward/continuation
        heads; NOT the actor/critic, which live on the policy)."""

    @abstractmethod
    def model_loss(self, batch) -> tuple[dict[str, torch.Tensor], State]:
        """Computes this model's own training losses (never the actor/critic
        losses -- those read the posterior states this returns, but are
        computed by the owning policy/algorithm) from ``batch`` (a
        subclass-defined, algorithm-supplied structure -- typically an
        obs/action/reward window). Returns ``(losses, posterior_states)``
        where ``posterior_states`` is a ``State`` whose tensors are
        ``(horizon [+ 1], B, ...)`` time-major and **live** (graph-attached,
        not detached), ready to seed ``imagine()`` or feed a separately
        trained critic without the caller re-running ``observe()``.

        Revised 2026-09-14 (model-based-base plan item 0): posterior states
        were originally specified detached, forcing a caller whose critic
        must backprop into the model's own encoder/dynamics (TD-MPC2) to
        re-run a second, duplicate rollout for a live-gradient copy. Returning
        the live states instead lets such a caller read them directly; a
        caller that needs a detached copy (e.g. TD-MPC2's actor update,
        matching upstream's ``update_pi(zs.detach(), ...)``) calls
        ``.detach()`` on what it's given.
        """
