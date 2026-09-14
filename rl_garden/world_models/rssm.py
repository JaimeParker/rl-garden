"""``RSSM(WorldModel)``: DreamerV3's recurrent state-space model (plan
model-based-base Part 2, section B), ported from r2dreamer's ``RSSM``/
``Deter`` (``rssm.py``; identical block-GRU/KL math in official JAX
``dreamerv3/rssm.py``). See scratchpad ``dreamer-code-survey.md`` section 1.

State: ``{"deter": (B, deter), "stoch": (B, stoch=32, discrete), "logits":
(B, stoch, discrete)}`` -- ``"logits"`` holds whichever logits produced
``"stoch"`` (posterior logits after ``observe()``, prior logits after
``step()``); ``model_loss()`` needs the posterior ones for the KL loss, and
re-derives prior logits separately (once, batched over the whole training
window -- see that method's docstring) rather than reading them back out of
each step's own ``"logits"``.

``features(state) = cat([deter, stoch.flatten(-2)])`` -- deter first, then
flattened stoch (matches this repo's ``WorldModel.features`` docstring
example; r2dreamer's own ``get_feat`` concatenates stoch first then deter --
harmless order difference, see that method's docstring).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn

from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.dreamer_nets import BlockLinear, MLPHead, OneHotDist, RMSNorm, dreamer_weight_init_
from rl_garden.observations.schema import ObservationSchema
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.decoder import ObservationDecoder


@dataclass(frozen=True)
class RSSMSize:
    """One row of DreamerV3's model-size table (r2dreamer
    ``configs/model/size{12,25,50,100,200,400}M.yaml``; identical in
    official JAX ``configs.yaml``'s ``size12m``..``size400m`` anchors,
    scratchpad ``dreamer-code-survey.md`` section 8). ``discrete`` is the
    number of classes per stochastic variable (the ``stoch=32`` variable
    *count* itself is fixed across every size, an ``RSSM`` constructor
    argument, not part of this table); ``units`` sizes every MLP head/
    encoder-MLP/decoder-MLP hidden layer; ``cnn_depth`` sizes the
    encoder/decoder conv stacks' base channel depth."""

    deter: int
    hidden: int
    discrete: int
    units: int
    cnn_depth: int


RSSM_SIZES: dict[str, RSSMSize] = {
    "size12M": RSSMSize(deter=2048, hidden=256, discrete=16, units=256, cnn_depth=16),
    "size25M": RSSMSize(deter=3072, hidden=384, discrete=24, units=384, cnn_depth=24),
    "size50M": RSSMSize(deter=4096, hidden=512, discrete=32, units=512, cnn_depth=32),
    "size100M": RSSMSize(deter=6144, hidden=768, discrete=48, units=768, cnn_depth=48),
    "size200M": RSSMSize(deter=8192, hidden=1024, discrete=64, units=1024, cnn_depth=64),
    "size400M": RSSMSize(deter=12288, hidden=1536, discrete=96, units=1536, cnn_depth=96),
}


def kl_loss(
    post_logits: torch.Tensor, prior_logits: torch.Tensor, free: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """DreamerV3's dyn/rep KL losses with free-bits clipping (r2dreamer
    ``RSSM.kl_loss``, ``rssm.py:222-230``): ``dyn = KL(sg(post) || prior)``,
    ``rep = KL(post || sg(prior))``, each a categorical KL per stochastic
    variable (summed over the ``discrete`` classes) then summed over the 32
    stochastic variables, then clipped to a minimum of ``free`` nats --
    ``torch.clamp(x, min=free)`` is exactly free-bits: once a term is below
    the budget its gradient is zero (the clamp is constant there), matching
    r2dreamer's own comment on this line. ``post_logits``/``prior_logits``
    are ``(..., stoch, discrete)``; returns ``(dyn_loss, rep_loss)``, each
    ``(...,)`` (both trailing dims reduced).
    """

    def _categorical_kl(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:
        logprob_left = torch.log_softmax(logits_left, dim=-1)
        logprob_right = torch.log_softmax(logits_right, dim=-1)
        prob_left = torch.softmax(logits_left, dim=-1)
        return (prob_left * (logprob_left - logprob_right)).sum(dim=-1)

    rep_loss = _categorical_kl(post_logits, prior_logits.detach()).sum(dim=-1)
    dyn_loss = _categorical_kl(post_logits.detach(), prior_logits).sum(dim=-1)
    rep_loss = torch.clamp(rep_loss, min=free)
    dyn_loss = torch.clamp(dyn_loss, min=free)
    return dyn_loss, rep_loss


class _Deter(nn.Module):
    """Block-GRU dynamics core (r2dreamer ``Deter``, ``rssm.py:10-75``):
    ``deter``/``stoch``/``action`` each project (independent ``Linear ->
    RMSNorm -> act``) to ``hidden`` width, concatenate, broadcast over
    ``blocks``, concatenate with the per-block deterministic state, and run
    through ``dyn_layers`` ``BlockLinear`` layers then one final
    ``BlockLinear`` producing GRU-style reset/candidate/update gates.
    Supports any number of leading batch dims (unlike upstream's ``(B,
    ...)``-only version)."""

    def __init__(
        self,
        deter: int,
        flat_stoch: int,
        action_dim: int,
        hidden: int,
        blocks: int,
        dyn_layers: int,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in0 = nn.Sequential(nn.Linear(deter, hidden, bias=True), RMSNorm(hidden), act())
        self.in1 = nn.Sequential(nn.Linear(flat_stoch, hidden, bias=True), RMSNorm(hidden), act())
        self.in2 = nn.Sequential(nn.Linear(action_dim, hidden, bias=True), RMSNorm(hidden), act())
        self.in0.apply(dreamer_weight_init_)
        self.in1.apply(dreamer_weight_init_)
        self.in2.apply(dreamer_weight_init_)

        hidden_layers: list[nn.Module] = []
        in_ch = (3 * hidden + deter // blocks) * blocks
        for _ in range(dyn_layers):
            hidden_layers.append(BlockLinear(in_ch, deter, blocks))
            hidden_layers.append(RMSNorm(deter))
            hidden_layers.append(act())
            in_ch = deter
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.gru = BlockLinear(in_ch, 3 * deter, blocks)
        # BlockLinear self-initializes at construction (see its docstring);
        # RMSNorm defaults to weight=1 (torch's own nn.RMSNorm.reset_parameters).

    def _flat2group(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _group2flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch_flat: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        x0 = self.in0(deter)
        x1 = self.in1(stoch_flat)
        x2 = self.in2(action)
        x = torch.cat([x0, x1, x2], dim=-1)
        expand_shape = list(x.shape)
        expand_shape.insert(-1, self.blocks)
        x = x.unsqueeze(-2).expand(*expand_shape)

        combined = torch.cat([self._flat2group(deter), x], dim=-1)
        flat = self._group2flat(combined)
        h = self.hidden_layers(flat)
        gru_out = self._flat2group(self.gru(h))
        reset, cand, update = (self._group2flat(g) for g in torch.chunk(gru_out, 3, dim=-1))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter


class RSSM(WorldModel):
    """DreamerV3's world model: recurrent posterior/prior over a categorical
    stochastic state plus deterministic GRU state, an observation decoder,
    and reward/continue heads -- all trained by one call to ``model_loss()``.

    ``contdisc`` (constructor flag, default ``False`` -- plan decision 2):
    official JAX's default is ``True``, which multiplies the continue
    head's training target by ``(1 - 1/discount_horizon)`` and, as a
    consequence, uses ``disc=1`` (no separate discount factor) when building
    the actor-critic's imagination weight elsewhere (the discount is already
    baked into what the continue head itself predicts). r2dreamer has no
    such switch: its continue target is always the plain ``1 - is_terminal``,
    and ``disc = 1 - 1/discount_horizon`` is instead applied outside the
    continue head, when building that same imagination weight. This
    class's default (``False``) matches r2dreamer's behaviour; setting it
    ``True`` matches official JAX's. Either way, the caller (the owning
    algorithm's actor-critic update, plan section F) is responsible for
    picking ``disc`` (``1`` vs. ``1 - 1/discount_horizon``) consistently
    with this flag -- this class only changes the continue head's own
    training target.
    """

    def __init__(
        self,
        encoder: BaseFeaturesExtractor,
        schema: ObservationSchema,
        action_dim: int,
        size: RSSMSize,
        *,
        stoch: int = 32,
        unimix: float = 0.01,
        blocks: int = 8,
        obs_layers: int = 1,
        img_layers: int = 2,
        dyn_layers: int = 1,
        decoder_layers: int = 3,
        reward_bins: int = 255,
        kl_free: float = 1.0,
        contdisc: bool = False,
        discount_horizon: float = 333.0,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self._stoch = stoch
        self._discrete = size.discrete
        self._deter = size.deter
        self._unimix = unimix
        self.kl_free = kl_free
        self.contdisc = contdisc
        self.discount_horizon = discount_horizon

        self.flat_stoch = stoch * size.discrete
        self.latent_dim = self.flat_stoch + self._deter

        self._deter_net = _Deter(
            self._deter, self.flat_stoch, action_dim, size.hidden, blocks, dyn_layers, act
        )

        embed_size = encoder.features_dim
        obs_net: list[nn.Module] = []
        inp = self._deter + embed_size
        for _ in range(obs_layers):
            obs_net += [nn.Linear(inp, size.hidden, bias=True), RMSNorm(size.hidden), act()]
            inp = size.hidden
        obs_net.append(nn.Linear(inp, self.flat_stoch, bias=True))
        self._obs_net = nn.Sequential(*obs_net)
        self._obs_net.apply(dreamer_weight_init_)

        img_net: list[nn.Module] = []
        inp = self._deter
        for _ in range(img_layers):
            img_net += [nn.Linear(inp, size.hidden, bias=True), RMSNorm(size.hidden), act()]
            inp = size.hidden
        img_net.append(nn.Linear(inp, self.flat_stoch, bias=True))
        self._img_net = nn.Sequential(*img_net)
        self._img_net.apply(dreamer_weight_init_)

        self.reward_head = MLPHead(
            self.latent_dim, reward_bins, output="symexp_twohot", layers=1, units=size.units, outscale=0.0
        )
        self.cont_head = MLPHead(
            self.latent_dim, 1, output="binary", layers=1, units=size.units, outscale=1.0
        )
        self.decoder = ObservationDecoder(
            schema,
            self._deter,
            self.flat_stoch,
            depth=size.cnn_depth,
            units=size.units,
            state_layers=decoder_layers,
        )

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def encode(self, obs: Obs) -> torch.Tensor:
        return self.encoder.extract(obs)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        deter = torch.zeros(batch_size, self._deter, device=device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, device=device)
        return {"deter": deter, "stoch": stoch, "logits": torch.zeros_like(stoch)}

    def _prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        return self._img_net(deter).reshape(*deter.shape[:-1], self._stoch, self._discrete)

    def _posterior_logits(self, deter: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, embed], dim=-1)
        return self._obs_net(x).reshape(*deter.shape[:-1], self._stoch, self._discrete)

    def observe(
        self, state: State, action: Optional[torch.Tensor], embed: torch.Tensor, is_first: torch.Tensor
    ) -> State:
        if action is None:
            raise ValueError(
                "RSSM.observe requires a real action tensor (zero it at true episode "
                "starts yourself, or rely on is_first masking below) -- unlike "
                "LatentConsistencyModel, RSSM has a recurrent carry that genuinely "
                "depends on the incoming action."
            )
        mask = is_first.bool()
        deter = torch.where(
            mask.reshape(*mask.shape, 1), torch.zeros_like(state["deter"]), state["deter"]
        )
        stoch = torch.where(
            mask.reshape(*mask.shape, 1, 1), torch.zeros_like(state["stoch"]), state["stoch"]
        )
        action = torch.where(mask.reshape(*mask.shape, 1), torch.zeros_like(action), action)

        stoch_flat = stoch.reshape(*stoch.shape[:-2], -1)
        new_deter = self._deter_net(stoch_flat, deter, action)
        posterior_logits = self._posterior_logits(new_deter, embed)
        new_stoch = OneHotDist(posterior_logits, unimix=self._unimix).rsample()
        return {"deter": new_deter, "stoch": new_stoch, "logits": posterior_logits}

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        stoch_flat = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        new_deter = self._deter_net(stoch_flat, state["deter"], action)
        prior_logits = self._prior_logits(new_deter)
        dist = OneHotDist(prior_logits, unimix=self._unimix)
        new_stoch = dist.rsample() if sample else dist.mode
        return {"deter": new_deter, "stoch": new_stoch, "logits": prior_logits}

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        """Raw symexp-twohot bin logits (``WorldModel.reward``'s documented
        contract) -- decode with ``rl_garden.networks.twohot.twohot_mean``.
        Ignores ``action`` (reward/continue heads read only ``feat``, see
        module docstring)."""
        del action
        feat = self.features(state)
        return self.reward_head.last(self.reward_head.mlp(feat))

    def continue_(self, state: State) -> torch.Tensor:
        """Continuation probability, shape ``(..., 1)`` (unsqueezed, matching
        ``LatentConsistencyModel.continue_``'s own shape convention)."""
        feat = self.features(state)
        logits = self.cont_head.last(self.cont_head.mlp(feat))
        return torch.sigmoid(logits)

    def decode(self, state: State) -> dict[str, object]:
        stoch_flat = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        return self.decoder(state["deter"], stoch_flat)

    def features(self, state: State) -> torch.Tensor:
        stoch_flat = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        return torch.cat([state["deter"], stoch_flat], dim=-1)

    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        model_params = itertools.chain(
            self._deter_net.parameters(),
            self._obs_net.parameters(),
            self._img_net.parameters(),
            self.reward_head.parameters(),
            self.cont_head.parameters(),
            self.decoder.parameters(),
        )
        return {"encoder": self.encoder.parameters(), "model": model_params}

    def model_loss(self, batch) -> tuple[dict[str, torch.Tensor], State]:
        """Trains the world model over a ``DreamerSequenceBatch``
        (``rl_garden.buffers.sequence_replay_buffer``, plan section D):
        ``horizon`` posterior steps warm-started from ``batch.carry``,
        dyn/rep KL (free bits ``self.kl_free``), per-key reconstruction
        (``self.decoder``), reward (``symexp_twohot``), and continue
        (``binary``, target ``contdisc``-aware -- see class docstring).

        Every returned loss is already mean-reduced but NOT scaled by the
        loss-scale table (``dyn=1.0, rep=0.1, recon=1.0/key, rew=1.0,
        con=1.0`` -- r2dreamer/JAX ``loss_scales``, scratchpad
        ``dreamer-code-survey.md`` section 3) -- the owning algorithm applies
        those and sums (``WorldModel.model_loss``'s own docstring; this
        repo's model-based-base convention, see ``ModelBasedAlgorithm``).

        Prior logits for the KL are computed ONCE, batched over the whole
        ``(horizon, B)`` posterior sequence AFTER the per-step ``observe()``
        loop (r2dreamer ``dreamer.py:369-371``) -- not per-step inside the
        loop, since nothing in ``observe()`` itself needs them (only the KL
        loss, computed here, does).

        Returns ``(losses, {"deter", "stoch", "logits"})`` where every
        tensor is ``(horizon, B, ...)`` and LIVE (graph-attached, see
        ``WorldModel.model_loss``'s docstring) -- ``horizon`` rows only
        (batch.obs/carry's row 0 seeds the recurrence but is never itself a
        loss row, r2dreamer ``buffer.py``'s exact convention).
        """
        horizon = batch.action.shape[0]

        deter0 = batch.carry["deter"]
        stoch0 = batch.carry["stoch"].reshape(*batch.carry["stoch"].shape[:-1], self._stoch, self._discrete)
        state: State = {"deter": deter0, "stoch": stoch0, "logits": torch.zeros_like(stoch0)}

        # Slice obs to rows 1..horizon BEFORE encoding -- row 0 only seeds
        # batch.carry (never observed itself, see this method's docstring),
        # so encoding it too and then discarding the result (self.encode
        # (batch.obs)[1:]) wastes a full encoder forward pass on data never
        # used.
        embed = self.encode(index_obs(batch.obs, slice(1, None)))  # (horizon, B, embed_dim)

        deters, stochs, post_logits_steps = [], [], []
        for t in range(horizon):
            state = self.observe(state, batch.action[t], embed[t], batch.is_first[t])
            deters.append(state["deter"])
            stochs.append(state["stoch"])
            post_logits_steps.append(state["logits"])
        post_deter = torch.stack(deters, dim=0)
        post_stoch = torch.stack(stochs, dim=0)
        post_logits = torch.stack(post_logits_steps, dim=0)

        prior_logits = self._prior_logits(post_deter)
        dyn_loss, rep_loss = kl_loss(post_logits, prior_logits, self.kl_free)

        stoch_flat = post_stoch.reshape(*post_stoch.shape[:-2], -1)
        feat = torch.cat([post_deter, stoch_flat], dim=-1)

        losses: dict[str, torch.Tensor] = {
            "dyn": dyn_loss.mean(),
            "rep": rep_loss.mean(),
        }

        for key, dist in self.decoder(post_deter, stoch_flat).items():
            target = batch.obs[key][1:]
            if key.startswith("rgb_"):
                target = target.float() / 255.0
            losses[key] = -dist.log_prob(target).mean()

        reward_dist = self.reward_head(feat)
        losses["rew"] = -reward_dist.log_prob(batch.reward.unsqueeze(-1)).mean()

        if self.contdisc:
            cont_target = (1.0 - batch.is_terminal.float()) * (1.0 - 1.0 / self.discount_horizon)
        else:
            cont_target = 1.0 - batch.is_terminal.float()
        cont_dist = self.cont_head(feat)
        losses["con"] = -cont_dist.log_prob(cont_target.unsqueeze(-1)).mean()

        return losses, {"deter": post_deter, "stoch": post_stoch, "logits": post_logits}
