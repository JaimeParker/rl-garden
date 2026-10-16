"""``RSSM(WorldModel)``: DreamerV3's recurrent state-space model, ported from r2dreamer's ``RSSM``/
``Deter`` (``rssm.py``; identical block-GRU/KL math in official JAX
``dreamerv3/rssm.py``).

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
from typing import Iterator, Literal, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.networks.dreamer_nets import BlockLinear, MLPHead, OneHotDist, RMSNorm, dreamer_weight_init_
from rl_garden.networks.ncrl_nets import NCRLBlockGRU, NCRLConvDecoder, NCRLConvEncoder, NCRLHead, ncrl_logits_mlp
from rl_garden.networks.symlog import symexp, symlog
from rl_garden.networks.twohot import twohot_mean
from rl_garden.observations.schema import ObservationSchema
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.decoder import ObservationDecoder


@dataclass(frozen=True)
class RSSMSize:
    """One row of DreamerV3's model-size table (r2dreamer
    ``configs/model/size{12,25,50,100,200,400}M.yaml``; identical in
    official JAX ``configs.yaml``'s ``size12m``..``size400m`` anchors). ``discrete`` is the
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
    # NCRL stage-1 pretraining (deviation from the official code: the 280M model scaled down): the
    # official 280M model (deter 12288 / hidden 1536 / 32x96 / cnn 96) scaled
    # down to fit one 24-48 GB card. ``units`` = ``hidden`` as in NCRL's
    # decoder (``units=hidden``).
    "ncrl_m": RSSMSize(deter=4096, hidden=1024, discrete=48, units=1024, cnn_depth=48),
}

RSSM_HEADS = ("reward", "continue")
RSSM_BACKBONES = ("dreamer_v3", "ncrl")


def build_rssm_encoder(
    backbone: str, observation_space: spaces.Dict, schema: ObservationSchema, size: RSSMSize
) -> BaseFeaturesExtractor:
    """The encoder that goes with ``RSSM(backbone=...)``: ``DreamerConvEncoder``
    or the official NCRL ``NCRLConvEncoder``."""
    if backbone == "ncrl":
        return NCRLConvEncoder(observation_space, schema, depth=size.cnn_depth)
    if backbone == "dreamer_v3":
        return DreamerConvEncoder(observation_space, schema, depth=size.cnn_depth, units=size.units)
    raise ValueError(f"backbone must be one of {RSSM_BACKBONES}, got {backbone!r}.")


def _categorical_kl(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:
    logprob_left = torch.log_softmax(logits_left, dim=-1)
    logprob_right = torch.log_softmax(logits_right, dim=-1)
    prob_left = torch.softmax(logits_left, dim=-1)
    return (prob_left * (logprob_left - logprob_right)).sum(dim=-1)


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
    rep_loss = _categorical_kl(post_logits, prior_logits.detach()).sum(dim=-1)
    dyn_loss = _categorical_kl(post_logits.detach(), prior_logits).sum(dim=-1)
    rep_loss = torch.clamp(rep_loss, min=free)
    dyn_loss = torch.clamp(dyn_loss, min=free)
    return dyn_loss, rep_loss


def kl_loss_mean_free(
    post_logits: torch.Tensor, prior_logits: torch.Tensor, free: float, unimix: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """NCRL's free-nats variant (``3rd_party/ncrl`` ``rssm.py:201-208``):
    same dyn/rep KL terms as :func:`kl_loss`, but averaged over every leading
    (batch/time) dim FIRST and only then clipped to ``free`` -- so the clip
    gates the batch-mean KL, not each element. Returns two scalars.

    ``unimix > 0`` mixes that fraction of uniform mass into both sides
    before the KL, as NCRL's ``_get_dist`` does (``rssm.py:59-66,126``);
    ``unimix == 0`` leaves the logits untouched."""
    if unimix > 0.0:
        post_logits = OneHotDist(post_logits, unimix=unimix).logits
        prior_logits = OneHotDist(prior_logits, unimix=unimix).logits
    rep_loss = _categorical_kl(post_logits, prior_logits.detach()).sum(dim=-1).mean()
    dyn_loss = _categorical_kl(post_logits.detach(), prior_logits).sum(dim=-1).mean()
    return torch.clamp(dyn_loss, min=free), torch.clamp(rep_loss, min=free)


def _masked_mean(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` ``(horizon, B, ...)`` over the ``(horizon, B)`` rows ``row_mask`` keeps."""
    mask = row_mask.to(values.dtype).reshape(row_mask.shape + (1,) * (values.dim() - row_mask.dim()))
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp(min=1)


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

    ``contdisc`` (constructor flag, default ``False``):
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
    algorithm's actor-critic update) is responsible for
    picking ``disc`` (``1`` vs. ``1 - 1/discount_horizon``) consistently
    with this flag -- this class only changes the continue head's own
    training target.

    NCRL options (defaults keep DreamerV3's behaviour):

    - ``heads``: which of ``("reward", "continue")`` to build. A head left out
      is never constructed (``self.reward_head``/``self.cont_head`` is
      ``None``), contributes no parameters and no ``model_loss`` key
      (``"rew"``/``"con"``). ``reward()``/``reward_value()`` then raise
      ``RuntimeError`` and ``continue_()`` returns ``None`` (the caller
      supplies its own discount, ``WorldModel.continue_``'s documented
      fallback). NCRL pretraining uses ``()``, fine-tuning ``("reward",)``.
    - ``reward_detached``: the reward loss reads ``features(posterior)
      .detach()`` so no reward gradient reaches the representation (NCRL).
    - ``kl_free_mode``: ``"per_element"`` clips each dyn/rep KL element to
      ``kl_free`` (DreamerV3, :func:`kl_loss`); ``"mean"`` averages over
      batch/time first and clips the mean, with ``unimix`` applied to both
      distributions first (as in NCRL, :func:`kl_loss_mean_free`).
    - ``reward_layers``/``reward_units``: reward-head depth and width
      (``units=None`` means ``size.units``). NCRL uses 2 x 512.
    - ``reward_twohot_space``: ``"raw"`` (DreamerV3) or ``"symlog"`` (NCRL's
      ``SymTwoHot``) two-hot for the reward head, see ``TwoHotDist``.
    - ``straight_through``: the stochastic-state ``rsample`` gradient,
      ``"gumbel_hard"`` (DreamerV3) or ``"probs"`` (NCRL), see ``OneHotDist``.
    - ``backbone``: ``"dreamer_v3"`` (default) or ``"ncrl"``, the official NCRL
      networks (``rl_garden.networks.ncrl_nets``): LayerNorm and torch default
      init, a block-GRU without action normalization, prior/posterior logits
      heads, ``NCRLHead`` reward/continue heads (``reward_units``/
      ``reward_layers`` as above) and the transposed-conv ``NCRLConvDecoder``
      (image-only, 64x64, hidden width ``size.hidden``; ``decoder_layers`` is
      unused). Pair it with an ``NCRLConvEncoder``. ``features()`` is the same
      ``cat[deter, stoch_flat]``; ``NCRLHead`` splits it again.
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
        heads: tuple[str, ...] = RSSM_HEADS,
        reward_detached: bool = False,
        kl_free_mode: Literal["per_element", "mean"] = "per_element",
        reward_layers: int = 1,
        reward_units: Optional[int] = None,
        reward_twohot_space: Literal["raw", "symlog"] = "raw",
        straight_through: Literal["gumbel_hard", "probs"] = "gumbel_hard",
        backbone: Literal["dreamer_v3", "ncrl"] = "dreamer_v3",
    ) -> None:
        super().__init__()
        unknown_heads = set(heads) - set(RSSM_HEADS)
        if unknown_heads:
            raise ValueError(f"Unknown RSSM heads {sorted(unknown_heads)}; choose from {RSSM_HEADS}.")
        if kl_free_mode not in ("per_element", "mean"):
            raise ValueError(f"kl_free_mode must be 'per_element' or 'mean', got {kl_free_mode!r}.")
        if straight_through not in ("gumbel_hard", "probs"):
            raise ValueError(f"straight_through must be 'gumbel_hard' or 'probs', got {straight_through!r}.")
        if backbone not in RSSM_BACKBONES:
            raise ValueError(f"backbone must be one of {RSSM_BACKBONES}, got {backbone!r}.")
        self.backbone = backbone
        self.heads = tuple(heads)
        self.reward_detached = reward_detached
        self.kl_free_mode = kl_free_mode
        self.straight_through = straight_through
        self.encoder = encoder
        self._stoch = stoch
        self._discrete = size.discrete
        self._deter = size.deter
        self._unimix = unimix
        self.blocks = blocks
        self.kl_free = kl_free
        self.contdisc = contdisc
        self.discount_horizon = discount_horizon

        self.flat_stoch = stoch * size.discrete
        self.latent_dim = self.flat_stoch + self._deter

        if backbone == "ncrl":
            self._build_ncrl_networks(
                schema, action_dim, size, blocks, obs_layers, img_layers, dyn_layers, reward_bins,
                reward_layers, reward_units, reward_twohot_space, act,
            )
            return

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

        self.reward_head: Optional[MLPHead] = None
        if "reward" in self.heads:
            self.reward_head = MLPHead(
                self.latent_dim,
                reward_bins,
                output="symexp_twohot",
                layers=reward_layers,
                units=size.units if reward_units is None else reward_units,
                outscale=0.0,
                twohot_space=reward_twohot_space,
            )
        self.cont_head: Optional[MLPHead] = None
        if "continue" in self.heads:
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

    def _build_ncrl_networks(
        self,
        schema: ObservationSchema,
        action_dim: int,
        size: RSSMSize,
        blocks: int,
        obs_layers: int,
        img_layers: int,
        dyn_layers: int,
        reward_bins: int,
        reward_layers: int,
        reward_units: Optional[int],
        reward_twohot_space: Literal["raw", "symlog"],
        act: type[nn.Module],
    ) -> None:
        """``backbone="ncrl"``: the same attributes as the DreamerV3 branch,
        built from ``rl_garden.networks.ncrl_nets``."""
        deter, flat_stoch = self._deter, self.flat_stoch
        self._deter_net = NCRLBlockGRU(deter, flat_stoch, action_dim, size.hidden, blocks, dyn_layers, act)
        self._obs_net = ncrl_logits_mlp(deter + self.encoder.features_dim, size.hidden, obs_layers, flat_stoch, act)
        self._img_net = ncrl_logits_mlp(deter, size.hidden, img_layers, flat_stoch, act)
        head_units = size.units if reward_units is None else reward_units
        self.reward_head = None
        if "reward" in self.heads:
            self.reward_head = NCRLHead(
                deter, flat_stoch, reward_bins, output="symexp_twohot", layers=reward_layers,
                units=head_units, blocks=blocks, act=act, twohot_space=reward_twohot_space,
            )
        self.cont_head = None
        if "continue" in self.heads:
            self.cont_head = NCRLHead(
                deter, flat_stoch, 1, output="binary", layers=1, units=head_units, blocks=blocks, act=act
            )
        self.decoder = NCRLConvDecoder(
            schema, deter, flat_stoch, depth=size.cnn_depth, units=size.hidden, blocks=blocks, act=act
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
        new_stoch = OneHotDist(posterior_logits, self._unimix, self.straight_through).rsample()
        return {"deter": new_deter, "stoch": new_stoch, "logits": posterior_logits}

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        stoch_flat = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        new_deter = self._deter_net(stoch_flat, state["deter"], action)
        prior_logits = self._prior_logits(new_deter)
        dist = OneHotDist(prior_logits, self._unimix, self.straight_through)
        new_stoch = dist.rsample() if sample else dist.mode
        return {"deter": new_deter, "stoch": new_stoch, "logits": prior_logits}

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        """Raw two-hot bin logits (``WorldModel.reward``'s documented
        contract) -- decode with :meth:`reward_value`.
        Ignores ``action`` (reward/continue heads read only ``feat``, see
        module docstring). Raises ``RuntimeError`` when built without the
        reward head (``heads`` excludes ``"reward"``)."""
        del action
        self._require_reward_head()
        feat = self.features(state)
        return self.reward_head.last(self.reward_head.mlp(feat))

    def reward_value(
        self, state: State, action: Optional[torch.Tensor] = None, *, symlog_space: bool = False
    ) -> torch.Tensor:
        """Scalar reward ``(..., 1)``, identical to ``reward_head(feat).mode``.
        With ``symlog_space=True`` the same probabilities are averaged over
        the bins' symlog positions, i.e. ``E_p[symlog(bin)]`` -- the mean in
        symlog space, which is not ``symlog`` of the real-scale mean.

        ``reward_twohot_space="raw"``: ``E_p[bin]`` over the symexp-spaced
        bins. ``"symlog"``: ``symexp(E_p[bin])`` over the symlog-spaced bins
        (``E_p[bin]`` itself when ``symlog_space=True``). Ignores ``action``."""
        self._require_reward_head()
        logits = self.reward(state, action)
        bins = self.reward_head._bins
        if self.reward_head.twohot_space == "symlog":
            mean = twohot_mean(logits, bins)
            return mean if symlog_space else symexp(mean)
        if symlog_space:
            bins = symlog(bins)
        return twohot_mean(logits, bins)

    def _require_reward_head(self) -> None:
        if self.reward_head is None:
            raise RuntimeError("This RSSM was built without a reward head (heads excludes 'reward').")

    def continue_(self, state: State) -> Optional[torch.Tensor]:
        """Continuation probability, shape ``(..., 1)`` (unsqueezed, matching
        ``LatentConsistencyModel.continue_``'s own shape convention), or
        ``None`` when built without the continue head."""
        if self.cont_head is None:
            return None
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
        modules = [self._deter_net, self._obs_net, self._img_net, self.reward_head, self.cont_head, self.decoder]
        model_params = itertools.chain.from_iterable(m.parameters() for m in modules if m is not None)
        return {"encoder": self.encoder.parameters(), "model": model_params}

    def model_loss(self, batch: SequenceBatch) -> tuple[dict[str, torch.Tensor], State]:
        """Trains the world model over a ``SequenceBatch`` -- in practice a
        ``DreamerSequenceBatch`` (``rl_garden.buffers.sequence_replay_buffer``,), reading ``obs``/``action``/``reward``/``is_first``/
        ``terminated``/``carry`` and the optional ``reward_valid`` (``(B,)``
        bool; ``None`` = every column labelled, otherwise the reward loss
        averages over the labelled columns only). ``reward`` may be
        ``(horizon, B)`` (the buffer's layout) or ``(horizon, B, 1)``:
        ``horizon`` posterior steps warm-started from ``batch.carry``,
        dyn/rep KL (free bits ``self.kl_free``), per-key reconstruction
        (``self.decoder``), reward (``symexp_twohot``), and continue
        (``binary``, target ``contdisc``-aware -- see class docstring). The
        ``"rew"``/``"con"`` keys appear only when that head was built
        (``heads``); ``batch.terminated`` is read only for the continue head.

        Every returned loss is already mean-reduced but NOT scaled by the
        loss-scale table (``dyn=1.0, rep=0.1, recon=1.0/key, rew=1.0,
        con=1.0`` -- r2dreamer/JAX ``loss_scales``) -- the owning algorithm applies
        those and sums (``WorldModel.model_loss``'s own docstring; this
        repo's convention, see ``ModelBasedAlgorithm``).

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
        if self.kl_free_mode == "mean":
            dyn_loss, rep_loss = kl_loss_mean_free(post_logits, prior_logits, self.kl_free, self._unimix)
        else:
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

        # A loss row whose arriving transition ended the episode is a reset
        # observation; its reward/terminated belong to the previous episode.
        is_last = getattr(batch, "is_last", None)
        row_valid = None if is_last is None else ~is_last.bool()

        if self.reward_head is not None:
            reward = batch.reward if batch.reward.dim() == batch.action.dim() else batch.reward.unsqueeze(-1)
            reward_dist = self.reward_head(feat.detach() if self.reward_detached else feat)
            reward_nll = -reward_dist.log_prob(reward)
            if row_valid is not None:
                reward_mask = row_valid if batch.reward_valid is None else row_valid & batch.reward_valid
                losses["rew"] = _masked_mean(reward_nll, reward_mask)
            elif batch.reward_valid is None:
                losses["rew"] = reward_nll.mean()
            else:
                losses["rew"] = reward_nll[:, batch.reward_valid].mean()

        if self.cont_head is not None:
            terminated = batch.terminated.float()
            if terminated.dim() == batch.action.dim():
                terminated = terminated.squeeze(-1)
            if self.contdisc:
                cont_target = (1.0 - terminated) * (1.0 - 1.0 / self.discount_horizon)
            else:
                cont_target = 1.0 - terminated
            cont_dist = self.cont_head(feat)
            cont_nll = -cont_dist.log_prob(cont_target.unsqueeze(-1))
            losses["con"] = cont_nll.mean() if row_valid is None else _masked_mean(cont_nll, row_valid)

        return losses, {"deter": post_deter, "stoch": post_stoch, "logits": post_logits}
