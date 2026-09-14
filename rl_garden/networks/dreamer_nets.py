"""DreamerV3 building blocks shared by ``rl_garden.world_models.rssm.RSSM``
and ``rl_garden.world_models.decoder.ObservationDecoder`` (model-based-base
plan Part 2, section A): normalization, the block-diagonal dynamics/decoder
linear layer, weight init, the generic MLP builder, return normalization,
and every distribution head DreamerV3 uses. Ported numerically from
``3rd_party/r2dreamer/{networks,distributions,tools}.py``; official JAX
(``3rd_party/dreamerv3/dreamerv3/{nets,heads}.py``) confirms every constant
below is identical between the two reference implementations (scratchpad
``dreamer-code-survey.md`` sections 1, 3, 7).

Every module here is autocast-safe (no float64, no in-place ops on autocast
outputs) since the owning ``RSSM``/heads are wrapped in ``torch.autocast``
by the algorithm (``compute_dtype``, decision 5) -- this file itself does
not touch autocast.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.networks.symlog import symexp, symlog
from rl_garden.networks.twohot import symexp_twohot_bins, twohot_logprob, twohot_mean

# ----------------------------------------------------------------------
# Normalization / weight init
# ----------------------------------------------------------------------


class RMSNorm(nn.RMSNorm):
    """DreamerV3's normalization everywhere (NOT LayerNorm) -- ``eps=1e-4``,
    parameters kept ``float32`` even under bf16/fp16 autocast: ``nn.RMSNorm``
    computes its statistics in the input's dtype but the weight itself
    staying float32 is load-bearing for numerical stability under AMP, not
    incidental (r2dreamer/JAX both keep every norm's scale in float32,
    ``networks.py`` / ``configs.yaml``'s ``norm: rms``)."""

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__(dim, eps=eps, dtype=torch.float32)


def dreamer_weight_init_(module: nn.Module) -> None:
    """DreamerV3's default weight init, ported verbatim from r2dreamer's
    ``weight_init_`` (``tools.py:114-134``; identical constant in official
    JAX ``nets.py:168-170``): an ``RMSNorm``'s weight is filled with 1; every
    other module exposing a ``.weight`` gets a truncated-normal draw with
    ``std = 1.1368 * sqrt(1 / fan_in)``, truncated to ``+/- 2*std`` (the
    constant ``1.1368`` is exact in both reference implementations); a
    ``.bias`` (if any) is zeroed.

    Apply via ``module.apply(dreamer_weight_init_)`` over a whole submodule
    tree, not on a single leaf directly -- container modules
    (``nn.Sequential``, etc.) have no ``.weight`` of their own and are
    silently skipped. ``BlockLinear``'s weight layout
    (``(out_ch // blocks, in_ch // blocks, blocks)``) is deliberately chosen
    so ``nn.init._calculate_fan_in_and_fan_out`` on it still reports the
    layer's FULL ``in_ch`` as ``fan_in`` (not one block's share) -- see that
    class's docstring.
    """
    if isinstance(module, nn.RMSNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    std = 1.1368 * math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


def _apply_outscale(module: nn.Module, outscale: float) -> None:
    """Post-init output-layer rescale (reward/critic ``outscale=0.0``,
    actor ``outscale=0.01``, decoder ``outscale=1.0`` == no-op) -- a
    targeted rescale of one already-initialized layer's (``nn.Linear`` or
    ``BlockLinear``) weight, not part of ``dreamer_weight_init_`` itself
    (r2dreamer ``networks.py:369-372``)."""
    if outscale != 1.0:
        with torch.no_grad():
            module.weight.mul_(outscale)


# ----------------------------------------------------------------------
# Block-diagonal linear layer
# ----------------------------------------------------------------------


class BlockLinear(nn.Module):
    """Block-diagonal linear layer: ``blocks`` independent
    ``(in_ch // blocks) -> (out_ch // blocks)`` linear maps applied in
    parallel over a shared last dim, ported verbatim from r2dreamer
    ``networks.py:24-56`` (identical block-diagonal einsum in official JAX
    ``nets.py:254-278``). Used for the RSSM's block-GRU dynamics core and
    the decoder's deterministic-state space projection -- ``blocks=8`` is
    DreamerV3's fixed choice (plan decision), independent of hidden size.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0) -> None:
        super().__init__()
        if in_ch % blocks != 0 or out_ch % blocks != 0:
            raise ValueError(
                "BlockLinear requires in_ch and out_ch divisible by blocks; got "
                f"in_ch={in_ch}, out_ch={out_ch}, blocks={blocks}."
            )
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.blocks = blocks
        # Weight layout (O/G, I/G, G): see dreamer_weight_init_'s docstring
        # for why this ordering (not (G, O/G, I/G)) is load-bearing.
        self.weight = nn.Parameter(torch.empty(out_ch // blocks, in_ch // blocks, blocks))
        self.bias = nn.Parameter(torch.empty(out_ch))
        dreamer_weight_init_(self)
        _apply_outscale(self, outscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        return x.reshape(*batch_shape, self.out_ch) + self.bias


# ----------------------------------------------------------------------
# Generic MLP builder
# ----------------------------------------------------------------------


class MLP(nn.Module):
    """``layers`` blocks of ``Linear -> RMSNorm -> act``, with an optional
    ``symlog`` applied to the raw input first -- r2dreamer's generic MLP
    (``networks.py:313-336``; JAX's equivalent inline in ``nets.py``).
    ``symlog_inputs=True`` for encoder MLPs (state observations), ``False``
    for decoder MLPs (the decoder's own symlog lives on its OUTPUT
    distribution, see ``SymlogMSEDist`` below, not its input).

    Does not run ``dreamer_weight_init_`` on itself -- matching r2dreamer,
    where every leaf builder (``MLP``, ``BlockLinear`` aside, which inits at
    construction since it has no natural owner to defer to) leaves init to
    the owning container (``MLPHead``, an encoder, a decoder) that applies
    it once over the whole submodule tree after every piece is built.
    """

    def __init__(
        self,
        inp_dim: int,
        units: int,
        layers: int,
        *,
        symlog_inputs: bool = False,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.symlog_inputs = symlog_inputs
        modules: list[nn.Module] = []
        dim = inp_dim
        for _ in range(layers):
            modules.append(nn.Linear(dim, units, bias=True))
            modules.append(RMSNorm(units))
            modules.append(act())
            dim = units
        self.layers = nn.Sequential(*modules)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symlog_inputs:
            x = symlog(x)
        return self.layers(x)


# ----------------------------------------------------------------------
# Return normalization
# ----------------------------------------------------------------------


class ReturnEMA(nn.Module):
    """Percentile-based return-scale EMA (r2dreamer ``ReturnEMA``,
    ``networks.py:390-406``; official JAX ``retnorm``, ``impl='perc'``):
    tracks the ``[5, 95]``-percentile range of imagined returns with an EMA
    of rate ``0.01``, and reports ``(offset, scale)`` where ``scale =
    max(1, p95 - p05)`` -- used to normalize the actor's advantage, not to
    renormalize the critic's own regression target (scratchpad
    ``dreamer-code-survey.md`` section 4).

    ``x`` is cast to float32 before ``torch.quantile`` regardless of its
    input dtype: quantile is unreliable in bf16/fp16, and this module may be
    called under autocast.
    """

    def __init__(self, rate: float = 1e-2) -> None:
        super().__init__()
        self.rate = rate
        self.register_buffer("_percentiles", torch.tensor([0.05, 0.95]))
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_quantile = torch.quantile(torch.flatten(x.detach().float()), self._percentiles)
        self.ema_vals.copy_(self.rate * x_quantile + (1 - self.rate) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()


# ----------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------


class OneHotDist(torch.distributions.OneHotCategorical):
    """RSSM stochastic-state distribution: one-hot categorical with
    ``unimix`` probability mass mixed in uniformly (r2dreamer
    ``OneHotDist``, ``distributions.py:16-36``; identical in official JAX).
    ``.mode`` and ``.rsample()`` are both straight-through (the hard one-hot
    forward value, gradient flowing through the softmax/logits instead) --
    this is what lets the RSSM's discrete stochastic state participate in a
    differentiable rollout at all. NOT used for the discrete action head
    (decision 4: that one is a plain ``torch.distributions.Categorical``,
    no unimix, matching official JAX's ``policy_dist_disc: categorical``).
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.0) -> None:
        probs = F.softmax(logits.float(), dim=-1)
        uniform = unimix / probs.shape[-1]
        probs = probs * (1.0 - unimix) + uniform
        super().__init__(logits=torch.log(probs))

    @property
    def mode(self) -> torch.Tensor:
        hard = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[-1]).to(self.logits.dtype)
        return hard.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 1.0) -> torch.Tensor:
        del sample_shape
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        raise NotImplementedError("OneHotDist is straight-through only -- use rsample().")


def bounded_normal(x: torch.Tensor, min_std: float = 0.1, max_std: float = 1.0) -> torch.distributions.Independent:
    """Continuous action head (decision 4): ``Normal(tanh(mean), std)``,
    ``std = (max_std - min_std) * sigmoid(raw_std + 2.0) + min_std`` --
    ``tanh`` bounds the mean, NOT a tanh-squashed sample (unlike SAC's
    TanhNormal, the sampled action itself is not squashed through tanh
    afterward; the caller clips samples to the action bounds before the env
    instead, this port's own choice per decision 4). Ported verbatim from
    r2dreamer ``distributions.py:217-222`` (identical in official JAX
    ``heads.py:146-155``). ``x`` is the raw ``2 * action_dim`` linear
    output, split evenly into ``(mean, raw_std)``.
    """
    mean, raw_std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(raw_std + 2.0) + min_std
    return torch.distributions.Independent(torch.distributions.Normal(torch.tanh(mean), std), 1)


def _sum_trailing_dims(x: torch.Tensor) -> torch.Tensor:
    """Sums every dim beyond the first two (matches r2dreamer's
    ``MSEDist``/``SymlogDist`` convention, ``distributions.py:150-152,185-
    189``, of treating the first two dims as (time, batch) and reducing
    everything else -- a scalar-shaped ``(T, B)`` input is returned as-is)."""
    if x.dim() <= 2:
        return x
    return x.sum(dim=list(range(2, x.dim())))


class MSEDist:
    """Distribution-like decoder-head wrapper (r2dreamer ``MSEDist``,
    ``distributions.py:132-155``): ``mode``/``mean`` is the raw prediction,
    ``log_prob`` is negative squared error. Used for image reconstruction
    (DreamerV3 does not use symlog on pixel targets, only on vector/state
    targets -- ``SymlogMSEDist`` below, see ``dreamer-code-survey.md``
    section 2)."""

    def __init__(self, mode: torch.Tensor) -> None:
        self._mode = mode

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    @property
    def mean(self) -> torch.Tensor:
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -_sum_trailing_dims((self._mode - value) ** 2)


class SymlogMSEDist:
    """Distribution-like decoder-head wrapper (r2dreamer ``SymlogDist``,
    ``distributions.py:158-190``, ``dist="mse"`` branch): the network
    predicts in symlog space, ``mode``/``mean`` decode back with ``symexp``,
    and ``log_prob`` compares against ``symlog(value)`` -- DreamerV3's
    default for vector/state decoder targets (reconstruction) and the
    reward/continue targets that don't use ``symexp_twohot``. Differences
    smaller than ``1e-8`` are zeroed (matches upstream's own numerical-noise
    floor)."""

    _tol = 1e-8

    def __init__(self, mode: torch.Tensor) -> None:
        self._mode = mode  # symlog-space prediction

    @property
    def mode(self) -> torch.Tensor:
        return symexp(self._mode)

    @property
    def mean(self) -> torch.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        distance = (self._mode - symlog(value)) ** 2
        distance = torch.where(distance < self._tol, torch.zeros_like(distance), distance)
        return -_sum_trailing_dims(distance)


class TwoHotDist:
    """Distribution-like reward/critic-head wrapper around
    ``rl_garden.networks.twohot.twohot_logprob``/``twohot_mean`` (r2dreamer
    ``symexp_twohot``/``TwoHot``, ``distributions.py:67-129,242-251``) --
    255 symexp-spaced bins, ``outscale=0.0`` at construction (``MLPHead``'s
    job) so the head starts near a uniform distribution over bins."""

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor) -> None:
        self.logits = logits
        self.bins = bins

    @property
    def mode(self) -> torch.Tensor:
        return twohot_mean(self.logits, self.bins)

    @property
    def mean(self) -> torch.Tensor:
        return twohot_mean(self.logits, self.bins)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return twohot_logprob(self.logits, self.bins, value)


# ----------------------------------------------------------------------
# MLPHead
# ----------------------------------------------------------------------

_OUTPUT_KINDS = ("symexp_twohot", "binary", "symlog_mse", "mse", "bounded_normal", "categorical")


class MLPHead(nn.Module):
    """``MLP`` trunk + a final linear layer producing one of DreamerV3's
    distribution heads (r2dreamer ``MLPHead``, ``networks.py:339-377``):

    - ``"symexp_twohot"``: reward/critic heads, 255 symexp-spaced bins
      (``out_dim`` IS the bin count) -> ``TwoHotDist``.
    - ``"binary"``: continue head -> ``Independent(Bernoulli(logits), 1)``.
    - ``"symlog_mse"``: vector/state decoder heads -> ``SymlogMSEDist``.
    - ``"mse"``: image decoder heads -> ``MSEDist``.
    - ``"bounded_normal"``: continuous action head -> ``bounded_normal()``
      (``Independent(Normal, 1)``); the last linear layer outputs
      ``2 * out_dim`` (mean, raw_std).
    - ``"categorical"``: discrete action head (decision 4: no unimix) ->
      ``torch.distributions.Categorical``.

    ``outscale`` rescales the final linear layer's weight post-init (reward/
    critic ``0.0``, actor ``0.01``, decoder ``1.0`` == no-op) -- see
    ``_apply_outscale``.
    """

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        *,
        output: str,
        layers: int = 1,
        units: int = 256,
        outscale: float = 1.0,
        symlog_inputs: bool = False,
        min_std: float = 0.1,
        max_std: float = 1.0,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if output not in _OUTPUT_KINDS:
            raise ValueError(f"Unknown MLPHead output kind {output!r}; expected one of {_OUTPUT_KINDS}.")
        self.output = output
        self.out_dim = out_dim
        self.min_std = min_std
        self.max_std = max_std

        self.mlp = MLP(inp_dim, units, layers, symlog_inputs=symlog_inputs, act=act)
        last_dim = 2 * out_dim if output == "bounded_normal" else out_dim
        self.last = nn.Linear(self.mlp.out_dim, last_dim, bias=True)

        self.mlp.apply(dreamer_weight_init_)
        self.last.apply(dreamer_weight_init_)
        _apply_outscale(self.last, outscale)

        if output == "symexp_twohot":
            self.register_buffer("_bins", symexp_twohot_bins(out_dim), persistent=False)

    def forward(self, x: torch.Tensor):
        out = self.last(self.mlp(x))
        if self.output == "bounded_normal":
            return bounded_normal(out, self.min_std, self.max_std)
        if self.output == "categorical":
            return torch.distributions.Categorical(logits=out)
        if self.output == "binary":
            return torch.distributions.Independent(torch.distributions.Bernoulli(logits=out), 1)
        if self.output == "symexp_twohot":
            return TwoHotDist(out, self._bins)
        if self.output == "symlog_mse":
            return SymlogMSEDist(out)
        return MSEDist(out)  # output == "mse"
