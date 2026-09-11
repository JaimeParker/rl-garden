"""Flow-matching critic network for FloQ (Farebrother et al., "floq: Training
Critics Via Flow-Matching For Scaling Compute In Value-Based RL",
arXiv 2509.06863).

Ports floq's ``CriticVectorField`` (``3rd_party/floq/utils/networks.py``): an
ensemble of velocity-field MLPs over the scalar-return axis, trained with a
flow-matching TD loss and Euler-integrated (via ``integrate_returns``) both
to bootstrap a Polyak target and to produce a distillation target for the
plain scalar critic used by the actor's Q-loss (see
``rl_garden/algorithms/floq.py``).

Two deliberate deviations from the reference, per the project's floq port
decisions:

- ``FourierTimeEmbedding`` replaces the reference's scalar ``cos(t)`` time
  embedding, which silently ignores ``time_embed_dim`` -- this port implements
  a real sinusoidal Fourier embedding (half sin / half cos over log-spaced
  frequencies, standard diffusion-timestep-embedding form) that actually uses
  the configured dimension.
- No ``CriticResVectorField``/``use_resnets`` port: only the plain MLP
  velocity head is implemented (see the port's top-level plan).

Follows ``EnsembleQCritic``'s (``rl_garden/networks/actor_critic.py``)
``torch.func.stack_module_state`` + meta-prototype + ``functional_call``/
``vmap`` ensembling pattern and reuses its ``_safe_name``/``_PARAM_PREFIX``/
``_BUFFER_PREFIX`` naming convention for checkpoint-key stability. Unlike
``EnsembleQCritic``, ``returns`` here is itself ensemble-batched (each
ensemble member integrates its own return estimate), so the fused vmap
forward cannot reuse ``EnsembleQCritic._vmapped_forward`` -- it needs its own
version that vmaps over ``returns`` (``in_dims=0``) while broadcasting
``features``/``actions``/``times`` (``in_dims=None``).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.actor_critic import _BUFFER_PREFIX, _PARAM_PREFIX, _safe_name
from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: ``t (...,1) -> (...,dim)``, half sin / half
    cos over log-spaced frequencies (standard diffusion-timestep-embedding
    form). ``dim`` must be even."""

    def __init__(self, dim: int, max_period: float = 1e4) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim must be a positive even number, got {dim}.")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def compute_support(q_min: float, q_max: float, num_bins: int) -> torch.Tensor:
    return torch.linspace(q_min, q_max, num_bins)


def hl_gauss_to_probs(
    target: torch.Tensor, support: torch.Tensor, sigma_eff: torch.Tensor | float
) -> torch.Tensor:
    """HL-Gauss target distribution (erf-CDF bin-probability differences).

    ``target``: ``(N,)``. ``support``: ``(num_bins,)``. ``sigma_eff`` is the
    caller-multiplied ``sigma * bin_width`` (matches the reference's
    ``to_probs(returns, support, self.sigma*bin_width)``). Returns
    ``(N, num_bins - 1)``, rows summing to 1.
    """
    cdf_evals = torch.erf(
        (support - target.unsqueeze(-1)) / (math.sqrt(2.0) * sigma_eff)
    )
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / z


class _VelocityHead(nn.Module):
    """Single velocity-field MLP: concatenated ``(features, actions,
    returns_embed, times_embed) -> scalar velocity``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation_fn: Optional[Activation],
        use_layer_norm: bool,
        kernel_init: Optional[KernelInit],
    ) -> None:
        super().__init__()
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            net_arch=hidden_dims,
            activation_fn=resolve_activation(activation_fn, default=nn.ReLU),
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CriticVectorField(nn.Module):
    """Ensemble of velocity fields over the scalar-return axis.

    ``forward(features (B,fd), actions (B,ad), returns (E,B,1), times (B,1))
    -> (E,B,1)``: ``features``/``actions``/``times`` are shared across the
    ensemble (broadcast); ``returns`` carries a distinct value per ensemble
    member (vmapped).
    """

    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        num_ensembles: int = 1,
        use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        activation_fn: Optional[Activation] = None,
        embed_time: bool = True,
        time_embed_dim: int = 64,
        use_prob_embed: bool = True,
        q_min: float,
        q_max: float,
        num_bins: int = 51,
        sigma: float = 16.0,
    ) -> None:
        super().__init__()
        if num_ensembles < 1:
            raise ValueError(f"num_ensembles must be >= 1, got {num_ensembles}.")

        self.num_ensembles = num_ensembles
        self.embed_time = embed_time
        self.use_prob_embed = use_prob_embed
        self.q_min = q_min
        self.q_max = q_max
        self.num_bins = num_bins
        self.sigma = sigma

        if embed_time:
            self.time_embedding: Optional[FourierTimeEmbedding] = FourierTimeEmbedding(
                time_embed_dim
            )
            time_dim = time_embed_dim
        else:
            self.time_embedding = None
            time_dim = 1

        returns_dim = (num_bins - 1) if use_prob_embed else 1
        input_dim = features_dim + action_dim + returns_dim + time_dim

        self.register_buffer("support", compute_support(q_min, q_max, num_bins))

        head_kwargs = dict(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )
        self._head_kwargs = head_kwargs

        heads = [_VelocityHead(**head_kwargs) for _ in range(num_ensembles)]
        stacked_params, stacked_buffers = torch.func.stack_module_state(heads)

        self._dotted_param_names: list[str] = list(stacked_params.keys())
        for dotted in self._dotted_param_names:
            self.register_parameter(
                _safe_name(dotted, _PARAM_PREFIX),
                nn.Parameter(stacked_params[dotted].detach().clone()),
            )
        self._dotted_buffer_names: list[str] = list(stacked_buffers.keys())
        for dotted in self._dotted_buffer_names:
            self.register_buffer(
                _safe_name(dotted, _BUFFER_PREFIX),
                stacked_buffers[dotted].detach().clone(),
            )

        # Save/restore CPU RNG state so prototype init doesn't shift
        # downstream random ops, matching EnsembleQCritic's own convention.
        _cpu_rng = torch.get_rng_state()
        prototype = _VelocityHead(**head_kwargs)
        torch.set_rng_state(_cpu_rng)
        prototype.to("meta")
        object.__setattr__(self, "_prototype", prototype)

    def _gather_params(self) -> dict[str, torch.Tensor]:
        return {
            dotted: getattr(self, _safe_name(dotted, _PARAM_PREFIX))
            for dotted in self._dotted_param_names
        }

    def _gather_buffers(self) -> dict[str, torch.Tensor]:
        return {
            dotted: getattr(self, _safe_name(dotted, _BUFFER_PREFIX))
            for dotted in self._dotted_buffer_names
        }

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        obs_act = torch.cat([features, actions], dim=-1)

        if self.use_prob_embed:
            bin_width = self.support[1] - self.support[0]
            sigma_eff = self.sigma * bin_width
            num_ensembles, batch_size, _ = returns.shape
            flat_returns = returns.reshape(num_ensembles * batch_size)
            probs = hl_gauss_to_probs(flat_returns, self.support, sigma_eff)
            returns_embed = probs.reshape(num_ensembles, batch_size, self.num_bins - 1)
        else:
            returns_embed = (returns - self.q_min) / (self.q_max - self.q_min)

        if self.embed_time:
            assert self.time_embedding is not None
            times_embed = self.time_embedding(times)
        else:
            times_embed = times

        prototype = self._prototype

        def single(params, buffers, obs_act_, ret_embed, times_embed_):
            x = torch.cat([obs_act_, ret_embed, times_embed_], dim=-1)
            return torch.func.functional_call(prototype, (params, buffers), (x,))

        return torch.func.vmap(single, in_dims=(0, 0, None, 0, None))(
            self._gather_params(), self._gather_buffers(), obs_act, returns_embed, times_embed
        )


def integrate_returns(
    net: CriticVectorField,
    features: torch.Tensor,
    actions: torch.Tensor,
    noise_ratios: torch.Tensor,
    *,
    noise_min: float,
    noise_max: float,
    steps: int,
) -> torch.Tensor:
    """Euler-integrate ``net``'s velocity field from an ``x_0`` derived from
    ``noise_ratios`` (``noise_min*(1-r) + noise_max*r``) over ``steps``.

    ``features``/``actions``: ``(B, fd)``/``(B, ad)``. ``noise_ratios``:
    ``(B, 1)``. Returns ``(E, B, 1)``, unaggregated -- callers aggregate over
    the ensemble dimension themselves.
    """
    batch_size = features.shape[0]
    device, dtype = features.device, features.dtype
    x_0 = noise_min * (1.0 - noise_ratios) + noise_max * noise_ratios  # (B, 1)
    returns = x_0.unsqueeze(0).expand(net.num_ensembles, batch_size, 1)
    for step in range(steps):
        t = torch.full((batch_size, 1), step / steps, device=device, dtype=dtype)
        velocity = net(features, actions, returns, t)
        returns = returns + velocity / steps
    return returns
