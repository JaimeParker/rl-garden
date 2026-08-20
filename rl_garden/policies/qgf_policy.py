"""QGF policy: BC flow-matching actor + IQL-style critic/value, with five
inference-time action-selection modes (``sampling_mode``).

Unlike FQL/ACFQL, QGF's actor is never trained with an RL objective -- it is
pure BC (see ``QGFCore._compute_losses``). All "policy improvement" happens
at inference time in ``predict()``:

- ``"guided"`` (QGF itself, ``qgf.py:157-246``): at each Euler denoising
  step, adds ``guidance_weight * qgrad`` to the BC velocity, where ``qgrad``
  is the critic's Q-gradient evaluated at a one-Euler-step approximation of
  the clean action (or at the raw noisy action, if
  ``denoised_action_approx="noisy"``).
- ``"grad_step"`` (GradStep baseline, ``agents/grad_step.py``): denoise fully
  via plain BC first, then run ``qgrad_steps`` of post-hoc gradient ascent
  directly in clean action space.
- ``"best_of_n"`` (IFQL baseline, ``agents/ifql.py``): draw
  ``actor_num_samples`` independent plain-BC denoising trajectories and keep
  the one with the highest critic Q -- no gradient guidance at all.
- ``"bptt"`` (BPTT baseline, ``agents/bptt.py``, **reconstructed** -- see
  below): at each Euler denoising step, runs the *full remaining-steps* BC
  rollout to a clean action, then backprops the Q-gradient through that
  *entire* rollout back to the current noisy point -- true backprop-through-
  time, no Jacobian approximation. O(``denoise_steps``²) network calls.
- ``"robust_q"`` (RobustQ baseline, ``agents/robust_q.py``): guidance from a
  separate, *noise-conditioned* critic ``Q_robust(s, a_t, t)`` trained to
  regress onto the clean target-critic Q -- evaluated directly at the noisy
  point (no approximation needed, since it's trained to live there).

``sampling_mode``/``guidance_weight``/``denoised_action_approx``/
``qgrad_step_size``/``qgrad_steps``/``use_sign_gradient``/``actor_num_samples``
are plain settable attributes (not construction-only): the paper's entire
premise is "train once, sweep these at eval time," so
``examples/eval_checkpoint.py`` can override them on a loaded checkpoint
without retraining.

Gradient-at-inference note: rl-garden's rollout/eval path calls ``predict()``
under the caller's ``torch.no_grad()`` (see ``off_policy.py``,
``fql_policy.py``'s docstring). QGF is the first algorithm here that needs a
gradient *during* inference, so ``predict()`` manages its own no_grad/
enable_grad boundaries rather than relying on the caller: the denoising loop
runs under an explicit ``torch.no_grad()``, and only the small per-step
Q-gradient computation locally re-enables autograd via
``torch.enable_grad()`` + a fresh leaf tensor. Critically, that gradient is
computed with ``torch.autograd.grad(q.sum(), a_leaf)``, never
``q.sum().backward()`` -- the latter would accumulate into
``critic.weight.grad``, silently corrupting the next training step whenever
guided sampling runs during periodic eval inside a training run. ``"bptt"``'s
gradient follows the exact same rule, just with a more expensive forward
pass (a multi-step rollout) inside the ``enable_grad()`` block.

``"bptt"`` and ``"robust_q"`` are **best-effort reconstructions** of code
broken in the upstream reference (``agents/bptt.py:48`` calls a
``self._bc_flow_from`` that does not exist anywhere in the tree;
``agents/robust_q.py:148`` references an undefined ``cfg``), not faithful
ports -- see each mode's own docstring below for exactly what was
reconstructed and why.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    ActorVectorField,
    Activation,
    BackboneType,
    EnsembleQCritic,
    KernelInit,
    ValueNetwork,
)
from rl_garden.networks.diffusion_mlp import _SinusoidalPosEmb
from rl_garden.policies.base import BasePolicy

SamplingMode = Literal["guided", "grad_step", "best_of_n", "bptt", "robust_q"]
DenoisedActionApprox = Literal["one_euler_step_approx", "noisy"]


class QGFPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        *,
        n_critics: int = 2,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        value_use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
        q_agg: Literal["mean", "min"] = "min",
        denoise_steps: int = 10,
        sampling_mode: SamplingMode = "guided",
        guidance_weight: float = 1.0,
        denoised_action_approx: DenoisedActionApprox = "one_euler_step_approx",
        qgrad_step_size: float = 0.1,
        qgrad_steps: int = 1,
        use_sign_gradient: bool = False,
        actor_num_samples: int = 32,
        robust_critic_lr: float = 3e-4,
        robust_critic_t_emb_size: int = 16,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "QGF requires a Box action space."
        if q_agg not in ("mean", "min"):
            raise ValueError(f"q_agg must be 'mean' or 'min', got {q_agg!r}.")
        if sampling_mode not in ("guided", "grad_step", "best_of_n", "bptt", "robust_q"):
            raise ValueError(f"Unknown sampling_mode: {sampling_mode!r}.")
        if denoised_action_approx not in ("one_euler_step_approx", "noisy"):
            raise ValueError(
                f"Unknown denoised_action_approx: {denoised_action_approx!r}."
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        net_arch = list(net_arch)

        self.actor = ActorVectorField(
            fd,
            action_dim,
            hidden_dims=net_arch,
            use_time_conditioning=True,
            use_layer_norm=actor_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )
        self.critic = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )
        self.critic_target = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.value = ValueNetwork(
            fd,
            net_arch,
            use_layer_norm=value_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )

        # RobustQ mode only (reconstructed -- see _robust_q_denoise's
        # docstring): a noise-conditioned critic Q_robust(s, a_t, t),
        # implemented by reusing EnsembleQCritic(n_critics=1) with its
        # "actions" argument set to concat([a_t, timestep_embedding(t)]) --
        # the same "treat a concatenated extra input as an expanded action"
        # trick this codebase already uses for chunked action spaces.
        self.robust_critic: Optional[EnsembleQCritic] = None
        self.robust_time_embed: Optional[_SinusoidalPosEmb] = None
        if sampling_mode == "robust_q":
            self.robust_time_embed = _SinusoidalPosEmb(robust_critic_t_emb_size)
            robust_action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(action_dim + robust_critic_t_emb_size,),
                dtype=np.float32,
            )
            self.robust_critic = EnsembleQCritic(
                fd,
                robust_action_space,
                hidden_dims=net_arch,
                n_critics=1,
                use_layer_norm=critic_use_layer_norm,
                kernel_init=kernel_init,
                backbone_type=backbone_type,
                activation_fn=activation_fn,
            )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        self.q_agg = q_agg
        self.denoise_steps = denoise_steps
        # Inference-time knobs -- deliberately plain settable attributes, not
        # constructor-only, so a loaded checkpoint's guidance can be swept
        # without retraining (see module docstring).
        self.sampling_mode: SamplingMode = sampling_mode
        self.guidance_weight = guidance_weight
        self.denoised_action_approx: DenoisedActionApprox = denoised_action_approx
        self.qgrad_step_size = qgrad_step_size
        self.qgrad_steps = qgrad_steps
        self.use_sign_gradient = use_sign_gradient
        self.actor_num_samples = actor_num_samples
        self.robust_critic_t_emb_size = robust_critic_t_emb_size

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def _aggregate_q(self, q_all: torch.Tensor) -> torch.Tensor:
        if self.q_agg == "min":
            return q_all.min(dim=0).values
        return q_all.mean(dim=0)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def critic_and_value_parameters(self):
        yield from self.critic.parameters()
        yield from self.value.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()

    def robust_critic_parameters(self):
        assert self.robust_critic is not None
        yield from self.robust_critic.parameters()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        # QGF always samples fresh N(0,1) flow-matching noise as the
        # generative latent (like FQL's own actor) -- `deterministic` is
        # accepted for interface compatibility but has no effect.
        del deterministic
        with torch.no_grad():
            features = self.extract_features(obs)
            batch_size = features.shape[0]
            device, dtype = features.device, features.dtype
            if self.sampling_mode == "guided":
                return self._guided_denoise(features, batch_size, device, dtype)
            if self.sampling_mode == "grad_step":
                return self._grad_step_denoise(features, batch_size, device, dtype)
            if self.sampling_mode == "bptt":
                return self._bptt_denoise(features, batch_size, device, dtype)
            if self.sampling_mode == "robust_q":
                return self._robust_q_denoise(features, batch_size, device, dtype)
            return self._best_of_n_denoise(features, batch_size, device, dtype)

    def _bc_denoise(self, features: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """Plain Euler-integrated BC denoise, no guidance. Shared by
        ``grad_step`` and ``best_of_n`` (``qgf.py``'s baselines both run this
        exact loop before their own post-processing)."""
        return self.actor.integrate(
            features, x_0, self.denoise_steps, low=self.action_low, high=self.action_high
        )

    def _q_grad(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``torch.autograd.grad`` (never ``.backward()``) of the aggregated
        target-critic Q w.r.t. a fresh leaf built from ``actions``. Does not
        write into any network parameter's ``.grad``."""
        with torch.enable_grad():
            a_leaf = actions.detach().requires_grad_(True)
            q = self._aggregate_q(self.q_values_all(features, a_leaf, target=True)).sum()
            (grad,) = torch.autograd.grad(q, a_leaf)
        return grad

    def _best_of_n_denoise(
        self, features: torch.Tensor, batch_size: int, device, dtype
    ) -> torch.Tensor:
        n = self.actor_num_samples
        action_dim = self.actor.action_dim
        rep_features = features.repeat_interleave(n, dim=0)
        x_0 = torch.randn(batch_size * n, action_dim, device=device, dtype=dtype)
        actions = self._bc_denoise(rep_features, x_0)
        q = self._aggregate_q(
            self.q_values_all(rep_features, actions, target=True)
        ).reshape(batch_size, n)
        actions = actions.reshape(batch_size, n, action_dim)
        best = q.argmax(dim=1)
        return actions[torch.arange(batch_size, device=device), best]

    def _grad_step_denoise(
        self, features: torch.Tensor, batch_size: int, device, dtype
    ) -> torch.Tensor:
        action_dim = self.actor.action_dim
        x_0 = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        actions = self._bc_denoise(features, x_0)
        for _ in range(self.qgrad_steps):
            grad = self._q_grad(features, actions)
            if self.use_sign_gradient:
                grad = grad.sign()
            actions = (actions + self.qgrad_step_size * grad).clamp(
                self.action_low, self.action_high
            )
        return actions

    def _guided_denoise(
        self, features: torch.Tensor, batch_size: int, device, dtype
    ) -> torch.Tensor:
        action_dim = self.actor.action_dim
        a = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        dt = 1.0 / self.denoise_steps
        for step in range(self.denoise_steps):
            t = torch.full(
                (batch_size, 1), step / self.denoise_steps, device=device, dtype=dtype
            )
            v_bc = self.actor(features, a, t)
            if self.denoised_action_approx == "one_euler_step_approx":
                a_approx = (a + (1 - t) * v_bc.detach()).clamp(
                    self.action_low, self.action_high
                )
            else:  # "noisy"
                a_approx = a
            qgrad = self._q_grad(features, a_approx)
            a = a + (v_bc + self.guidance_weight * qgrad) * dt
        return a.clamp(self.action_low, self.action_high)

    # ------------------------------------------------------------------
    # BPTT (reconstructed: agents/bptt.py:48 calls a self._bc_flow_from that
    # does not exist anywhere in 3rd_party/qgf -- reconstructed here from
    # the class's own docstring, "runs the full BC denoising process from
    # a_t to get a_clean = ODE(a_t)": Euler-integrate from `start_step` to
    # `denoise_steps` through the already-trained base flow. Structurally
    # identical to `_bc_denoise`, just starting partway through instead of
    # at t=0 from pure noise.)
    # ------------------------------------------------------------------

    def _bc_denoise_from(
        self, features: torch.Tensor, a_t: torch.Tensor, start_step: int
    ) -> torch.Tensor:
        x = a_t
        for step in range(start_step, self.denoise_steps):
            t = torch.full(
                (a_t.shape[0], 1),
                step / self.denoise_steps,
                device=a_t.device,
                dtype=a_t.dtype,
            )
            x = x + self.actor(features, x, t) / self.denoise_steps
        return x.clamp(self.action_low, self.action_high)

    def _bptt_grad(
        self, features: torch.Tensor, a_t: torch.Tensor, start_step: int
    ) -> torch.Tensor:
        """True backprop-through-time: differentiates through the *entire*
        remaining-steps rollout (``_bc_denoise_from``), unlike ``_q_grad``'s
        single-forward-pass gradient. Still never calls ``.backward()`` --
        only ``torch.autograd.grad``, so nothing leaks into any network
        parameter's ``.grad`` even though the forward pass inside
        ``enable_grad()`` is now a multi-step rollout instead of one call."""
        with torch.enable_grad():
            a_leaf = a_t.detach().requires_grad_(True)
            a_clean = self._bc_denoise_from(features, a_leaf, start_step)
            q = self._aggregate_q(self.q_values_all(features, a_clean, target=True)).sum()
            (grad,) = torch.autograd.grad(q, a_leaf)
        return grad

    def _bptt_denoise(
        self, features: torch.Tensor, batch_size: int, device, dtype
    ) -> torch.Tensor:
        action_dim = self.actor.action_dim
        a = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        dt = 1.0 / self.denoise_steps
        for step in range(self.denoise_steps):
            t = torch.full(
                (batch_size, 1), step / self.denoise_steps, device=device, dtype=dtype
            )
            v_bc = self.actor(features, a, t)
            qgrad = self._bptt_grad(features, a, step)
            a = a + (v_bc + self.guidance_weight * qgrad) * dt
        return a.clamp(self.action_low, self.action_high)

    # ------------------------------------------------------------------
    # RobustQ (agents/robust_q.py -- network construction is complete
    # upstream; the only bug is `sample_actions`'s undefined `cfg`, which
    # every sibling agent's `sample_actions` receives as `guidance_weight`.
    # Reconstructed by wiring `guidance_weight` into the signature, matching
    # every other mode in this file.)
    # ------------------------------------------------------------------

    def _robust_q_value(self, features: torch.Tensor, a_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert self.robust_critic is not None and self.robust_time_embed is not None
        t_emb = self.robust_time_embed(t.squeeze(-1))
        return self.robust_critic.forward_all(features, torch.cat([a_t, t_emb], dim=-1))

    def _robust_q_grad(
        self, features: torch.Tensor, a_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Guidance from the noise-conditioned critic, evaluated directly at
        the noisy point -- no Jacobian/one-euler-step-approx machinery
        needed (unlike ``"guided"`` mode), since ``robust_critic`` is
        trained specifically to be queried at noisy actions."""
        with torch.enable_grad():
            a_leaf = a_t.detach().requires_grad_(True)
            q = self._aggregate_q(self._robust_q_value(features, a_leaf, t)).sum()
            (grad,) = torch.autograd.grad(q, a_leaf)
        return grad

    def _robust_q_denoise(
        self, features: torch.Tensor, batch_size: int, device, dtype
    ) -> torch.Tensor:
        action_dim = self.actor.action_dim
        a = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        dt = 1.0 / self.denoise_steps
        for step in range(self.denoise_steps):
            t = torch.full(
                (batch_size, 1), step / self.denoise_steps, device=device, dtype=dtype
            )
            v_bc = self.actor(features, a, t)
            qgrad = self._robust_q_grad(features, a, t)
            a = a + (v_bc + self.guidance_weight * qgrad) * dt
        return a.clamp(self.action_low, self.action_high)
