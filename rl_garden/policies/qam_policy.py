"""QAM policy: two flow actors (base + adjoint-matching-trained correction),
a critic (DDPG- or IQL-style bootstrap), and two mutually-exclusive optional
bolt-ons -- ``fql_alpha`` (one-step distillation) or ``edit_scale`` (a small
residual-editing policy).

Ports ``3rd_party/qgf/agents/qam.py``. Network ownership mirrors
``FQLPolicy``'s "run the network forward several times to build a training
target" convention (``compute_flow_actions``): all such heavy-forward-pass
methods live here, not on ``QAMCore``, exactly like
``FQLPolicy.compute_flow_actions``.

``adj_matching`` is the genuinely novel piece with no prior rl-garden
precedent -- it needs to (1) sample a *stochastic* Euler-Maruyama trajectory
through the flow (unlike every other integration loop in this repo, which is
a plain deterministic ODE), then (2) propagate a "adjoint state" backward
through that trajectory via a vector-Jacobian product at each step. Both
pieces run entirely as target-construction for ``QAMCore.actor_loss`` (never
differentiated a second time), so the whole method wraps in
``torch.no_grad()`` -- but ``torch.func.grad``/``torch.func.vjp`` (unlike
``torch.autograd.grad``) work correctly even under an ambient ``no_grad()``
with no extra bookkeeping (confirmed by a standalone sanity check this
session, on this environment's PyTorch 2.13.0+cu130): no
``requires_grad_()``/``enable_grad()`` gymnastics needed, and neither
primitive ever writes into a network parameter's ``.grad`` (they return
gradients as values, matching JAX's ``jax.grad``/``jax.vjp`` semantics
exactly, which is what the reference itself uses).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.alpha_tuning import AlphaTuner
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    ActorVectorField,
    Activation,
    BackboneType,
    EnsembleQCritic,
    KernelInit,
    SquashedGaussianActor,
    ValueNetwork,
)
from rl_garden.policies.base import BasePolicy

CriticLossType = Literal["ddpg", "iql"]


class QAMPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        *,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        value_use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
        critic_loss_type: CriticLossType = "ddpg",
        rho: float = 0.0,
        expectile: float = 0.9,
        flow_steps: int = 10,
        best_of_n: int = 1,
        inv_temp: float = 0.3,
        residual: bool = False,
        target_actor: bool = True,
        clip_adj: bool = True,
        use_target_grad: bool = True,
        fql_alpha: float = 0.0,
        edit_scale: float = 0.0,
        edit_target_entropy: Optional[float] = None,
        edit_target_entropy_multiplier: float = 0.5,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "QAM requires a Box action space."
        if critic_loss_type not in ("ddpg", "iql"):
            raise ValueError(
                f"critic_loss_type must be 'ddpg' or 'iql', got {critic_loss_type!r}."
            )
        if fql_alpha > 0.0 and edit_scale > 0.0:
            raise ValueError(
                "Only one of fql_alpha and edit_scale can be non-zero "
                f"(got fql_alpha={fql_alpha}, edit_scale={edit_scale})."
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        fd = features_extractor.features_dim
        self.full_action_dim = int(np.prod(action_space.shape))
        net_arch = list(net_arch)

        def _flow(use_time: bool) -> ActorVectorField:
            return ActorVectorField(
                fd,
                self.full_action_dim,
                hidden_dims=net_arch,
                use_time_conditioning=use_time,
                use_layer_norm=actor_use_layer_norm,
                kernel_init=kernel_init,
                activation_fn=activation_fn,
            )

        self.actor_slow = _flow(True)
        self.actor_fast = _flow(True)
        self.target_actor_slow = _flow(True)
        self.target_actor_slow.load_state_dict(self.actor_slow.state_dict())
        for p in self.target_actor_slow.parameters():
            p.requires_grad_(False)

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
        self.target_critic = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.critic_loss_type: CriticLossType = critic_loss_type
        self.value: Optional[ValueNetwork] = None
        if critic_loss_type == "iql":
            self.value = ValueNetwork(
                fd,
                net_arch,
                use_layer_norm=value_use_layer_norm,
                kernel_init=kernel_init,
                backbone_type=backbone_type,
            )

        self.one_step_actor: Optional[ActorVectorField] = None
        if fql_alpha > 0.0:
            self.one_step_actor = _flow(False)

        self.edit_actor: Optional[SquashedGaussianActor] = None
        self.edit_alpha: Optional[AlphaTuner] = None
        if edit_scale > 0.0:
            edit_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.full_action_dim,), dtype=np.float32
            )
            self.edit_actor = SquashedGaussianActor(
                fd + self.full_action_dim,
                edit_space,
                hidden_dims=net_arch,
                use_layer_norm=actor_use_layer_norm,
                kernel_init=kernel_init,
                backbone_type=backbone_type,
            )
            # Reuses AlphaTuner's "lagrange_softplus" mode as-is (zero changes
            # to the shared module): its loss shape (`alpha * (entropy -
            # target_entropy)`) already matches qam.py:308-310 exactly. Only
            # the scalar's own parameterization differs from the reference's
            # exp()-based LogParam (softplus here vs exp there) -- both are
            # smooth positive reparameterizations of the same free scalar,
            # functionally inert for what the loss actually optimizes.
            self.edit_alpha = AlphaTuner("lagrange_softplus", init_value=1.0)
            self.edit_target_entropy = (
                edit_target_entropy
                if edit_target_entropy is not None
                else -edit_target_entropy_multiplier * self.full_action_dim
            )

        self.rho = rho
        self.expectile = expectile
        self.flow_steps = flow_steps
        self.best_of_n = best_of_n
        self.inv_temp = inv_temp
        self.residual = residual
        self.target_actor = target_actor
        self.clip_adj = clip_adj
        self.use_target_grad = use_target_grad
        self.fql_alpha = fql_alpha
        self.edit_scale = edit_scale

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.target_critic if target else self.critic
        return net.forward_all(features, actions)

    def critic_and_value_parameters(self):
        yield from self.critic.parameters()
        if self.value is not None:
            yield from self.value.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        yield from self.actor_slow.parameters()
        yield from self.actor_fast.parameters()
        if self.one_step_actor is not None:
            yield from self.one_step_actor.parameters()
        if self.edit_actor is not None:
            yield from self.edit_actor.parameters()

    def _effective_actor_slow(self) -> ActorVectorField:
        return self.target_actor_slow if self.target_actor else self.actor_slow

    # ------------------------------------------------------------------
    # Flow integration (deterministic ODE, shared by inference and the
    # fql_alpha distillation target)
    # ------------------------------------------------------------------

    def compute_flow_actions(
        self, features: torch.Tensor, noises: torch.Tensor, num_steps: int, *, model: str = "slow"
    ) -> torch.Tensor:
        """Deterministic multi-step Euler integration, summing the velocity
        of every network named in ``model`` (comma-separated, from
        ``{"slow", "fast"}``). Generalizes ``ActorVectorField.integrate()``
        (which only supports one network) to QAM's residual
        (``actor_slow + actor_fast``) case; matches ``qam.py:450-473``."""
        networks = [getattr(self, f"actor_{m}") for m in model.split(",")]
        x = noises
        for step in range(num_steps):
            t = torch.full(
                (features.shape[0], 1),
                step / num_steps,
                device=features.device,
                dtype=features.dtype,
            )
            velocity = sum(net(features, x, t) for net in networks)
            x = x + velocity / num_steps
        return x.clamp(self.action_low, self.action_high)

    # ------------------------------------------------------------------
    # Adjoint matching (training-target construction only)
    # ------------------------------------------------------------------

    def _aggregate_mean(self, q_all: torch.Tensor) -> torch.Tensor:
        return q_all.mean(dim=0)

    def adj_matching(
        self, features: torch.Tensor, flow_steps: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Returns ``(xs, adjs, ts, pre_adj_info)``, each of shape
        ``(flow_steps, batch, ...)`` except ``pre_adj_info`` (a metrics
        dict) -- matches ``qam.py:111-186`` exactly. Entirely
        target-construction for ``QAMCore.actor_loss``; never differentiated
        a second time."""
        flow_steps = self.flow_steps if flow_steps is None else flow_steps
        batch_size = features.shape[0]
        device, dtype = features.device, features.dtype
        h = 1.0 / flow_steps
        actor_slow = self._effective_actor_slow()

        with torch.no_grad():
            x = torch.randn(batch_size, self.full_action_dim, device=device, dtype=dtype)
            xs = [x]
            ts = []
            for i in range(flow_steps):
                t = torch.full((batch_size, 1), i / flow_steps, device=device, dtype=dtype)
                sigma = torch.sqrt(2 * (1 - t + h) / (t + h))
                noise = torch.randn_like(x)
                if i != flow_steps - 1:
                    if self.residual:
                        v = self.actor_fast(features, x, t) + actor_slow(features, x, t)
                    else:
                        v = self.actor_fast(features, x, t)
                    x = x + h * (2 * v - x / (t + h)) + (h**0.5) * sigma * noise
                else:  # last step: plain ODE integration through the base flow
                    x = x + h * actor_slow(features, x, t)
                xs.append(x)
                ts.append(t)

            critic_net = self.target_critic if self.use_target_grad else self.critic

            def q_fn(y: torch.Tensor) -> torch.Tensor:
                y_in = y.clamp(-1.0, 1.0) if self.clip_adj else y
                return self._aggregate_mean(critic_net.forward_all(features, y_in)).sum()

            grad = torch.func.grad(q_fn)(xs[-1])
            adj = -grad * self.inv_temp
            pre_adj_info = {
                "adj_max": float(adj.abs().max().item()),
                "adj_std": float(adj.abs().std().item()),
                "adj_mean": float(adj.abs().mean().item()),
            }

            adjs = []
            adj_current = adj
            for i in reversed(range(flow_steps)):
                t = torch.full((batch_size, 1), i / flow_steps, device=device, dtype=dtype)

                def fn(xi: torch.Tensor, t=t) -> torch.Tensor:
                    return 2 * actor_slow(features, xi, t + h) - xi / (t + h)

                _, vjp_fn = torch.func.vjp(fn, xs[i])
                (vjp_result,) = vjp_fn(adj_current)
                adj_current = adj_current + h * vjp_result
                adjs.append(adj_current)

        xs_used = torch.stack(xs[:-1], dim=0)
        adjs_used = torch.stack(list(reversed(adjs)), dim=0)
        ts_used = torch.stack(ts, dim=0)
        return xs_used, adjs_used, ts_used, pre_adj_info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic
        with torch.no_grad():
            features = self.extract_features(obs)
            batch_size = features.shape[0]
            device, dtype = features.device, features.dtype
            n = self.best_of_n
            rep_features = features.repeat_interleave(n, dim=0)

            if self.fql_alpha > 0.0:
                assert self.one_step_actor is not None
                noises = torch.randn(
                    batch_size * n, self.full_action_dim, device=device, dtype=dtype
                )
                actions = self.one_step_actor(rep_features, noises).clamp(
                    self.action_low, self.action_high
                )
            else:
                noises = torch.randn(
                    batch_size * n, self.full_action_dim, device=device, dtype=dtype
                )
                if self.inv_temp == 0.0:
                    actions = self.compute_flow_actions(
                        rep_features, noises, self.flow_steps, model="slow"
                    )
                else:
                    model = "slow,fast" if self.residual else "fast"
                    actions = self.compute_flow_actions(
                        rep_features, noises, self.flow_steps, model=model
                    )
                if self.edit_scale > 0.0:
                    assert self.edit_actor is not None
                    edit_features = torch.cat([rep_features, actions], dim=-1)
                    edit, _ = self.edit_actor.action_log_prob(edit_features)
                    actions = (actions + edit * self.edit_scale).clamp(
                        self.action_low, self.action_high
                    )

            if n > 1:
                q = self._aggregate_mean(self.q_values_all(rep_features, actions, target=False))
                q = q.reshape(batch_size, n)
                best = q.argmax(dim=1)
                actions = actions.reshape(batch_size, n, self.full_action_dim)
                actions = actions[torch.arange(batch_size, device=device), best]
            return actions
