"""FlowPPO policy: a single trainable flow-matching actor (``ActorVectorField``)
fine-tuned online via a stochastic (SDE) integration of its ODE, plus an
obs-only critic. See ``rl_garden/algorithms/flow_ppo.py``'s module docstring
for the algorithm-level design notes (why this is not a ``DPPO`` subclass,
the sigma/tau sign-convention re-derivation, the final-step density fix).

Unlike ``DPPOPolicy``, there is no frozen/trainable actor pair: RL-100's
flow fine-tuning trains the whole network directly (every SDE step must stay
stochastic for the log-prob to be well-defined for PPO ratios -- no partial
frozen-prefix trick like DPPO's ``ft_denoising_steps``). Action chunking
follows FQL/ACFQL's convention, not DPPO's: a chunk of ``horizon_length``
future actions is denoised jointly as one flat ``action_dim =
horizon_length * base_action_dim`` vector (no separate per-timestep
``horizon_steps`` axis inside the network).

Box or Dict (CNN-based vision, via ``CombinedExtractor``) observations.
``_features(obs)`` is the single seam where obs becomes conditioning --
mirrors ``DPPOPolicy._cond``'s own docstring note: rollout collection
(``FlowPPO._rollout_step``) and training (``FlowPPO._flow_ppo_loss``) both
call it once per env-step/minibatch and reuse the resulting tensor across
the whole K-step SDE trajectory. Gradient isolation (critic trains the
encoder, actor path detaches) is implemented at the
``FlowPPO._flow_ppo_loss`` call site, not inside this method.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_vector_field import ActorVectorField, flow_logprob, flow_sde_step
from rl_garden.networks.value import ValueNetwork
from rl_garden.policies.base import BasePolicy


class FlowPPOPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        *,
        features_extractor: BaseFeaturesExtractor,
        flow_steps: int,
        horizon_length: int = 1,
        actor_mlp_dims: Sequence[int] = (256, 256, 256),
        actor_activation_fn: Optional[Activation] = None,
        critic_mlp_dims: Sequence[int] = (256, 256, 256),
        critic_activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
        sde_type: Literal["sde", "cps"] = "cps",
        noise_level: float = 0.7,
        clip_std_min: float = 0.0067,
        sigma_safe_max: float = 0.9,
        logprob_mode: Literal["gaussian", "pseudo"] = "gaussian",
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "FlowPPOPolicy requires a Box action space."
        assert isinstance(
            observation_space, (spaces.Box, spaces.Dict)
        ), "FlowPPOPolicy supports Box or Dict observation spaces only."
        if flow_steps <= 0:
            raise ValueError(f"flow_steps must be positive, got {flow_steps}.")
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}.")

        self.observation_space = observation_space
        self.features_extractor = features_extractor
        # `action_space` is the RAW per-step Box (shape (base_action_dim,)),
        # matching DPPOPolicy's convention -- the caller strips any
        # ActionChunkWrapper horizon axis before constructing this policy.
        self.action_space = action_space
        self.flow_steps = flow_steps
        self.horizon_length = horizon_length
        self.sde_type = sde_type
        self.noise_level = noise_level
        self.clip_std_min = clip_std_min
        self.sigma_safe_max = sigma_safe_max
        self.logprob_mode = logprob_mode

        self.base_action_dim = int(np.prod(action_space.shape))
        self.action_dim = self.base_action_dim * horizon_length
        fd = features_extractor.features_dim

        self.actor = ActorVectorField(
            features_dim=fd,
            action_dim=self.action_dim,
            hidden_dims=actor_mlp_dims,
            use_time_conditioning=True,
            kernel_init=kernel_init,
            activation_fn=actor_activation_fn,
        )
        self.critic = ValueNetwork(
            fd,
            critic_mlp_dims,
            kernel_init=kernel_init,
            activation_fn=critic_activation_fn,
        )

        # 1D (base_action_dim,); broadcasts against (N, horizon_length,
        # base_action_dim) in clamp_action, same pattern as DPPOPolicy.
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def _features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def to_chunk_shape(self, flat_action: torch.Tensor) -> torch.Tensor:
        """``(N, action_dim) -> (N, horizon_length, base_action_dim)``, the
        shape ``ActionChunkWrapper``-wrapped envs expect."""
        return flat_action.reshape(-1, self.horizon_length, self.base_action_dim)

    def _tau(self, step: int, batch: int, device, dtype) -> torch.Tensor:
        return torch.full((batch, 1), step / self.flow_steps, device=device, dtype=dtype)

    def _sde_step(self, velocity: torch.Tensor, x: torch.Tensor, tau: torch.Tensor):
        return flow_sde_step(
            velocity,
            x,
            tau,
            1.0 / self.flow_steps,
            sde_type=self.sde_type,
            noise_level=self.noise_level,
            clip_std_min=self.clip_std_min,
            sigma_safe_max=self.sigma_safe_max,
        )

    def sample_rollout_chain(
        self, obs: Obs, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """No-grad sampling. ``deterministic=False`` (rollout collection):
        full stochastic SDE trajectory. ``deterministic=True`` (evaluation):
        the noise-free mean trajectory, matching RL-100's ``step_mean`` --
        for ``"cps"`` this reuses the same mean formula (noise is purely
        additive on top of it); for ``"sde"`` this reverts to plain ODE
        Euler (``x + velocity*dtau``, i.e. ``ActorVectorField.integrate``'s
        own step), since ``"sde"``'s mean formula includes drift-correction
        terms that exist only to compensate for the noise term it otherwise
        adds.

        Returns ``(chain, log_probs, action)``: ``chain`` is
        ``(N, flow_steps+1, action_dim)`` (index 0 = noise, index
        ``flow_steps`` = the final action), ``log_probs`` is
        ``(N, flow_steps, action_dim)`` (elementwise, one entry per SDE
        step; all-zero when ``deterministic=True``, since there is no
        well-defined density for a noise-free trajectory), ``action`` is
        ``(N, action_dim)`` (== ``chain[:, -1]``).
        """
        features = self._features(obs)
        batch = features.shape[0]
        x = torch.randn(batch, self.action_dim, device=features.device, dtype=features.dtype)
        chain = [x]
        log_probs = []
        for step in range(self.flow_steps):
            tau = self._tau(step, batch, features.device, features.dtype)
            velocity = self.actor(features, x, tau)
            if deterministic:
                if self.sde_type == "cps":
                    mean, _ = self._sde_step(velocity, x, tau)
                else:
                    mean = x + velocity / self.flow_steps
                x = mean
                log_probs.append(torch.zeros_like(x))
            else:
                mean, std = self._sde_step(velocity, x, tau)
                x = mean + std * torch.randn_like(mean)
                log_probs.append(flow_logprob(x, mean, std, mode=self.logprob_mode))
            chain.append(x)
        return torch.stack(chain, dim=1), torch.stack(log_probs, dim=1), x

    def get_logprobs_subsample(
        self,
        features: torch.Tensor,
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        flow_step_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Grad-enabled log-probs for a subsample of ``(env-step,
        flow-step)`` pairs -- the PPO update's recompute."""
        tau = (flow_step_inds.float() / self.flow_steps).unsqueeze(-1)
        velocity = self.actor(features, chains_prev, tau)
        mean, std = self._sde_step(velocity, chains_prev, tau)
        return flow_logprob(chains_next, mean, std, mode=self.logprob_mode)

    def predict_values(self, obs: Obs) -> torch.Tensor:
        return self.critic(self._features(obs)).view(-1)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        _, _, action = self.sample_rollout_chain(obs, deterministic=deterministic)
        return self.to_chunk_shape(action)

    def clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        """``actions``: ``(N, horizon_length, base_action_dim)``."""
        return actions.clamp(self.action_low, self.action_high)
