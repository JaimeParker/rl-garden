"""State-based SAC (base algorithm).

Template: ManiSkill's ``examples/baselines/sac/sac.py``, restructured as
an ``OffPolicyAlgorithm`` subclass with a ``SACPolicy`` that owns a
``FlattenExtractor`` (state obs go through an identity flatten and the
actor/critic MLPs handle the rest).

RGBD SAC subclasses this and only overrides:
  - ``_build_features_extractor`` (to use a CombinedExtractor with images)
  - ``train`` (to reuse visual features + detach encoder on actor path)
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.sac_policy import SACPolicy


class SAC(OffPolicyAlgorithm):
    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 1024,
        gamma: float = 0.8,
        tau: float = 0.01,
        training_freq: int = 64,
        utd: float = 0.5,
        bootstrap_at_done: str = "always",
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        n_critics: int = 2,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            seed=seed,
            device=device,
            logger=logger,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
        )
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.ent_coef_init = ent_coef
        self.target_entropy_arg = target_entropy
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.n_critics = n_critics

        self._setup_model()

    # --- construction hooks (overridden by RGBDSAC) ---

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        assert isinstance(self.env.single_observation_space, spaces.Box), (
            "SAC base class expects a flat Box observation space; "
            "use RGBDSAC for dict observations."
        )
        return FlattenExtractor(self.env.single_observation_space)

    def _build_replay_buffer(self):
        return TensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = SACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            n_critics=self.n_critics,
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            list(self.policy.critic_and_encoder_parameters()), lr=self.q_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor_parameters()), lr=self.policy_lr
        )

        # Entropy coefficient (auto-tuned by default).
        self.autotune = isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith(
            "auto"
        )
        if self.autotune:
            init = 1.0
            if isinstance(self.ent_coef_init, str) and "_" in self.ent_coef_init:
                init = float(self.ent_coef_init.split("_")[1])
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * init
            ).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = torch.tensor(float(self.ent_coef_init), device=self.device)

        if self.target_entropy_arg == "auto":
            self.target_entropy = float(
                -np.prod(self.env.single_action_space.shape).astype(np.float32)
            )
        else:
            self.target_entropy = float(self.target_entropy_arg)

        self.replay_buffer = self._build_replay_buffer()

    # --- hot-path helpers overridden by RGBDSAC ---

    def _critic_forward(self, obs, actions, target: bool = False):
        features = self.policy.extract_features(obs, detach=False)
        return self.policy.q_values(features, actions, target=target)

    def _target_q(self, replay_data) -> torch.Tensor:
        alpha = self._current_alpha().detach()
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.actor_action_log_prob(
                replay_data.next_obs, detach_encoder=False
            )
            q_values_t = self.policy.q_values(
                self.policy.extract_features(replay_data.next_obs),
                next_action,
                target=True,
            )
            min_q_next = torch.min(torch.stack(q_values_t, dim=0), dim=0).values
            min_q_next = min_q_next - alpha * next_log_prob
            target = replay_data.rewards.reshape(-1, 1) + (
                1 - replay_data.dones.reshape(-1, 1)
            ) * self.gamma * min_q_next
        return target

    def _actor_loss(self, obs):
        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, detach_encoder=False
        )
        q_values = self.policy.q_values(features, action, target=False)
        min_q = torch.min(torch.stack(q_values, dim=0), dim=0).values
        return (alpha * log_prob - min_q).mean(), log_prob.detach()

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.log_alpha.exp()
        return self._fixed_alpha

    # --- training step ---

    def train(self, gradient_steps: int) -> dict[str, float]:
        q_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        alphas: list[float] = []

        for step in range(gradient_steps):
            self._global_update += 1
            data = self.replay_buffer.sample(self.batch_size)

            # --- critic update ---
            target_q = self._target_q(data)
            current_q_values = self._critic_forward(data.obs, data.actions, target=False)
            q_loss = sum(F.mse_loss(q, target_q) for q in current_q_values)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            q_losses.append(q_loss.item())

            # --- actor + alpha updates ---
            if self._global_update % self.policy_frequency == 0:
                actor_loss, log_prob_detached = self._actor_loss(data.obs)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                if self.autotune:
                    alpha_loss = -(
                        self.log_alpha.exp() * (log_prob_detached + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_losses.append(alpha_loss.item())

            alphas.append(self._current_alpha().item())

            # --- target critic update ---
            if self._global_update % self.target_network_frequency == 0:
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )

        out = {
            "qf_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": float(np.mean(alphas)) if alphas else 0.0,
        }
        if alpha_losses:
            out["alpha_loss"] = float(np.mean(alpha_losses))
        return out
