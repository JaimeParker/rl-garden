"""WSRL (Warm-Start RL) algorithm with CQL and Cal-QL support.

Implements offline→online training:
- Offline phase: Cal-QL (Conservative Q-Learning with calibration)
- Online phase: SAC or CQL (configurable)

Key features:
- Q-ensemble (REDQ) with 10 critics by default
- CQL regularization to prevent Q-value overestimation
- Cal-QL lower bounds using Monte Carlo returns
- Seamless offline→online mode switching
- High-UTD training support

Based on:
- WSRL paper: https://arxiv.org/abs/2412.07762
- Cal-QL paper: https://arxiv.org/abs/2303.05479
- CQL paper: https://arxiv.org/abs/2006.04779
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.buffers.mc_buffer import MCReplayBufferSample, MCTensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.wsrl_policy import WSRLPolicy


class WSRL(OffPolicyAlgorithm):
    """WSRL algorithm with CQL/Cal-QL for offline→online training.

    Supports:
    - Offline pre-training with Cal-QL
    - Online fine-tuning with SAC or CQL
    - Q-ensemble (REDQ) with configurable size
    - High-UTD training
    """

    _SUPPORTED_POLICY_KWARGS = frozenset({"features_extractor_class", "features_extractor_kwargs"})

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        # Buffer and training
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "always",
        # Optimizers
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        cql_alpha_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        # Entropy
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        # Network architecture
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        # Q-ensemble (REDQ)
        n_critics: int = 10,
        critic_subsample_size: int = 2,
        # CQL parameters
        use_cql_loss: bool = True,
        cql_n_actions: int = 10,
        cql_alpha: float = 5.0,
        cql_autotune_alpha: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        cql_target_action_gap: float = 1.0,
        cql_importance_sample: bool = True,
        cql_max_target_backup: bool = True,
        cql_temp: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        # Cal-QL parameters
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        # Phase control
        use_td_loss: bool = True,
        online_cql_alpha: Optional[float] = None,
        online_use_cql_loss: Optional[bool] = None,
        # General
        policy_kwargs: Optional[dict[str, Any]] = None,
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

        # Optimizers
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.cql_alpha_lr = cql_alpha_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

        # Entropy
        self.ent_coef_init = ent_coef
        self.target_entropy_arg = target_entropy

        # Network architecture
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        # CQL parameters
        self.use_cql_loss = use_cql_loss
        self.cql_n_actions = cql_n_actions
        self.cql_alpha = cql_alpha
        self.cql_autotune_alpha = cql_autotune_alpha
        self.cql_alpha_lagrange_init = cql_alpha_lagrange_init
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_temp = cql_temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max

        # Cal-QL parameters
        self.use_calql = use_calql
        self.calql_bound_random_actions = calql_bound_random_actions

        # Phase control
        self.use_td_loss = use_td_loss
        self.online_cql_alpha = online_cql_alpha
        self.online_use_cql_loss = online_use_cql_loss

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()

    # --- Construction hooks (WSRLRGBD overrides defaults only) ---

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        assert isinstance(self.env.single_observation_space, spaces.Box), (
            "WSRL base class expects a flat Box observation space; "
            "use WSRLRGBD for dict observations."
        )
        return FlattenExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        return {}

    def _normalize_policy_kwargs(
        self, policy_kwargs: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized = dict(policy_kwargs or {})
        unknown_keys = sorted(set(normalized) - self._SUPPORTED_POLICY_KWARGS)
        if unknown_keys:
            raise ValueError(
                "Unsupported policy_kwargs keys: "
                + ", ".join(unknown_keys)
                + ". Supported keys are: features_extractor_class, "
                + "features_extractor_kwargs."
            )

        features_extractor_kwargs = normalized.get("features_extractor_kwargs", {})
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        if not isinstance(features_extractor_kwargs, dict):
            raise TypeError("policy_kwargs['features_extractor_kwargs'] must be a dict.")

        normalized["features_extractor_kwargs"] = dict(features_extractor_kwargs)
        return normalized

    def _resolve_policy_kwargs(self) -> dict[str, Any]:
        default_features_extractor_class = self._default_features_extractor_class()
        default_features_extractor_kwargs = dict(self._default_features_extractor_kwargs())
        resolved = {
            "features_extractor_class": default_features_extractor_class,
            "features_extractor_kwargs": default_features_extractor_kwargs,
        }

        if "features_extractor_class" in self.policy_kwargs:
            resolved["features_extractor_class"] = self.policy_kwargs["features_extractor_class"]
            if resolved["features_extractor_class"] is not default_features_extractor_class:
                resolved["features_extractor_kwargs"] = {}
        if "features_extractor_kwargs" in self.policy_kwargs:
            if resolved["features_extractor_class"] is default_features_extractor_class:
                resolved["features_extractor_kwargs"] = {
                    **resolved["features_extractor_kwargs"],
                    **self.policy_kwargs["features_extractor_kwargs"],
                }
            else:
                resolved["features_extractor_kwargs"] = dict(
                    self.policy_kwargs["features_extractor_kwargs"]
                )
        return resolved

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        resolved = self._resolve_policy_kwargs()
        features_extractor_class = resolved["features_extractor_class"]
        if not isinstance(features_extractor_class, type) or not issubclass(
            features_extractor_class, BaseFeaturesExtractor
        ):
            raise TypeError(
                "policy_kwargs['features_extractor_class'] must be a "
                "BaseFeaturesExtractor subclass."
            )
        return features_extractor_class(
            observation_space=self.env.single_observation_space,
            **resolved["features_extractor_kwargs"],
        )

    def _build_replay_buffer(self):
        return MCTensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = WSRLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            use_cql_alpha_lagrange=self.cql_autotune_alpha,
            cql_alpha_lagrange_init=self.cql_alpha_lagrange_init,
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            list(self.policy.critic_and_encoder_parameters()), lr=self.q_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor_parameters()), lr=self.policy_lr
        )

        # CQL alpha Lagrange multiplier optimizer
        if self.cql_autotune_alpha:
            self.cql_alpha_optimizer = torch.optim.Adam(
                list(self.policy.cql_alpha_lagrange_parameters()), lr=self.cql_alpha_lr
            )
        else:
            self.cql_alpha_optimizer = None

        # Entropy coefficient (auto-tuned by default)
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
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
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

    # --- Helper methods ---

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.log_alpha.exp()
        return self._fixed_alpha

    def _current_cql_alpha(self) -> torch.Tensor:
        if self.cql_autotune_alpha:
            return self.policy.get_cql_alpha()
        return torch.tensor(self.cql_alpha, device=self.device)

    def switch_to_online_mode(self):
        """Switch from offline to online training mode."""
        if self.online_use_cql_loss is not None:
            self.use_cql_loss = self.online_use_cql_loss
        if self.online_cql_alpha is not None:
            self.cql_alpha = self.online_cql_alpha

        if self.logger:
            self.logger.record("wsrl/switched_to_online", 1)
            self.logger.record("wsrl/use_cql_loss", int(self.use_cql_loss))
            self.logger.record("wsrl/cql_alpha", self.cql_alpha)

    # --- CQL Loss Computation ---

    def _sample_ood_actions(
        self, obs: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample out-of-distribution actions for CQL.

        Returns:
            random_actions: (batch, cql_n_actions, action_dim)
            current_actions: (batch, cql_n_actions, action_dim)
            next_actions: (batch, cql_n_actions, action_dim)
        """
        action_dim = self.env.single_action_space.shape[0]

        # Sample random actions uniformly in [-1, 1]
        random_actions = torch.rand(
            batch_size, self.cql_n_actions, action_dim, device=self.device
        ) * 2 - 1

        # Sample actions from current policy
        current_actions_list = []
        for _ in range(self.cql_n_actions):
            action, _ = self.policy.actor.action_log_prob(
                self.policy.extract_features(obs)
            )
            current_actions_list.append(action)
        current_actions = torch.stack(current_actions_list, dim=1)  # (batch, n_actions, dim)

        # Sample actions from next-state policy (reuse obs for simplicity in offline)
        next_actions_list = []
        for _ in range(self.cql_n_actions):
            action, _ = self.policy.actor.action_log_prob(
                self.policy.extract_features(obs)
            )
            next_actions_list.append(action)
        next_actions = torch.stack(next_actions_list, dim=1)

        return random_actions, current_actions, next_actions

    def _cql_loss(
        self,
        data: MCReplayBufferSample,
        q_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute CQL regularization loss with optional Cal-QL bounds.

        Args:
            data: Replay buffer sample with MC returns
            q_pred: Predicted Q-values for dataset actions (n_critics, batch, 1)

        Returns:
            cql_loss: CQL regularization loss
            info: Dictionary with logging information
        """
        batch_size = data.rewards.shape[0]
        features = self.policy.extract_features(data.obs)

        # Sample OOD actions
        random_actions, current_actions, next_actions = self._sample_ood_actions(
            data.obs, batch_size
        )

        # Concatenate all sampled actions: (batch, 3*cql_n_actions, action_dim)
        all_actions = torch.cat([random_actions, current_actions, next_actions], dim=1)

        # Evaluate Q-values for all sampled actions
        # Reshape for batch evaluation: (batch * 3*cql_n_actions, action_dim)
        all_actions_flat = all_actions.reshape(-1, all_actions.shape[-1])
        features_repeated = features.unsqueeze(1).repeat(
            1, 3 * self.cql_n_actions, 1
        ).reshape(-1, features.shape[-1])

        q_ood = self.policy.q_values_all(features_repeated, all_actions_flat, target=False)
        # Reshape back: (n_critics, batch, 3*cql_n_actions)
        q_ood = q_ood.reshape(self.n_critics, batch_size, 3 * self.cql_n_actions)

        # Apply Cal-QL lower bounds if enabled
        info = {}
        if self.use_calql and hasattr(data, 'mc_returns') and data.mc_returns is not None:
            mc_returns = data.mc_returns.reshape(batch_size, 1)  # (batch, 1)

            if self.calql_bound_random_actions:
                # Bound all actions
                mc_lower_bound = mc_returns.unsqueeze(0).repeat(
                    self.n_critics, 1, 3 * self.cql_n_actions
                )
            else:
                # Only bound current and next policy actions, not random
                fake_bound = torch.full(
                    (self.n_critics, batch_size, self.cql_n_actions),
                    -float('inf'),
                    device=self.device
                )
                real_bound = mc_returns.unsqueeze(0).repeat(
                    self.n_critics, 1, 2 * self.cql_n_actions
                )
                mc_lower_bound = torch.cat([fake_bound, real_bound], dim=2)

            # Track bound violation rate
            num_vals = q_ood.numel()
            calql_bound_rate = (q_ood < mc_lower_bound).sum().item() / num_vals
            info["calql_bound_rate"] = calql_bound_rate

            # Apply bounds
            q_ood = torch.maximum(q_ood, mc_lower_bound)

        # Importance sampling (optional)
        if self.cql_importance_sample:
            # Compute log probabilities for sampled actions
            action_dim = self.env.single_action_space.shape[0]
            random_log_prob = np.log(0.5 ** action_dim)  # Uniform in [-1, 1]

            # Current policy log probs
            current_log_probs = []
            for i in range(self.cql_n_actions):
                _, log_prob = self.policy.actor.action_log_prob(features)
                current_log_probs.append(log_prob)
            current_log_probs = torch.cat(current_log_probs, dim=1)  # (batch, n_actions)

            # Next policy log probs (reuse current for simplicity)
            next_log_probs = current_log_probs

            # Concatenate: (batch, 3*n_actions)
            all_log_probs = torch.cat([
                torch.full((batch_size, self.cql_n_actions), random_log_prob, device=self.device),
                current_log_probs,
                next_log_probs
            ], dim=1)

            # Apply importance weighting: (n_critics, batch, 3*n_actions)
            all_log_probs_expanded = all_log_probs.unsqueeze(0).repeat(self.n_critics, 1, 1)
            q_ood = q_ood - all_log_probs_expanded

        # Logsumexp over actions
        q_ood_logsumexp = torch.logsumexp(q_ood / self.cql_temp, dim=2) * self.cql_temp
        # (n_critics, batch)

        # CQL penalty: logsumexp(Q_ood) - Q(s, a_dataset)
        q_pred_squeezed = q_pred.squeeze(-1)  # (n_critics, batch)
        cql_q_diff = q_ood_logsumexp - q_pred_squeezed  # (n_critics, batch)

        # Clip if specified
        if self.cql_clip_diff_min > -np.inf or self.cql_clip_diff_max < np.inf:
            cql_q_diff = torch.clamp(cql_q_diff, self.cql_clip_diff_min, self.cql_clip_diff_max)

        # Mean over critics and batch
        cql_loss = cql_q_diff.mean()

        info.update({
            "cql_q_diff": cql_q_diff.mean().item(),
            "cql_ood_values": q_ood_logsumexp.mean().item(),
        })

        return cql_loss, info

    # --- Training Methods ---

    def _critic_forward(self, obs, actions, target: bool = False):
        """Forward pass through critic network."""
        features = self.policy.extract_features(obs, detach=False)
        return self.policy.q_values_all(features, actions, target=target)

    def _target_q(self, data: MCReplayBufferSample) -> torch.Tensor:
        """Compute target Q-values for TD loss."""
        alpha = self._current_alpha().detach()

        with torch.no_grad():
            # Sample next actions
            next_action, next_log_prob, _ = self.policy.actor_action_log_prob(
                data.next_obs, detach_encoder=False
            )

            if self.cql_max_target_backup and self.use_cql_loss:
                # CQL max target backup: sample multiple actions and take max
                next_actions_list = [next_action]
                next_log_probs_list = [next_log_prob]

                for _ in range(self.cql_n_actions - 1):
                    action, log_prob = self.policy.actor.action_log_prob(
                        self.policy.extract_features(data.next_obs)
                    )
                    next_actions_list.append(action)
                    next_log_probs_list.append(log_prob)

                # Stack: (batch, n_actions, action_dim)
                next_actions_stacked = torch.stack(next_actions_list, dim=1)
                next_log_probs_stacked = torch.stack(next_log_probs_list, dim=1)

                # Evaluate Q-values for all actions
                batch_size = data.rewards.shape[0]
                next_features = self.policy.extract_features(data.next_obs)
                next_features_repeated = next_features.unsqueeze(1).repeat(
                    1, self.cql_n_actions, 1
                ).reshape(-1, next_features.shape[-1])
                next_actions_flat = next_actions_stacked.reshape(-1, next_actions_stacked.shape[-1])

                q_next_all = self.policy.q_values_subsampled(
                    next_features_repeated,
                    next_actions_flat,
                    subsample_size=self.critic_subsample_size,
                    target=True
                )
                # Reshape: (subsample_size, batch, n_actions)
                q_next_all = q_next_all.reshape(
                    self.critic_subsample_size, batch_size, self.cql_n_actions
                )

                # Min over subsampled critics: (batch, n_actions)
                q_next_min = q_next_all.min(dim=0).values

                # Take max over actions
                max_idx = q_next_min.argmax(dim=1, keepdim=True)  # (batch, 1)
                min_q_next = q_next_min.gather(1, max_idx)  # (batch, 1)
                next_log_prob = next_log_probs_stacked.gather(1, max_idx.unsqueeze(-1)).squeeze(-1)
            else:
                # Standard SAC target
                min_q_next = self.policy.min_q_value(
                    self.policy.extract_features(data.next_obs),
                    next_action,
                    subsample_size=self.critic_subsample_size,
                    target=True
                )

            # Subtract entropy term
            min_q_next = min_q_next - alpha * next_log_prob

            # Compute target
            target = data.rewards.reshape(-1, 1) + (
                1 - data.dones.reshape(-1, 1)
            ) * self.gamma * min_q_next

        return target

    def _critic_loss(self, data: MCReplayBufferSample) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute critic loss (TD + CQL)."""
        info = {}

        # Get current Q-values
        q_pred = self._critic_forward(data.obs, data.actions, target=False)
        # (n_critics, batch, 1)

        # TD loss
        td_loss = torch.tensor(0.0, device=self.device)
        if self.use_td_loss:
            target_q = self._target_q(data)
            target_q_expanded = target_q.unsqueeze(0).repeat(self.n_critics, 1, 1)
            td_loss = F.mse_loss(q_pred, target_q_expanded)
            info["td_loss"] = td_loss.item()
            info["target_q"] = target_q.mean().item()

        # CQL loss
        cql_loss = torch.tensor(0.0, device=self.device)
        if self.use_cql_loss:
            cql_loss_raw, cql_info = self._cql_loss(data, q_pred)
            info.update(cql_info)

            # Apply CQL alpha (with optional auto-tuning)
            cql_alpha = self._current_cql_alpha()
            if self.cql_autotune_alpha:
                # Lagrange penalty: alpha * (cql_loss - target_gap)
                cql_loss = (cql_loss_raw - self.cql_target_action_gap)
            else:
                cql_loss = cql_loss_raw

            cql_loss = cql_alpha * cql_loss
            info["cql_loss"] = cql_loss_raw.item()
            info["cql_alpha"] = cql_alpha.item()

        # Total critic loss
        critic_loss = td_loss + cql_loss
        info["critic_loss"] = critic_loss.item()
        info["predicted_q"] = q_pred.mean().item()

        return critic_loss, info

    def _actor_loss(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute actor loss."""
        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, detach_encoder=False
        )
        min_q = self.policy.min_q_value(
            features, action, subsample_size=self.critic_subsample_size, target=False
        )
        actor_loss = (alpha * log_prob - min_q).mean()
        return actor_loss, log_prob.detach()

    def _cql_alpha_loss(self, data: MCReplayBufferSample) -> torch.Tensor:
        """Compute CQL alpha Lagrange multiplier loss."""
        # Recompute CQL loss without gradients
        with torch.no_grad():
            q_pred = self._critic_forward(data.obs, data.actions, target=False)
            cql_loss_raw, _ = self._cql_loss(data, q_pred)

        # Lagrange penalty
        cql_alpha = self.policy.get_cql_alpha()
        cql_alpha_loss = -cql_alpha * (cql_loss_raw - self.cql_target_action_gap)
        return cql_alpha_loss

    def train(self, gradient_steps: int) -> dict[str, float]:
        """Training step with CQL/Cal-QL support."""
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        cql_alpha_losses: list[float] = []
        alphas: list[float] = []
        cql_alphas: list[float] = []

        # Aggregate info dicts
        info_keys = set()
        info_accum = {}

        for step in range(gradient_steps):
            self._global_update += 1
            data = self.replay_buffer.sample(self.batch_size)

            # --- Critic update ---
            critic_loss, critic_info = self._critic_loss(data)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()
            critic_losses.append(critic_loss.item())

            # Accumulate info
            for k, v in critic_info.items():
                if k not in info_accum:
                    info_accum[k] = []
                    info_keys.add(k)
                info_accum[k].append(v)

            # --- Actor + alpha updates ---
            if self._global_update % self.policy_frequency == 0:
                actor_loss, log_prob_detached = self._actor_loss(data.obs)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                # Entropy coefficient update
                if self.autotune:
                    alpha_loss = -(
                        self.log_alpha.exp() * (log_prob_detached + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_losses.append(alpha_loss.item())

                # CQL alpha Lagrange multiplier update
                if self.cql_autotune_alpha:
                    cql_alpha_loss = self._cql_alpha_loss(data)
                    self.cql_alpha_optimizer.zero_grad()
                    cql_alpha_loss.backward()
                    self.cql_alpha_optimizer.step()
                    cql_alpha_losses.append(cql_alpha_loss.item())

            alphas.append(self._current_alpha().item())
            if self.cql_autotune_alpha or self.use_cql_loss:
                cql_alphas.append(self._current_cql_alpha().item())

            # --- Target critic update ---
            if self._global_update % self.target_network_frequency == 0:
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )

        # Aggregate results
        out = {
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": float(np.mean(alphas)) if alphas else 0.0,
        }

        if alpha_losses:
            out["alpha_loss"] = float(np.mean(alpha_losses))
        if cql_alpha_losses:
            out["cql_alpha_loss"] = float(np.mean(cql_alpha_losses))
        if cql_alphas:
            out["cql_alpha"] = float(np.mean(cql_alphas))

        # Add accumulated info
        for k in info_keys:
            out[k] = float(np.mean(info_accum[k]))

        return out

