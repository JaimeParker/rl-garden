"""RLPDHybrid: RLPD's continuous ee-pose actor/critic unchanged, plus an
independent discrete Double-DQN head for a hybrid continuous+discrete
gripper action space (HIL-SERL's ``sac_hybrid_single``). No existing
algorithm in this repo adds a brand-new network+optimizer via subclassing
(RLPD-over-SAC and TD3-over-DDPG only add hooks/hyperparameters onto
networks the parent already owns), so this follows ``AGENTS.md``'s
explicit-optimizer-ownership rule directly: ``discrete_critic`` and
``dqn_optimizer`` are separate, explicit attributes, never folded into
``q_optimizer``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from rl_garden.algorithms.rlpd import RLPD
from rl_garden.buffers.demo_intervention import DemoInterventionMixin
from rl_garden.common.optim import make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.rlpd_hybrid_policy import _DISCRETE_ACTION_VALUES, RLPDHybridPolicy


def _discrete_labels_from_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map the recorded gripper action (last action dim, one of
    ``_DISCRETE_ACTION_VALUES``) back to its discrete index."""
    values = actions.new_tensor(_DISCRETE_ACTION_VALUES)
    gripper = actions[..., -1:]
    return (gripper - values).abs().argmin(dim=-1)


class RLPDHybrid(DemoInterventionMixin, RLPD):
    _compatible_checkpoint_algorithms = ("RLPDHybrid",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        discrete_hidden_dim: int = 256,
        discrete_lr: float = 3e-4,
        discrete_tau: Optional[float] = None,
        **rlpd_kwargs: Any,
    ) -> None:
        self.discrete_hidden_dim = discrete_hidden_dim
        self.discrete_lr = discrete_lr
        self._discrete_tau = discrete_tau
        super().__init__(env, eval_env, **rlpd_kwargs)

    def _build_policy(self, features_extractor: BaseFeaturesExtractor) -> RLPDHybridPolicy:
        return RLPDHybridPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_impl=self.critic_impl,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            use_pnorm=self.use_pnorm,
            log_std_min=self.actor_log_std_min,
            log_std_mode=self.actor_log_std_mode,
            actor_feature_dim=self.actor_feature_dim,
            critic_spatial_emb_dim=self.critic_spatial_emb_dim,
            discrete_hidden_dims=(self.discrete_hidden_dim,),
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self.dqn_optimizer = make_optimizer(
            list(self.policy.discrete_critic_parameters()),
            lr=self.discrete_lr,
        )

    def _critic_forward(self, obs, actions, target: bool = False):
        # The continuous critic was built for the non-gripper action dims
        # only (RLPDHybridPolicy strips the last dim); ``actions`` here is
        # always a raw recorded action from the replay buffer (full env
        # action width), never an actor-generated sample -- SACCore only
        # calls this from `_critic_loss`, and the actor-loss path goes
        # through `policy.min_q_value` with the actor's own (already
        # correctly-shaped) continuous sample instead.
        return super()._critic_forward(obs, actions[..., :-1], target=target)

    @property
    def _discrete_tau_value(self) -> float:
        return self._discrete_tau if self._discrete_tau is not None else self.tau

    def _train_discrete_critic(self, gradient_steps: int) -> None:
        for _ in range(gradient_steps):
            data = self._sample_train_batch(self.batch_size)
            self.policy.features_extractor.prepare_batch(data.obs, data.next_obs)
            labels = _discrete_labels_from_actions(data.actions)

            features = self.policy.extract_features(data.obs, stop_gradient=True)
            q_pred = self.policy.discrete_critic(features).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_features = self.policy.extract_features(data.next_obs, stop_gradient=True)
                next_online_q = self.policy.discrete_critic(next_features)
                next_action = next_online_q.argmax(dim=-1, keepdim=True)
                next_target_q = self.policy.discrete_target_critic(next_features)
                next_q = next_target_q.gather(-1, next_action).squeeze(-1)
                target = data.rewards.reshape(-1) + (1 - data.dones.reshape(-1)) * self.gamma * next_q

            loss = F.mse_loss(q_pred, target)
            self.dqn_optimizer.zero_grad()
            loss.backward()
            self.dqn_optimizer.step()

            polyak_update(
                self.policy.discrete_critic.parameters(),
                self.policy.discrete_target_critic.parameters(),
                self._discrete_tau_value,
            )

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        info = super().train(gradient_steps, compute_info)
        self._train_discrete_critic(gradient_steps)
        return info

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*super()._optimizer_names(), "dqn_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "discrete_hidden_dim": self.discrete_hidden_dim,
            "discrete_lr": self.discrete_lr,
        }
