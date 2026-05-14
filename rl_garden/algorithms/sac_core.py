"""Shared SAC-family update core.

This mixin owns actor/critic/temperature update mechanics for SAC-like
algorithms. Rollout, evaluation, checkpointing, and replay-buffer lifecycle stay
in the algorithm shells (``OffPolicyAlgorithm`` / ``OfflineRLAlgorithm``).
Subclasses may override loss hooks to add CQL/Cal-QL while reusing the update
schedule, REDQ target subsampling, high-UTD splitting, optimizer stepping, and
target-network updates.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rl_garden.common.utils import polyak_update


class SACCore:
    """Mixin containing SAC-family update logic."""

    # --- hooks subclasses may override ---

    def _sample_train_batch(self, batch_size: int):
        return self.replay_buffer.sample(batch_size)

    def _actor_stop_gradient(self) -> bool:
        return False

    def _backup_entropy_enabled(self) -> bool:
        return True

    def _target_critic_subsample_size(self) -> Optional[int]:
        return getattr(self, "critic_subsample_size", None)

    def _step_critic_scheduler(self) -> None:
        return None

    def _step_actor_scheduler(self) -> None:
        return None

    def _clip_grad_norm(self, params) -> None:
        return None

    def _alpha_loss(self, log_prob_detached: torch.Tensor) -> torch.Tensor:
        return -(
            self._current_alpha() * (log_prob_detached + self.target_entropy)
        ).mean()

    def _alpha_parameters(self):
        if getattr(self, "autotune", False) and getattr(self, "log_alpha", None) is not None:
            yield self.log_alpha
        elif getattr(self, "autotune", False) and getattr(self, "temperature_lagrange", None):
            yield from self.temperature_lagrange.parameters()

    def _post_actor_update(self, data) -> dict[str, float]:
        return {}

    def _target_update(self) -> None:
        polyak_update(
            self.policy.critic.parameters(),
            self.policy.critic_target.parameters(),
            self.tau,
        )

    def _actor_action_log_prob(
        self,
        obs,
        *,
        base_actions: Optional[torch.Tensor] = None,
        stop_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del base_actions
        return self.policy.actor_action_log_prob(obs, stop_gradient=stop_gradient)

    def _target_action_log_prob(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._actor_action_log_prob(data.next_obs, stop_gradient=False)

    def _actor_loss_from_batch(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        return self._actor_loss(data.obs)

    # --- SAC losses ---

    def _critic_forward(self, obs, actions, target: bool = False):
        features = self.policy.extract_features(obs, stop_gradient=False)
        return self.policy.q_values_all(features, actions, target=target)

    def _target_q(self, data) -> torch.Tensor:
        alpha = self._current_alpha().detach()
        with torch.no_grad():
            next_action, next_log_prob, next_features = self._target_action_log_prob(data)
            min_q_next = self.policy.min_q_value(
                next_features,
                next_action,
                subsample_size=self._target_critic_subsample_size(),
                target=True,
            )
            if self._backup_entropy_enabled():
                min_q_next = min_q_next - alpha * next_log_prob
            target = data.rewards.reshape(-1, 1) + (
                1 - data.dones.reshape(-1, 1)
            ) * self.gamma * min_q_next
        return target

    def _td_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target_q = self._target_q(data)
        target_q_expanded = target_q.unsqueeze(0).expand_as(q_pred)
        td_loss = F.mse_loss(q_pred, target_q_expanded)
        return td_loss, {
            "td_loss": td_loss.item(),
            "target_q": target_q.mean().item(),
        }

    def _critic_loss(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        q_pred = self._critic_forward(data.obs, data.actions, target=False)
        td_loss, info = self._td_loss(data, q_pred)
        info["critic_loss"] = td_loss.item()
        info["predicted_q"] = q_pred.mean().item()
        return td_loss, info

    def _actor_loss(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, features = self._actor_action_log_prob(
            obs, stop_gradient=self._actor_stop_gradient()
        )
        min_q = self.policy.min_q_value(features, action, subsample_size=None, target=False)
        return (alpha * log_prob - min_q).mean(), log_prob.detach()

    # --- update loops ---

    @staticmethod
    def _mean_infos(infos: list[dict[str, float]]) -> dict[str, float]:
        if not infos:
            return {}
        keys = set().union(*(info.keys() for info in infos))
        return {
            key: float(np.mean([info[key] for info in infos if key in info]))
            for key in keys
        }

    def train(self, gradient_steps: int) -> dict[str, float]:
        high_utd_ratio = int(self.utd) if float(self.utd).is_integer() else 1
        if high_utd_ratio > 1:
            groups = gradient_steps // high_utd_ratio
            remainder = gradient_steps % high_utd_ratio
            infos: list[dict[str, float]] = []
            for _ in range(groups):
                infos.append(self.train_high_utd(utd_ratio=high_utd_ratio))
            if remainder:
                old_utd = self.utd
                self.utd = 1.0
                try:
                    infos.append(self.train(remainder))
                finally:
                    self.utd = old_utd
            return self._mean_infos(infos)

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        alphas: list[float] = []
        info_accum: dict[str, list[float]] = {}

        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            critic_loss, critic_info = self._critic_loss(data)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.q_optimizer.step()
            self._step_critic_scheduler()
            critic_losses.append(critic_loss.item())

            for key, value in critic_info.items():
                info_accum.setdefault(key, []).append(value)

            if self._global_update % self.policy_frequency == 0:
                actor_loss, log_prob_detached = self._actor_loss_from_batch(data)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self._clip_grad_norm(self.policy.actor_parameters())
                self.actor_optimizer.step()
                self._step_actor_scheduler()
                actor_losses.append(actor_loss.item())

                if self.autotune:
                    alpha_loss = self._alpha_loss(log_prob_detached)
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self._clip_grad_norm(self._alpha_parameters())
                    self.alpha_optimizer.step()
                    alpha_losses.append(alpha_loss.item())

                for key, value in self._post_actor_update(data).items():
                    info_accum.setdefault(key, []).append(value)

            alphas.append(self._current_alpha().item())

            if self._global_update % self.target_network_frequency == 0:
                self._target_update()

        out = {
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "qf_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": float(np.mean(alphas)) if alphas else 0.0,
        }
        if alpha_losses:
            out["alpha_loss"] = float(np.mean(alpha_losses))
        for key, values in info_accum.items():
            out[key] = float(np.mean(values))
        return out

    def train_high_utd(self, utd_ratio: int) -> dict[str, float]:
        assert utd_ratio >= 1, f"utd_ratio must be >= 1, got {utd_ratio}"
        assert (
            self.batch_size % utd_ratio == 0
        ), f"batch_size ({self.batch_size}) must be divisible by utd_ratio ({utd_ratio})"

        full_batch = self._sample_train_batch(self.batch_size)
        minibatch_size = self.batch_size // utd_ratio

        critic_losses: list[float] = []
        info_accum: dict[str, list[float]] = {}

        for j in range(utd_ratio):
            self._global_update += 1
            mb = self._slice_batch(full_batch, j * minibatch_size, minibatch_size)

            critic_loss, critic_info = self._critic_loss(mb)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.q_optimizer.step()
            self._step_critic_scheduler()
            critic_losses.append(critic_loss.item())

            for key, value in critic_info.items():
                info_accum.setdefault(key, []).append(value)

            if self._global_update % self.target_network_frequency == 0:
                self._target_update()

        actor_loss, log_prob_detached = self._actor_loss_from_batch(full_batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_grad_norm(self.policy.actor_parameters())
        self.actor_optimizer.step()
        self._step_actor_scheduler()

        alpha_loss_val = None
        if self.autotune:
            alpha_loss = self._alpha_loss(log_prob_detached)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._clip_grad_norm(self._alpha_parameters())
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        out: dict[str, float] = {
            "critic_loss": float(np.mean(critic_losses)),
            "qf_loss": float(np.mean(critic_losses)),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self._current_alpha().item()),
            "utd_ratio": float(utd_ratio),
        }
        if alpha_loss_val is not None:
            out["alpha_loss"] = alpha_loss_val
        for key, value in self._post_actor_update(full_batch).items():
            out[key] = value
        for key, values in info_accum.items():
            out[key] = float(np.mean(values))
        return out

    @staticmethod
    def _slice_batch(batch: Any, start: int, size: int):
        end = start + size

        def _slice(x):
            if isinstance(x, dict):
                return {k: v[start:end] for k, v in x.items()}
            if x is None:
                return None
            return x[start:end]

        kwargs = {
            "obs": _slice(batch.obs),
            "next_obs": _slice(batch.next_obs),
            "actions": _slice(batch.actions),
            "rewards": _slice(batch.rewards),
            "dones": _slice(batch.dones),
        }
        if hasattr(batch, "mc_returns"):
            kwargs["mc_returns"] = _slice(batch.mc_returns)
        if hasattr(batch, "base_actions"):
            kwargs["base_actions"] = _slice(batch.base_actions)
        if hasattr(batch, "next_base_actions"):
            kwargs["next_base_actions"] = _slice(batch.next_base_actions)
        return type(batch)(**kwargs)
