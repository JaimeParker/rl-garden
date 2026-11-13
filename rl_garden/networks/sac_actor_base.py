"""The base class of SAC-family actors: ``rl_garden.policies.contracts.SACActor``
with its defaults."""

from __future__ import annotations

import sys
from typing import ClassVar

import torch
import torch.nn as nn


class SACActorBase(nn.Module):
    """A SAC-family actor over ``(B, features_dim)`` features.

    A subclass implements ``action_log_prob`` (a reparameterized sample and its
    ``(B, 1)`` log-probability) and ``deterministic_action``, and overrides a
    class attribute when its actor differs from a stochastic density actor
    that cannot score external actions:

    - ``stochastic = False``: a deterministic (TD3-shaped) actor, whose
      ``action_log_prob`` returns a zero ``(B, 1)`` log-probability;
    - ``log_prob_is_density = False``: the log-probability is a path
      likelihood (a flow actor), not the density of the returned action;
    - ``supports_evaluate_log_prob = True``: ``evaluate_action_log_prob``
      scores external (e.g. dataset) actions.
    """

    stochastic: ClassVar[bool] = True
    log_prob_is_density: ClassVar[bool] = True
    supports_evaluate_log_prob: ClassVar[bool] = False

    def action_log_prob(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A reparameterized ``(action (B, A), log_prob (B, 1))``."""
        raise NotImplementedError(f"{type(self).__name__} must implement action_log_prob")

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        """The ``(B, A)`` action used for deterministic evaluation."""
        raise NotImplementedError(f"{type(self).__name__} must implement deterministic_action")

    def sample_actions_with_log_prob(
        self, features: torch.Tensor, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``n`` reparameterized samples per row, ``(B, n, A)`` actions and
        ``(B, n)`` log-probabilities, from one ``action_log_prob`` call on the
        features repeated ``n`` times along a new leading axis."""
        batch = features.shape[0]
        repeated = features.unsqueeze(0).expand(n, *features.shape).reshape(
            n * batch, *features.shape[1:]
        )
        actions, log_prob = self.action_log_prob(repeated)
        actions = actions.reshape(n, batch, *actions.shape[1:]).transpose(0, 1)
        log_prob = log_prob.reshape(n, batch).transpose(0, 1)
        return actions, log_prob

    def evaluate_action_log_prob(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """The ``(B, 1)`` log-probability of external ``actions``. A subclass
        that implements it sets ``supports_evaluate_log_prob = True``."""
        caller = sys._getframe(1).f_code
        raise NotImplementedError(
            f"{type(self).__name__} cannot score external actions "
            "(supports_evaluate_log_prob=False); required by "
            f"{getattr(caller, 'co_qualname', caller.co_name)}"
        )

    def diagnostics(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detached statistics of one sample: the log-probability's mean and
        (unbiased) std and the mean absolute action."""
        with torch.no_grad():
            actions, log_prob = self.action_log_prob(features)
        return {
            "log_prob_mean": log_prob.mean(),
            "log_prob_std": log_prob.std(),
            "action_abs_mean": actions.abs().mean(),
        }


__all__ = ["SACActorBase"]
