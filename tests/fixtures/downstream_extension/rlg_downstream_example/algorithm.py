"""RecurrentPPO with a two-head action mean and an auxiliary BCE head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rl_garden.algorithms import RecurrentPPO
from rl_garden.algorithms.ppo import masked_mean
from rl_garden.common.effective_config import ConfigError
from rlg_downstream_example.policy import AUX_TARGET_KEY, AuxSplitRecurrentPolicy


class AuxSplitRecurrentPPO(RecurrentPPO):
    """``RecurrentPPO`` whose policy is ``AuxSplitRecurrentPolicy``, plus
    ``aux_weight * BCE(aux_head(latent), infos["aux_target"])``, reduced over
    decision rows when the window has a decision mask."""

    policy_class = AuxSplitRecurrentPolicy
    action_split = 2

    def __init__(self, env: Any, *, aux_weight: float = 0.1, aux_hidden: int = 16, **kwargs: Any):
        if kwargs.get("policy_graph") is not None:
            raise ConfigError(f"{type(self).__name__} builds its own policy; drop --policy-graph.")
        # Set before super().__init__(): it ends by building the policy,
        # which reads them.
        self.aux_weight = aux_weight
        self.aux_hidden = aux_hidden
        self._aux_logit_means: list[torch.Tensor] = []
        super().__init__(env, **kwargs)

    # --- network -------------------------------------------------------------

    def _policy_head_kwargs(self, observation_space):
        return {
            **super()._policy_head_kwargs(observation_space),
            "action_split": self.action_split,
            "aux_hidden": self.aux_hidden,
        }

    # --- auxiliary loss ------------------------------------------------------

    def _aux_loss(self, data, evaluation, decision_mask):
        aux_loss, terms = super()._aux_loss(data, evaluation, decision_mask)
        logits = self.policy.aux_head(evaluation.outputs["latent"])
        target = data.extras[AUX_TARGET_KEY].reshape(-1, 1).to(logits.dtype)
        row_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")[:, 0]
        if decision_mask is None:
            bce = row_loss.mean()
        else:
            bce = masked_mean(row_loss, decision_mask, decision_mask.sum())
        self._aux_logit_means.append(logits.detach().mean())
        weighted = self.aux_weight * bce
        aux_loss = weighted if aux_loss is None else aux_loss + weighted
        # train() returns the terms as aux/<name>; they are logged as
        # losses/aux/<name>.
        return aux_loss, {**terms, "bce": bce.detach()}

    # --- metric --------------------------------------------------------------

    def train(self) -> dict[str, float]:
        self._aux_logit_means = []
        metrics = super().train()
        if self._aux_logit_means:
            # One host sync per update, not per minibatch.
            metrics["aux_logit_mean"] = torch.stack(self._aux_logit_means).mean().item()
        return metrics

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Every train() key is logged as losses/<key>; this one goes to train/.
        metrics = dict(metrics)
        aux_logit_mean = metrics.pop("aux_logit_mean", None)
        super()._log_update_metrics(metrics, step)
        if aux_logit_mean is not None:
            self._log_rollout_metric("aux_logit_mean", aux_logit_mean, step)

    # --- checkpoint ----------------------------------------------------------

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "aux_weight": self.aux_weight,
            "aux_hidden": self.aux_hidden,
        }
