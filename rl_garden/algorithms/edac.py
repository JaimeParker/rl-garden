"""EDAC: SAC-N (offline SAC, large critic ensemble) plus a gradient-
diversity penalty across the ensemble.

Ported from ``3rd_party/CORL/algorithms/offline/edac.py`` (arXiv
2110.01548). ``OfflineSAC`` (``rl_garden/algorithms/offline_sac.py``)
already implements everything EDAC needs except the diversity term: full
critic-ensemble entropy-corrected SAC (``SACCore``'s defaults --
``_backup_entropy_enabled()``→True, ``_target_critic_subsample_size()``→None,
neither overridden by EDAC), and -- critically -- ``OfflineSAC`` already
overrides ``_td_loss`` to sum per-critic MSE rather than ``SACCore``'s
default combined-MSE-over-the-stacked-ensemble form, which happens to be
**exactly** EDAC's own critic-loss convention
(``edac.py:422``, ``((q-target)**2).mean(1).sum(0)``). So ``EDAC`` subclasses
``OfflineSAC`` directly rather than rebuilding a parallel ``SACCore``-based
hierarchy -- the only actual difference is one new loss term.

Formula verified against ``edac.py:361-427``:
```
critic_loss = td_loss + eta * diversity_loss
```
where ``diversity_loss`` is a per-pair cosine-similarity penalty across the
ensemble's action-gradients: for each critic ``i``, ``g_i = ∇_a Q_i(s,a)``
at the real batch ``(s,a)``, L2-normalized, then
``mean_batch[ sum_{i≠j} g_i·g_j ] / (num_critics - 1)``.

**Implementation note** (no changes to ``EnsembleQCritic`` needed): the
reference repeats the action across a critic dimension then takes one
vectorized JAX grad; ``EnsembleQCritic.forward_all``'s ``torch.func.vmap``
broadcasts ``features``/``actions`` identically to every critic (confirmed
by a failed sanity check this session -- passing a pre-repeated per-critic
action tensor does not produce per-critic-isolated gradients the way it
does under JAX's native vmap+grad). The PyTorch equivalent used here calls
``EnsembleQCritic.forward()`` (the tuple-returning method) to get
``num_critics`` separate scalar outputs, then ``torch.autograd.grad(...,
create_graph=True)`` **once per critic** -- verified this session (a
linearity self-check: summing the per-critic grads equals the grad of the
summed output) to give mathematically identical, genuinely per-critic-
distinct gradients. ``create_graph=True`` is required because this
gradient is itself part of the differentiable loss (a standard "gradient
penalty" double-backward pattern, e.g. WGAN-GP) -- confirmed working
end-to-end: a combined ``td_loss + diversity_loss``'s ``.backward()``
correctly populates critic parameter gradients.

**Deliberate, documented non-reproduction**: CORL's own EDAC updates in
``alpha → actor → critic`` order (their comment: "we found EDAC paper uses
reverse [order], which gives better results" -- an explicitly optional
empirical tweak, not part of the algorithm's definition). Reproducing it
would need a full override of ``SACCore.train()`` (a large method with
high-UTD/``policy_frequency``/``target_network_frequency``/diagnostics logic
unrelated to EDAC's actual contribution) for a minor, authors-flagged-as-
optional ordering difference -- not worth it. This port keeps the existing
``critic → actor → alpha`` order (``SACCore``/``OfflineSAC``'s own
established convention).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.common.logger import Logger


class EDAC(OfflineSAC):
    """Offline EDAC: SAC-N (large critic ensemble) + gradient-diversity
    penalty. See module docstring."""

    _compatible_checkpoint_algorithms = ("EDAC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        eta: float = 1.0,
        n_critics: int = 10,
        **kwargs: Any,
    ) -> None:
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}.")
        super().__init__(env=env, n_critics=n_critics, **kwargs)
        self.eta = eta
        # OfflineSAC.__init__ never sets self.backup_entropy, which
        # SACCore._backup_entropy_enabled() requires (a pre-existing gap in
        # OfflineSAC itself, not touched here -- see this session's report).
        # EDAC's own reference entropy-corrects its critic target
        # (edac.py:415, `q_next - alpha*next_log_prob`), so this is also the
        # numerically correct value for EDAC specifically, not just a
        # workaround.
        self.backup_entropy = True

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {**super()._checkpoint_metadata(), "eta": self.eta}

    def _critic_diversity_loss(self, data) -> torch.Tensor:
        features = self.policy.extract_features(data.obs)
        actions = data.actions.detach().requires_grad_(True)
        qs = self.policy.critic.forward(features, actions)
        grads = []
        for q_i in qs:
            (g,) = torch.autograd.grad(q_i.sum(), actions, retain_graph=True, create_graph=True)
            grads.append(g)
        grads = torch.stack(grads, dim=0)  # (N, B, A)
        grads = grads / (grads.norm(dim=-1, keepdim=True) + 1e-10)
        grads = grads.transpose(0, 1)  # (B, N, A)
        sim = grads @ grads.transpose(1, 2)  # (B, N, N)
        mask = 1.0 - torch.eye(self.n_critics, device=grads.device)
        diversity_loss = (sim * mask).sum(dim=(1, 2)).mean() / (self.n_critics - 1)
        return diversity_loss

    def _critic_loss(self, data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_pred = self._critic_forward(data.obs, data.actions, target=False)
        td_loss, info = self._td_loss(data, q_pred)
        diversity_loss = self._critic_diversity_loss(data)
        critic_loss = td_loss + self.eta * diversity_loss
        info["td_loss"] = td_loss.detach()
        info["diversity_loss"] = diversity_loss.detach()
        info["critic_loss"] = critic_loss.detach()
        info["predicted_q"] = q_pred.mean().detach()
        return critic_loss, info
