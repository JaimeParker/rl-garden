"""Multitask world model, ported from the ``cfg.multitask`` branches of
``3rd_party/tdmpc2/tdmpc2/common/world_model.py``.

Deliberately a separate class from ``rl_garden.algorithms.tdmpc2.world_model.
WorldModel`` rather than an ``if multitask:`` branch bolted onto it: nearly
every method's call signature changes (a required ``task`` argument), and the
encoder doesn't go through rl-garden's ``BaseFeaturesExtractor`` at all (see
module docstring below) -- keeping this fully separate means the already
tested single-task path is never touched.

Task sets (upstream's ``mt30``/``mt80``) are state-only (no pixel
observations), so the encoder here is a plain MLP over ``[obs; task_emb]``
built with ``rl_garden.algorithms.tdmpc2.layers.mlp`` -- matching upstream's
own ``common.layers.enc(cfg)`` state branch -- rather than reusing
``FlattenExtractor``/``CombinedExtractor``: those extractors are called on raw
``obs`` alone and know nothing about concatenating a task embedding first,
which upstream does *before* the encoder (``task_emb(obs, task)`` then
``_encoder[obs_key](obs)``, see upstream ``world_model.py:108-112``).

Observations/actions are expected to already be zero-padded to
``max(obs_dims)``/``max(action_dims)`` by the caller (the replay buffer /
dataset loader), matching upstream's ``MultitaskWrapper._pad_obs`` /
``step()`` truncation scheme (``envs/wrappers/multitask.py:44-57``).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rl_garden.algorithms.tdmpc2 import math_utils
from rl_garden.algorithms.tdmpc2.layers import QEnsemble, RunningScale, SimNorm, mlp


class MultitaskWorldModel(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        obs_dims: Sequence[int],
        action_dims: Sequence[int],
        task_dim: int = 96,
        latent_dim: int = 512,
        enc_dim: int = 256,
        num_enc_layers: int = 2,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_q: int = 5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tau: float = 0.01,
    ) -> None:
        super().__init__()
        if len(obs_dims) != num_tasks or len(action_dims) != num_tasks:
            raise ValueError(
                f"obs_dims/action_dims must have length num_tasks={num_tasks}, "
                f"got {len(obs_dims)}/{len(action_dims)}."
            )
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        self.obs_dim = max(obs_dims)
        self.action_dim = max(action_dims)
        self.latent_dim = latent_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_size = (vmax - vmin) / (num_bins - 1) if num_bins > 1 else 0.0
        self.tau = tau

        self.task_emb = nn.Embedding(num_tasks, task_dim, max_norm=1)
        action_masks = torch.zeros(num_tasks, self.action_dim)
        for i, adim in enumerate(action_dims):
            action_masks[i, :adim] = 1.0
        self.register_buffer("_action_masks", action_masks)

        enc_hidden = max(num_enc_layers - 1, 1) * [enc_dim]
        self._encoder = mlp(self.obs_dim + task_dim, enc_hidden, latent_dim, act=SimNorm(simnorm_dim))
        self._dynamics = mlp(
            latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], latent_dim, act=SimNorm(simnorm_dim)
        )
        self._reward = mlp(latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], max(num_bins, 1))
        self._pi = mlp(latent_dim + task_dim, 2 * [mlp_dim], 2 * self.action_dim)
        self._Q = QEnsemble(
            latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], num_bins, num_q, dropout=dropout
        )
        self._target_Q = QEnsemble(
            latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], num_bins, num_q, dropout=dropout
        )
        self._target_Q.load_state_dict(self._Q.state_dict())
        for p in self._target_Q.parameters():
            p.requires_grad_(False)

        self.scale = RunningScale(tau=tau)

        self.register_buffer("log_std_min", torch.tensor(log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(log_std_max - log_std_min))

        self._apply_init()

    def _apply_init(self) -> None:
        from rl_garden.algorithms.tdmpc2 import init as tdmpc2_init

        self.apply(tdmpc2_init.weight_init)
        tdmpc2_init.zero_([self._reward[-1].weight])
        for q in self._Q.qs:
            tdmpc2_init.zero_([q[-1].weight])

    def soft_update_target_Q(self) -> None:
        from rl_garden.common.utils import polyak_update

        polyak_update(self._Q.parameters(), self._target_Q.parameters(), self.tau)

    def task_embed(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Concatenate the task embedding onto ``x``'s last dim.

        ``x`` may be ``(B, D)`` or ``(T, B, D)`` (a training-time window
        batch); ``task`` is ``(B,)`` (one task id per batch element, constant
        across any window -- see ``MmapMultitaskEpisodeBuffer``).
        """
        emb = self.task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1 and x.shape[0] != 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return self._encoder(self.task_embed(obs, task))

    def next(self, z: torch.Tensor, a: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        z = self.task_embed(z, task)
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        z = self.task_embed(z, task)
        return self._reward(torch.cat([z, a], dim=-1))

    def pi(
        self, z: torch.Tensor, task: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.task_embed(z, task)
        mean, log_std_raw = self._pi(z).chunk(2, dim=-1)
        log_std_ = math_utils.log_std(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        action_mask = self._action_masks[task]
        mean = mean * action_mask
        log_std_ = log_std_ * action_mask
        eps = eps * action_mask
        action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)

        log_prob = math_utils.gaussian_logprob(eps, log_std_)
        scaled_log_prob = log_prob * action_dims

        action = mean + eps * log_std_.exp()
        mean, action, log_prob = math_utils.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std_,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        task: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        assert return_type in ("min", "avg", "all")
        z = self.task_embed(z, task)
        za = torch.cat([z, a], dim=-1)
        qnet = self._target_Q if target else self._Q
        out = qnet(za)
        if detach and not target:
            out = out.detach()

        if return_type == "all":
            return out

        qidx = torch.randperm(self.num_q, device=out.device)[:2]
        q_scalar = math_utils.two_hot_inv(out[qidx], self.num_bins, self.vmin, self.vmax)
        if return_type == "min":
            return q_scalar.min(0).values
        return q_scalar.sum(0) / 2
