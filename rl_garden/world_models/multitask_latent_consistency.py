"""``MultitaskWorldModel``, ported from the ``cfg.multitask`` branches of
``3rd_party/tdmpc2/tdmpc2/common/world_model.py``.

Deliberately a separate class from ``rl_garden.world_models.latent_consistency
.LatentConsistencyModel`` rather than a ``task_embedding=...`` branch folded
into it: nearly every method's call signature changes (a ``task`` argument,
threaded as an optional keyword per the model-based-base plan's "task kwarg
threading" note purely so each overriding method's *signature* still matches
``WorldModel``'s -- every method that needs it (``encode``/``step``/
``reward``) asserts it is not ``None`` at runtime, so ``task`` is actually
**required** for this subclass, not truly optional; this class is therefore
not usable through a generic ``WorldModel`` consumer that doesn't know to
pass ``task``, e.g. ``rl_garden.world_models.imagine.imagine()`` -- only
``TDMPC2Multitask``, which always threads a task id through, drives this
model), and the encoder doesn't go through rl-garden's
``BaseFeaturesExtractor`` at all (see module docstring below) -- keeping
this fully separate means the already tested single-task path is never
touched.

Task sets (upstream's ``mt30``/``mt80``) are state-only (no pixel
observations), so the encoder here is a plain MLP over ``[obs; task_emb]``
built with ``rl_garden.networks.normed_mlp.mlp`` -- matching upstream's
own ``common.layers.enc(cfg)`` state branch -- rather than reusing
``FlattenExtractor``/``CombinedExtractor``: those extractors are called on raw
``obs`` alone and know nothing about concatenating a task embedding first,
which upstream does *before* the encoder (``task_emb(obs, task)`` then
``_encoder[obs_key](obs)``, see upstream ``world_model.py:108-112``). Because
this encoder isn't a ``BaseFeaturesExtractor``, this class does not rely on
``WorldModel``'s ``BaseFeaturesExtractor``-duck-typing surface
(``extract``/``features_dim``/``update_normalizer``) -- it is never used as a
``BasePolicy`` actor_extractor (``MultitaskTDMPC2Policy`` stays out of the
``BasePolicy`` contract, see that module's docstring).

Observations/actions are expected to already be zero-padded to
``max(obs_dims)``/``max(action_dims)`` by the caller (the replay buffer /
dataset loader), matching upstream's ``MultitaskWrapper._pad_obs`` /
``step()`` truncation scheme (``envs/wrappers/multitask.py:44-57``).

No termination head (``continue_`` always returns ``None``): upstream's own
multitask task sets default to non-episodic, and this port's multitask
training never runs a live rollout that could observe a true terminal
transition (see ``rl_garden.algorithms.tdmpc2_multitask``'s module docstring).
"""
from __future__ import annotations

import itertools
from typing import Iterator, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.networks.normed_mlp import SimNorm, mlp
from rl_garden.networks.twohot import soft_ce
from rl_garden.world_models.base import State, WorldModel


class MultitaskWorldModel(WorldModel):
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
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        rho: float = 0.5,
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
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_size = (vmax - vmin) / (num_bins - 1) if num_bins > 1 else 0.0
        self.episodic = False  # no termination head, see module docstring.
        self.rho = rho

        self.task_emb = nn.Embedding(num_tasks, task_dim, max_norm=1)
        action_masks = torch.zeros(num_tasks, self.action_dim)
        for i, adim in enumerate(action_dims):
            action_masks[i, :adim] = 1.0
        self.register_buffer("_action_masks", action_masks)

        enc_hidden = max(num_enc_layers - 1, 1) * [enc_dim]
        self.encoder = mlp(self.obs_dim + task_dim, enc_hidden, latent_dim, act=SimNorm(simnorm_dim))
        self._dynamics = mlp(
            latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], latent_dim, act=SimNorm(simnorm_dim)
        )
        self._reward = mlp(latent_dim + self.action_dim + task_dim, 2 * [mlp_dim], max(num_bins, 1))
        # NOTE: weight init deliberately NOT run here -- see apply_init()'s
        # docstring (mirrors LatentConsistencyModel.apply_init()) for why
        # it's deferred to (and orchestrated by) MultitaskTDMPC2Policy.__init__.

    # ------------------------------------------------------------------
    # Initialization (deferred -- see docstring)
    # ------------------------------------------------------------------

    def apply_init(self) -> None:
        """Initializes this model's own modules via TD-MPC2's
        ``trunc_normal_`` init, with the reward head's last layer zeroed.
        Deliberately not called from ``__init__`` -- see
        ``rl_garden.world_models.latent_consistency.LatentConsistencyModel
        .apply_init()``'s docstring for the RNG-stream-ordering rationale,
        which applies identically here (``MultitaskTDMPC2Policy.__init__``
        orchestrates the same construct-everything-then-init-everything
        sequence)."""
        from rl_garden.networks import init as net_init

        self.apply(net_init.weight_init)
        net_init.zero_([self._reward[-1].weight])

    # ------------------------------------------------------------------
    # Task conditioning
    # ------------------------------------------------------------------

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

    def action_mask(self, task: torch.Tensor) -> torch.Tensor:
        return self._action_masks[task]

    # ------------------------------------------------------------------
    # WorldModel interface (``task`` threaded as an optional kwarg -- see
    # module docstring)
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert task is not None, "MultitaskWorldModel.encode requires task"
        return self.encoder(self.task_embed(obs, task))

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return {"z": torch.zeros(batch_size, self.latent_dim, device=device)}

    def observe(
        self,
        state: Optional[State],
        action: Optional[torch.Tensor],
        embed: torch.Tensor,
        is_first: torch.Tensor,
        task: Optional[torch.Tensor] = None,
    ) -> State:
        del state, action, is_first, task  # no recurrent carry, see LatentConsistencyModel.
        return {"z": embed}

    def step(
        self, state: State, action: torch.Tensor, *, sample: bool = True, task: Optional[torch.Tensor] = None
    ) -> State:
        del sample
        assert task is not None, "MultitaskWorldModel.step requires task"
        z = self.task_embed(state["z"], task)
        return {"z": self._dynamics(torch.cat([z, action], dim=-1))}

    def reward(self, state: State, action: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert task is not None, "MultitaskWorldModel.reward requires task"
        z = self.task_embed(state["z"], task)
        return self._reward(torch.cat([z, action], dim=-1))

    def features(self, state: State) -> torch.Tensor:
        return state["z"]

    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        """``"encoder"`` == ``self.encoder`` alone (no separate projection
        module here, unlike ``LatentConsistencyModel`` -- see class
        docstring); ``"model"`` == dynamics/reward (no termination head,
        see class docstring) plus ``task_emb``, which upstream trains in the
        same base-LR group (``3rd_party/tdmpc2/tdmpc2/tdmpc2.py:28``)."""
        return {
            "encoder": self.encoder.parameters(),
            "model": itertools.chain(
                self._dynamics.parameters(),
                self._reward.parameters(),
                self.task_emb.parameters(),
            ),
        }

    def model_loss(self, batch: dict) -> tuple[dict[str, torch.Tensor], State]:
        """Rho-weighted latent-consistency + reward soft-CE losses, task-
        conditioned at every step -- mirrors
        ``LatentConsistencyModel.model_loss`` (see that method's docstring,
        including plan item 0: the returned posterior state is live, not
        detached).

        ``batch`` keys: ``"obs"`` (``(horizon + 1, B, obs_dim)``, already
        task-padded), ``"action"`` (``(horizon, B, action_dim)``),
        ``"reward"`` (``(horizon, B, 1)``), ``"task"`` (``(B,)``), ``"next_z"``
        (``(horizon, B, latent_dim)``, the caller's already-computed no-grad
        ``encode(obs[1:], task)``, reused here instead of re-encoding).
        """
        obs, action, reward, task, next_z = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["task"],
            batch["next_z"],
        )
        horizon, batch_size = action.shape[0], action.shape[1]

        z = self.encode(obs[0], task)
        zs = torch.empty(horizon + 1, batch_size, self.latent_dim, device=z.device, dtype=z.dtype)
        zs[0] = z
        for t in range(horizon):
            z = self.step({"z": z}, action[t], task=task)["z"]
            zs[t + 1] = z
        consistency_loss = torch.zeros((), device=zs.device)
        for t in range(horizon):
            consistency_loss = consistency_loss + F.mse_loss(zs[t + 1], next_z[t]) * self.rho**t
        consistency_loss = consistency_loss / horizon

        _zs = zs[:-1]
        reward_preds = self.reward({"z": _zs}, action, task=task)
        reward_loss = torch.zeros((), device=zs.device)
        for t in range(horizon):
            reward_loss = reward_loss + soft_ce(
                reward_preds[t], reward[t], self.num_bins, self.vmin, self.vmax, self.bin_size
            ).mean() * self.rho**t
        reward_loss = reward_loss / horizon

        losses = {"consistency_loss": consistency_loss, "reward_loss": reward_loss}
        return losses, {"z": zs}
