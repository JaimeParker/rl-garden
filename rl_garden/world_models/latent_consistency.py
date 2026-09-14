"""``LatentConsistencyModel``: TD-MPC2's implicit world model, ported from
``3rd_party/tdmpc2/tdmpc2/common/world_model.py`` and split out of this
repo's former single-class ``WorldModel`` that used to live in the
now-deleted ``rl_garden.algorithms.tdmpc2`` package, per the model-based-base
plan part 1.2/1.3 -- this class keeps
only the *model* half (encoder, latent projection, dynamics, reward, optional
termination head); the actor (policy prior) and critic (Q-ensemble) that used
to live on the same class now belong to
``rl_garden.policies.tdmpc2_policy.TDMPC2Policy``, which owns this model as
its ``actor_extractor``.

The encoder half of upstream's own ``common.layers.enc(cfg)`` is replaced by
an rl-garden ``BaseFeaturesExtractor`` (``FlattenExtractor`` for state obs,
``CombinedExtractor`` for pixel+state obs) so this port reuses the project's
existing encoder infrastructure instead of TD-MPC2's own conv/MLP encoder;
the SimNorm-projection half is kept, mapping the extractor's flat features
onto the fixed ``latent_dim`` the dynamics/reward heads expect.

Upstream's ``Ensemble`` (used for the Q-heads now living on the policy) uses
``torch.vmap`` over ``tensordict.nn.TensorDictParams`` for a single fused
forward pass across the ensemble -- pulling in ``tensordict``/functorch
machinery this port deliberately avoids (a "no new dependencies" scoping
decision); see ``rl_garden.networks.q_ensemble.QEnsemble``'s own docstring.

TD-MPC2 has no recurrent state to carry across timesteps: every latent is
fully determined by re-encoding the current real observation (or rolling the
previous latent through the dynamics head with no observation, for
planning/imagination). ``observe()`` therefore ignores its incoming
``state``/``action``/``is_first`` entirely and just wraps a fresh embedding
into ``State`` -- unlike a recurrent model (DreamerV3's RSSM), which folds
the previous state and action into the new one.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.normed_mlp import SimNorm, mlp
from rl_garden.networks.twohot import soft_ce
from rl_garden.world_models.base import State, WorldModel


class LatentConsistencyModel(WorldModel):
    def __init__(
        self,
        encoder: BaseFeaturesExtractor,
        action_dim: int,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        episodic: bool = False,
        rho: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_size = (vmax - vmin) / (num_bins - 1) if num_bins > 1 else 0.0
        self.episodic = episodic
        self.rho = rho

        self._latent_proj = nn.Sequential(
            nn.Linear(encoder.features_dim, latent_dim), SimNorm(simnorm_dim)
        )
        self._dynamics = mlp(
            latent_dim + action_dim, 2 * [mlp_dim], latent_dim, act=SimNorm(simnorm_dim)
        )
        self._reward = mlp(latent_dim + action_dim, 2 * [mlp_dim], max(num_bins, 1))
        self._termination = mlp(latent_dim, 2 * [mlp_dim], 1) if episodic else None
        # NOTE: weight init deliberately NOT run here -- see apply_init()'s
        # docstring for why it's deferred to (and orchestrated by)
        # TDMPC2Policy.__init__.

    # ------------------------------------------------------------------
    # Initialization (deferred -- see docstring)
    # ------------------------------------------------------------------

    def apply_init(self) -> None:
        """Initializes this model's own modules (encoder + latent_proj +
        dynamics + reward [+ termination]) via TD-MPC2's ``trunc_normal_``
        init, with the reward head's last layer zeroed.

        Deliberately **not** called from ``__init__``: upstream TD-MPC2 runs
        one combined ``self.apply(weight_init)`` pass over a single class
        holding the model *and* the actor/critic/target-critic together, so
        every ``nn.Linear``'s default-init draws (at construction) and every
        ``trunc_normal_`` draw (at ``apply(weight_init)``) interleave in one
        fixed order relative to the shared ``torch`` RNG stream. Splitting
        actor/critic out to ``TDMPC2Policy`` (plan 1.3) means that order can
        now only be reproduced if construction of *every* module (this
        model's, then the policy's actor, critic, critic-target) happens
        first, and *then* every module's init pass runs in that same order
        -- which requires this model's own init to be triggered by
        ``TDMPC2Policy.__init__`` at the right point instead of eagerly in
        this class's own ``__init__``. See ``TDMPC2Policy.__init__``'s
        module-level ordering comment for the exact sequence.
        """
        from rl_garden.networks import init as net_init

        self.apply(net_init.weight_init)
        net_init.zero_([self._reward[-1].weight])

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def encode(self, obs: Obs) -> torch.Tensor:
        features = self.encoder.extract(obs)
        return self._latent_proj(features)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return {"z": torch.zeros(batch_size, self.latent_dim, device=device)}

    def observe(
        self,
        state: Optional[State],
        action: Optional[torch.Tensor],
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> State:
        del state, action, is_first  # no recurrent carry, see module docstring.
        return {"z": embed}

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        del sample  # deterministic dynamics head, nothing stochastic to sample.
        return {"z": self._dynamics(torch.cat([state["z"], action], dim=-1))}

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        """Two-hot bin logits, not a scalar -- decode with
        ``rl_garden.networks.twohot.two_hot_inv(out, self.num_bins, self.vmin,
        self.vmax)``."""
        return self._reward(torch.cat([state["z"], action], dim=-1))

    def continue_(self, state: State) -> Optional[torch.Tensor]:
        """Probability of continuing past ``state``. ``None`` when
        ``episodic=False`` (no termination head is built at all -- callers
        must inject a ``termination_fn``, see ``WorldModel.continue_``)."""
        if self._termination is None:
            return None
        return torch.sigmoid(-self._termination(state["z"]))  # 1 - sigmoid(logit)

    def features(self, state: State) -> torch.Tensor:
        return state["z"]

    def parameter_groups(self) -> dict[str, Iterator[nn.Parameter]]:
        """``"encoder"`` == upstream's single ``_encoder`` module (module
        docstring): this port splits it into ``self.encoder`` (the
        rl-garden ``BaseFeaturesExtractor``) + ``self._latent_proj`` (the
        SimNorm projection upstream's own ``enc(cfg)`` ends with) -- both
        stay in this group so TD-MPC2's ``enc_lr_scale`` continues to apply
        to exactly the same parameters as upstream. ``"model"`` is
        everything else this model owns (dynamics/reward[/termination]);
        the caller adds the critic on top at the base ``lr``, matching
        upstream's single un-scaled optimizer group."""
        model_params = list(self._dynamics.parameters()) + list(self._reward.parameters())
        if self._termination is not None:
            model_params += list(self._termination.parameters())
        return {
            "encoder": itertools.chain(self.encoder.parameters(), self._latent_proj.parameters()),
            "model": iter(model_params),
        }

    def model_loss(self, batch: dict) -> tuple[dict[str, torch.Tensor], State]:
        """Rho-weighted latent-consistency + reward soft-CE losses over a
        ``(horizon + 1, B)`` obs window, moved (same numerics) from this
        repo's former ``TDMPC2._gradient_step``.

        ``batch`` keys: ``"obs"`` (``(horizon + 1, B, ...)`` windowed
        observations), ``"action"`` (``(horizon, B, action_dim)``),
        ``"reward"`` (``(horizon, B, 1)``), ``"next_z"`` (``(horizon, B,
        latent_dim)``, the caller's already-computed no-grad
        ``encode(obs[1:])`` -- reused here instead of re-encoding, since the
        caller needs it anyway for its own TD-target), and -- only read
        when ``self.episodic`` -- ``"terminated"`` (``(horizon, B, 1)``
        float).

        Returns ``({"consistency_loss", "reward_loss", "termination_loss"},
        {"z": zs})`` where ``zs`` is the **live** (graph-attached) dynamics
        rollout ``zs[0] = encode(obs[0])``, ``zs[t + 1] = step({"z": zs[t]},
        action[t])["z"]`` -- see ``WorldModel.model_loss``'s docstring (plan
        item 0). The caller's critic value loss reads this live tensor
        directly so its gradient reaches this model's dynamics/encoder
        exactly like upstream's single shared graph; the caller detaches it
        itself for the actor update, matching upstream's ``update_pi(zs
        .detach(), ...)`` call site.
        """
        obs, action, reward, next_z = batch["obs"], batch["action"], batch["reward"], batch["next_z"]
        horizon, batch_size = action.shape[0], action.shape[1]

        z = self.encode(index_obs(obs, 0))
        zs = torch.empty(horizon + 1, batch_size, self.latent_dim, device=z.device, dtype=z.dtype)
        zs[0] = z
        for t in range(horizon):
            z = self.step({"z": z}, action[t])["z"]
            zs[t + 1] = z
        consistency_loss = torch.zeros((), device=zs.device)
        for t in range(horizon):
            consistency_loss = consistency_loss + F.mse_loss(zs[t + 1], next_z[t]) * self.rho**t
        consistency_loss = consistency_loss / horizon

        _zs = zs[:-1]
        reward_preds = self.reward({"z": _zs}, action)
        reward_loss = torch.zeros((), device=zs.device)
        for t in range(horizon):
            reward_loss = reward_loss + soft_ce(
                reward_preds[t], reward[t], self.num_bins, self.vmin, self.vmax, self.bin_size
            ).mean() * self.rho**t
        reward_loss = reward_loss / horizon

        if self.episodic:
            termination_pred = self._termination(zs[1:])
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, batch["terminated"])
        else:
            termination_loss = torch.zeros((), device=zs.device)

        losses = {
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "termination_loss": termination_loss,
        }
        return losses, {"z": zs}
