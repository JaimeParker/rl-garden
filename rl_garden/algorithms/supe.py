"""SUPE (Wilcoxson et al. 2025, "Leveraging Skills from Unlabeled Prior Data
for Efficient Online Exploration"): ExPLORe's online RLPD backbone operating
over OPAL's learned skill space instead of raw actions, with a frozen OPAL
decoder mapping ``(obs, skill) -> raw action`` inside a macro-action env wrapper
(``rl_garden/envs/wrappers/skill_action_wrapper.py::SkillActionWrapper``,
applied at env-construction time in the training entrypoint -- see
``rl_garden/training/online/entrypoints/supe.py`` -- exactly like ``DPPO``'s
``ActionChunkWrapper`` wiring, not inside this class).

``SUPE`` adds no new online-training logic: `RewardMaskRelabeler`/`RNDBonus`
(``ExPLORe``) are confirmed dimension-generic and need zero changes to
operate on ``(obs, skill)`` instead of ``(obs, action)`` once
`SkillActionWrapper` republishes the env's `action_space` as `Box(skill_dim,)`
-- so `SUPE` inherits `_sample_train_batch`/`_relabel`/RND verbatim from
`ExPLORe`. This resolves `ExPLORe._sample_train_batch`'s own "reconcile once
SUPE is a second consumer" comment by direct subclassing rather than
extracting a new parent hook.

The one genuinely new piece is offline skill-relabeling: SUPE's offline
prior buffer must hold ``(obs, skill, reward, next_obs)`` tuples, not raw
``(obs, action)`` ones, so it cannot reuse `PriorDataReplayMixin`'s plain
H5 loader. `load_skill_relabeled_offline_buffer` sources from
`WindowedTrajectoryDataset` and OPAL's frozen `.encode()`,
reproducing upstream `ChunkDataset.create`'s exact aggregation
(``SUPE/supe/data/chunk_dataset.py:130-155``): per-window
discounted reward sum zeroed past the window's first termination, OR-of-dones
over the window, one relabeled row per valid window (a full, deterministic
pass -- not a random subsample), inserted directly as tensors via
`add_flat_transitions` (no intermediate H5 write).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms.explore import ExPLORe
from rl_garden.buffers.dataset_common import add_flat_transitions, match_obs_to_buffer
from rl_garden.buffers.windowed_trajectory_dataset import WindowedTrajectoryDataset
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.networks.opal_vae import OPALVAE


def build_frozen_opal_vae(
    hyperparameters: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    obs_dim: int,
    action_space: spaces.Box,
    *,
    device: str | torch.device,
) -> OPALVAE:
    """``obs_dim``/``action_space`` come from the caller (the *raw*, pre-skill-
    wrapping env): a checkpoint's space metadata (``space_metadata``) does not
    record the ``low``/``high`` bounds ``UnsquashedGaussianActor`` registers as
    buffers."""
    vae = OPALVAE(
        obs_dim, action_space, hyperparameters["skill_dim"], hyperparameters["chunk_size"],
        hidden_size=hyperparameters["hidden_size"],
        prior_hidden_dims=hyperparameters["vae_hidden_dims"],
        decoder_hidden_dims=hyperparameters["vae_hidden_dims"],
    ).to(device)
    vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_opal_vae(
    checkpoint_path: str,
    obs_dim: int,
    action_space: spaces.Box,
    *,
    device: str | torch.device,
) -> OPALVAE:
    """The frozen ``OPALVAE`` of an ``OPAL`` checkpoint file, read directly
    rather than through a full ``OPAL`` (which needs a dataset path)."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"OPAL checkpoint {checkpoint_path!r} does not exist. SUPE needs it to train from "
            "scratch, and to load a SUPE checkpoint that does not embed its OPAL VAE "
            "(--opal-checkpoint)."
        )
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    return build_frozen_opal_vae(
        checkpoint["metadata"]["hyperparameters"],
        checkpoint["state"]["policy"],
        obs_dim,
        action_space,
        device=device,
    )


def _linear_widths(module: nn.Module) -> list[int]:
    return [layer.out_features for layer in module.modules() if isinstance(layer, nn.Linear)]


def opal_vae_record(vae: OPALVAE) -> dict[str, Any]:
    """What a ``SUPE`` checkpoint embeds to rebuild ``vae``
    (``build_frozen_opal_vae``'s first two arguments)."""
    return {
        "hyperparameters": {
            "skill_dim": vae.skill_dim,
            "chunk_size": vae.chunk_size,
            "hidden_size": _linear_widths(vae.obs_mlp)[0],
            "vae_hidden_dims": _linear_widths(vae.prior_trunk),
        },
        "state_dict": vae.state_dict(),
    }


def embedded_opal_vae(checkpoint: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The ``opal_vae_record`` in a ``SUPE`` checkpoint, ``None`` without one."""
    if checkpoint is None:
        return None
    return checkpoint["state"].get("extra", {}).get("opal_vae")


class SUPE(ExPLORe):
    """SUPE as a thin extension of ExPLORe. See module docstring.

    ``opal_vae`` is the VAE whose decoder ``env`` (a ``SkillActionWrapper``)
    acts through; without it the VAE is read from ``opal_checkpoint``. A
    checkpoint embeds the VAE, and loading one restores it in place, so an
    ``env`` built on this agent's ``opal_vae.decoder`` follows."""

    _compatible_checkpoint_algorithms = ("SUPE",)

    def __init__(
        self,
        env: Any,
        opal_checkpoint: str,
        eval_env: Optional[Any] = None,
        *,
        opal_vae: Optional[OPALVAE] = None,
        **explore_kwargs: Any,
    ) -> None:
        self.opal_checkpoint = opal_checkpoint
        super().__init__(env, eval_env, **explore_kwargs)
        if opal_vae is None:
            # State-only by construction (ExPLORe.__init__ rejects images), so
            # the Dict contract's single "state" key is always present.
            obs_dim = int(np.prod(self.env.single_observation_space["state"].shape))
            opal_vae = load_opal_vae(
                opal_checkpoint, obs_dim, self.env.raw_action_space, device=self.device,
            )
        self.opal_vae = opal_vae.to(self.device)

    def load_skill_relabeled_offline_buffer(
        self,
        dataset_path: str,
        *,
        buffer_size: int,
        chunk_size: int,
        discount: float,
        offline_data_ratio: float,
        relabel_batch_size: int = 4096,
        num_traj: Optional[int] = None,
    ) -> int:
        """Mirrors ``load_offline_replay_buffer``
        (``rl_garden/buffers/prior_data_replay.py:86-127``)'s shape,
        sourcing from skill-relabeled windows instead of raw H5 actions."""
        if not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(
                f"offline_data_ratio must be in [0, 1], got {offline_data_ratio}."
            )
        windowed = WindowedTrajectoryDataset(
            dataset_path, chunk_size=chunk_size, device=self.device, num_traj=num_traj,
        )
        self.offline_replay_buffer = self._build_prior_data_buffer(int(buffer_size))
        discounts = discount ** torch.arange(chunk_size, device=self.device, dtype=torch.float32)

        loaded = 0
        with torch.no_grad():
            for batch in windowed.all_windows(relabel_batch_size):
                skills = self.opal_vae.encode(batch.obs_window, batch.action_window)
                # Reward discounting zeroed past the window's first *true*
                # termination (matches ChunkDataset.create's `masks` field,
                # chunk_dataset.py:139-146, d4rl_datasets.py:44 -- termination
                # only, excluding truncation): `alive` is a cumulative "not
                # yet terminated within this window" mask. The aggregated
                # `dones` output below intentionally uses the broader
                # done_window (terminated OR truncated) instead, matching
                # upstream's separate `dones` field (d4rl_datasets.py:40).
                alive = 1.0 - (torch.cumsum(batch.terminated_window, dim=-1) > 0).float()
                discounted = batch.reward_window * discounts[None, :]
                seq_rewards = torch.cat(
                    [discounted[:, :1], discounted[:, 1:] * alive[:, :-1]], dim=-1,
                )
                rewards = seq_rewards.sum(dim=-1)
                dones = (batch.done_window.sum(dim=-1) > 0).float()
                window_obs, next_obs = match_obs_to_buffer(
                    self.offline_replay_buffer, batch.obs_window[:, 0], batch.next_obs
                )
                loaded += add_flat_transitions(
                    self.offline_replay_buffer,
                    window_obs,
                    next_obs,
                    skills,
                    rewards,
                    dones,
                )

        self.offline_data_ratio = float(offline_data_ratio)
        if self.offline_relabel_type == "min":
            full = self.offline_replay_buffer.sample(len(self.offline_replay_buffer))
            self._offline_min_reward = full.rewards.min()
        if self.logger is not None:
            self.logger.add_summary("prior_data/offline_loaded_transitions", loaded)
            self.logger.add_summary(
                "prior_data/offline_data_ratio", self.offline_data_ratio
            )
            self.logger.add_summary("prior_data/offline_buffer_size", int(buffer_size))
        return loaded

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {**super()._checkpoint_metadata(), "opal_checkpoint": self.opal_checkpoint}

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {**super()._extra_checkpoint_state(), "opal_vae": opal_vae_record(self.opal_vae)}

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        if "opal_vae" in state:
            self.opal_vae.load_state_dict(state["opal_vae"]["state_dict"])
