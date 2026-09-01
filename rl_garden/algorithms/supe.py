"""SUPE (Wilcoxson et al. 2025, "Leveraging Skills from Unlabeled Prior Data
for Efficient Online Exploration"): ExPLORe's online RLPD backbone operating
over OPAL's learned skill space instead of raw actions, with a frozen OPAL
decoder mapping ``(obs, skill) -> raw action`` inside a macro-action env wrapper
(``rl_garden/envs/wrappers/skill_action_wrapper.py::SkillActionWrapper``,
applied at env-construction time in the training entrypoint -- see
``rl_garden/training/online/supe.py`` -- exactly like ``DPPO``'s
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
`_add_flat_transitions` (no intermediate H5 write).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.explore import ExPLORe
from rl_garden.buffers._dataset_common import _add_flat_transitions
from rl_garden.buffers.windowed_trajectory_dataset import WindowedTrajectoryDataset
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.networks.opal_vae import OPALVAE


def load_opal_vae(
    checkpoint_path: str,
    obs_dim: int,
    action_space: spaces.Box,
    *,
    device: str | torch.device,
) -> OPALVAE:
    """Reconstructs a frozen ``OPALVAE`` from an ``OPAL`` checkpoint.

    ``obs_dim``/``action_space`` come from the caller (the *raw*, pre-skill-
    wrapping env) rather than the checkpoint's own stored space metadata --
    that metadata (``checkpoint.py::space_metadata``) only records
    shape/dtype for compatibility checks, not the actual ``low``/``high``
    bounds ``UnsquashedGaussianActor`` needs to register as buffers. Mirrors
    ``DPPO``'s own ``bc_checkpoint`` loading (``dppo.py:313-316``): reads the
    raw checkpoint file directly rather than instantiating a full ``OPAL``
    algorithm (which would require a dummy dataset path just to read out the
    network).
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    meta = checkpoint["metadata"]["hyperparameters"]
    vae = OPALVAE(
        obs_dim, action_space, meta["skill_dim"], meta["chunk_size"],
        hidden_size=meta["hidden_size"],
        prior_hidden_dims=meta["vae_hidden_dims"],
        decoder_hidden_dims=meta["vae_hidden_dims"],
    ).to(device)
    vae.load_state_dict(checkpoint["state"]["policy"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


class SUPE(ExPLORe):
    """SUPE as a thin extension of ExPLORe. See module docstring."""

    _compatible_checkpoint_algorithms = ("SUPE",)

    def __init__(
        self,
        env: Any,
        opal_checkpoint: str,
        eval_env: Optional[Any] = None,
        **explore_kwargs: Any,
    ) -> None:
        self.opal_checkpoint = opal_checkpoint
        super().__init__(env, eval_env, **explore_kwargs)
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        self.opal_vae = load_opal_vae(
            opal_checkpoint, obs_dim, self.env.raw_action_space, device=self.device,
        )

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
                # Reward discounting zeroed past the window's first
                # termination (matches ChunkDataset.create,
                # chunk_dataset.py:139-146): `alive` is a cumulative
                # "not yet terminated within this window" mask.
                alive = 1.0 - (torch.cumsum(batch.done_window, dim=-1) > 0).float()
                discounted = batch.reward_window * discounts[None, :]
                seq_rewards = torch.cat(
                    [discounted[:, :1], discounted[:, 1:] * alive[:, :-1]], dim=-1,
                )
                rewards = seq_rewards.sum(dim=-1)
                dones = (batch.done_window.sum(dim=-1) > 0).float()
                loaded += _add_flat_transitions(
                    self.offline_replay_buffer,
                    batch.obs_window[:, 0],
                    batch.next_obs,
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
