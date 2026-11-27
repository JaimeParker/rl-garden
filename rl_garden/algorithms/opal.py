"""OPAL (Ajay et al. 2021, "OPAL: Offline Primitive Discovery for
Accelerating Offline Reinforcement Learning"): pretrains a skill-space VAE
(``OPALVAE``, ``rl_garden/networks/opal_vae.py``) over fixed-length action
chunks, producing a learned skill embedding and a frozen low-level decoder
that a downstream high-level policy can act through.

Standalone offline algorithm (``OfflineRLAlgorithm``), self-managed dataset
via ``WindowedTrajectoryDataset`` (``rl_garden/buffers/windowed_trajectory_dataset.py``)
-- no replay buffer, mirrors ``HILP``/``A2ABC``'s reasoning for the same
shape. State-based (Box observations) only.

Ported from ``SUPE/supe/pretraining/opal.py`` (read in full). SUPE's online
meta-policy (a plain SAC/RLPD actor-critic over this VAE's skill space, plus
a skill-macro-action env wrapper) is implemented separately as
``rl_garden/algorithms/supe.py``, which consumes an ``OPAL`` checkpoint
produced by this class.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.windowed_trajectory_dataset import WindowedTrajectoryDataset
from rl_garden.common.logger import Logger
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.opal import OPALParams
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.opal_vae import OPALVAE
from rl_garden.observations import ObservationContractError, ObsGroups


class OPAL(OfflineRLAlgorithm):
    """OPAL: pretrains a skill-space VAE over raw, flat-vector observations.

    ``OPALVAE`` (``rl_garden/networks/opal_vae.py``) is a state-only
    reconstruction model with no ``BaseFeaturesExtractor``-based policy of
    its own -- unlike SAC/BC, it never calls an encoder's ``forward()`` in
    its training loop, it just needs ``obs_dim``. Observation encoding is
    still resolved through the shared ``ObservationEncoderMixin`` (for
    ``obs_dim`` and API/checkpoint-metadata uniformity with the rest of the
    codebase), but only the state-only schema is actually supported --
    image observations are rejected with a clear error rather than silently
    fed raw into the VAE (see the reference's own "state-obs path only"
    note in ``opal_vae.py``).
    """

    params_class = OPALParams

    _compatible_checkpoint_algorithms = ("OPAL",)
    # There is no critic encoder, so "separate" does not exist.
    encoder_sharing_choices = ("shared_critic_grad", "shared")
    checkpoint_eval_unsupported_reason = (
        "it is a skill-VAE pretraining stage with no policy that acts in an environment"
    )

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        *,
        batch_size: int = 256,
        num_traj: Optional[int] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
        params: Optional[OPALParams] = None,
        **overrides,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        super().__init__(
            env=env,
            buffer_size=1,
            buffer_device="cpu",
            batch_size=batch_size,
            offline_sampling="with_replace",
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=0,
            eval_env=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=False,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._bind_params(params)
        self.dataset_path = dataset_path

        self.vae_hidden_dims = tuple(params.vae_hidden_dims)

        self.num_traj = num_traj
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups

        self._dataset = WindowedTrajectoryDataset(
            dataset_path, chunk_size=params.chunk_size, device=self.device, num_traj=num_traj,
        )
        self._setup_model()

    def _setup_model(self) -> None:
        self._resolve_observation_encoders(self.env.single_observation_space)
        if self.observation_encoders.schema.has_images:
            raise ObservationContractError(
                "OPAL only supports state observations; got a schema with "
                f"image keys {self.observation_encoders.schema.image_keys}."
            )
        obs_dim = self.observation_encoders.actor.features_dim
        action_space = self.env.single_action_space

        self.policy = OPALVAE(
            obs_dim, action_space, self.skill_dim, self.chunk_size,
            hidden_size=self.hidden_size,
            prior_hidden_dims=self.vae_hidden_dims,
            decoder_hidden_dims=self.vae_hidden_dims,
        ).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            batch = self._dataset.sample(self.batch_size)
            losses = self.policy.loss(batch.obs_window, batch.action_window, kl_coef=self.kl_coef)

            self.vae_optimizer.zero_grad(set_to_none=True)
            losses["vae_loss"].backward()
            self.vae_optimizer.step()

            for key, value in losses.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value.detach().item())

        del compute_info
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("vae_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
        }
