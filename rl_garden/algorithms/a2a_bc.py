"""A2A ("Action-to-Action") flow-matching BC pretraining.

Standalone sibling of ``VisionDiffusionBC`` (``rl_garden/algorithms/vision_diffusion_bc.py``),
not a modification of it -- same overall shape (``OfflineRLAlgorithm``,
dataset loaded directly via ``load_h5_dataset_as_chunks``, no replay buffer,
step-based training loop), but with **no EMA**: A2A's reference has no EMA
target network, unlike the diffusion-BC lineage this class's shape is
otherwise copied from.

Dataset loading is unchanged: ``load_h5_dataset_as_chunks``
(``buffers/chunked_dataset.py``) is already observation-space-generic (Box or
Dict); A2A only ever uses the Dict path (vision conditioning is mandatory).
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.chunked_dataset import load_h5_dataset_as_chunks
from rl_garden.common.logger import Logger
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.networks import Activation, KernelInit
from rl_garden.policies.a2a_policy import A2APolicy


class A2ABC(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("A2ABC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        *,
        horizon_steps: int = 8,
        cond_steps: int = 8,
        latent_dim: int = 512,
        cnn_num_layers: int = 3,
        cnn_hidden_channels: int = 512,
        cnn_kernel_size: int = 5,
        cnn_activation_fn: Optional[Activation] = "relu",
        decoder_net_arch: Optional[Sequence[int]] = None,
        decoder_activation_fn: Optional[Activation] = None,
        decoder_kernel_init: Optional[KernelInit] = None,
        flow_hidden_dims: Optional[Sequence[int]] = None,
        flow_use_layer_norm: bool = False,
        flow_kernel_init: Optional[KernelInit] = None,
        flow_activation_fn: Optional[Activation] = None,
        num_sampling_steps: int = 6,
        consistency_weight: float = 1.0,
        enc_recon_weight: float = 0.5,
        flow_recon_weight: float = 0.5,
        enc_contrastive_weight: float = 0.0,
        flow_contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        actor_lr: float = 1e-3,
        weight_decay: float = 1e-6,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        batch_size: int = 128,
        num_traj: Optional[int] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=1,
            buffer_device="cpu",
            batch_size=batch_size,
            gamma=0.99,
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
        if not isinstance(self.env.single_observation_space, spaces.Dict):
            raise TypeError("A2ABC requires a Dict observation space.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self._state_key = state_key if state_key is not None else "state"
        if self._state_key not in self.env.single_observation_space.spaces:
            raise ValueError(
                f"A2ABC requires state_key={self._state_key!r} in the observation "
                "space -- the state-history window is the flow's source (x_0), "
                "not optional."
            )
        self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
        if not any(
            k in self.env.single_observation_space.spaces for k in self._image_keys
        ):
            raise ValueError(
                "A2ABC requires at least one resolved image key for vision "
                f"conditioning (got image_keys={self._image_keys!r})."
            )

        self.dataset_path = dataset_path
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.latent_dim = latent_dim
        self.cnn_num_layers = cnn_num_layers
        self.cnn_hidden_channels = cnn_hidden_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_activation_fn = cnn_activation_fn
        self.decoder_net_arch: list[int] = (
            list(decoder_net_arch) if decoder_net_arch is not None else [512, 512, 512, 512]
        )
        self.decoder_activation_fn = decoder_activation_fn
        self.decoder_kernel_init = decoder_kernel_init
        self.flow_hidden_dims: list[int] = (
            list(flow_hidden_dims) if flow_hidden_dims is not None else [512, 512, 512, 512]
        )
        self.flow_use_layer_norm = flow_use_layer_norm
        self.flow_kernel_init = flow_kernel_init
        self.flow_activation_fn = flow_activation_fn
        self.num_sampling_steps = num_sampling_steps
        self.consistency_weight = consistency_weight
        self.enc_recon_weight = enc_recon_weight
        self.flow_recon_weight = flow_recon_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_fusion_mode = (
            image_fusion_mode if image_fusion_mode is not None else "stack_channels"
        )
        self._enable_stacking = enable_stacking if enable_stacking is not None else False
        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.num_traj = num_traj

        self._setup_model()
        self._load_dataset()

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_steps": self.horizon_steps,
            "cond_steps": self.cond_steps,
            "latent_dim": self.latent_dim,
            "cnn_num_layers": self.cnn_num_layers,
            "cnn_hidden_channels": self.cnn_hidden_channels,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_activation_fn": self.cnn_activation_fn,
            "decoder_net_arch": self.decoder_net_arch,
            "flow_hidden_dims": self.flow_hidden_dims,
            "num_sampling_steps": self.num_sampling_steps,
            "consistency_weight": self.consistency_weight,
            "enc_recon_weight": self.enc_recon_weight,
            "flow_recon_weight": self.flow_recon_weight,
            "enc_contrastive_weight": self.enc_contrastive_weight,
            "flow_contrastive_weight": self.flow_contrastive_weight,
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "image_fusion_mode": self._image_fusion_mode,
            "enable_stacking": self._enable_stacking,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ]
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        for sched, sched_state in zip(
            self._lr_schedulers, state.get("lr_scheduler_states", [])
        ):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)

    # --- model / data setup ---

    def _setup_model(self) -> None:
        features_extractor = CombinedExtractor(
            observation_space=self.env.single_observation_space,
            image_keys=self._image_keys,
            state_key=self._state_key,
            image_encoder_factory=self._image_encoder_factory,
            use_proprio=False,
            fusion_mode=self._image_fusion_mode,
            enable_stacking=self._enable_stacking,
        )
        self.policy = A2APolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            state_key=self._state_key,
            latent_dim=self.latent_dim,
            cnn_num_layers=self.cnn_num_layers,
            cnn_hidden_channels=self.cnn_hidden_channels,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_activation_fn=self.cnn_activation_fn,
            decoder_net_arch=self.decoder_net_arch,
            decoder_activation_fn=self.decoder_activation_fn,
            decoder_kernel_init=self.decoder_kernel_init,
            flow_hidden_dims=self.flow_hidden_dims,
            flow_use_layer_norm=self.flow_use_layer_norm,
            flow_kernel_init=self.flow_kernel_init,
            flow_activation_fn=self.flow_activation_fn,
            num_sampling_steps=self.num_sampling_steps,
            consistency_weight=self.consistency_weight,
            enc_recon_weight=self.enc_recon_weight,
            flow_recon_weight=self.flow_recon_weight,
            enc_contrastive_weight=self.enc_contrastive_weight,
            flow_contrastive_weight=self.flow_contrastive_weight,
            contrastive_temperature=self.contrastive_temperature,
        ).to(self.device)

        self.actor_optimizer = make_optimizer(
            list(self.policy.parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self._lr_schedulers = [
            make_lr_scheduler(
                self.actor_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
        ]

    def _load_dataset(self) -> None:
        obs_history, action_chunks = load_h5_dataset_as_chunks(
            self.dataset_path,
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            device=self.device,
            num_traj=self.num_traj,
        )
        if not isinstance(obs_history, dict):
            raise TypeError(
                "A2ABC requires a Dict-shaped H5 dataset (nested obs/<key> "
                "groups); got a flat Box-shaped obs array."
            )
        self._obs_history = obs_history
        self._action_chunks = action_chunks
        self._dataset_size = action_chunks.shape[0]

    # --- training ---

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            idx = torch.randint(
                0, self._dataset_size, (self.batch_size,), device=self._action_chunks.device
            )
            obs_history = index_obs(self._obs_history, idx)
            action_chunk = self._action_chunks[idx]

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.policy.loss_with_metrics(obs_history, action_chunk)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            for sched in self._lr_schedulers:
                if sched is not None:
                    sched.step()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {key: value / gradient_steps for key, value in metrics_sum.items()}
