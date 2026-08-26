"""Vision-conditioned Diffusion BC pretraining.

A standalone sibling of ``DiffusionBC`` (``rl_garden/algorithms/diffusion_bc.py``),
not a modification of it -- ``DiffusionBC`` is state-only by design and its
checkpoints (``ema_net_state_dict``) are loaded directly into ``DPPO``'s
``actor``/``actor_ft`` (``dppo_policy.py::load_actor_weights``); that
contract stays untouched. This class trains a ``VisionDiffusionPolicy``
(``policies/vision_diffusion_policy.py``, Dict observations via
``CombinedExtractor``) the same way ``DiffusionBC`` trains a
``DiffusionPolicy`` -- same step-based training loop, same hand-rolled
linear-blend EMA -- just with a Dict-only observation space instead of Box.

Dataset loading is unchanged: ``load_h5_dataset_as_chunks``
(``buffers/chunked_dataset.py``) is already observation-space-generic (Box
or Dict), so no dataset-loader changes were needed for this class either.
"""
from __future__ import annotations

import copy
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
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
from rl_garden.policies.vision_diffusion_policy import VisionDiffusionPolicy


class _EMA:
    """Matches ``3rd_party/dppo/agent/pretrain/train_agent.py::EMA`` exactly
    (duplicated from ``DiffusionBC``'s identical helper rather than imported,
    so this file has no dependency on ``diffusion_bc.py``'s internals)."""

    def __init__(self, decay: float) -> None:
        self.decay = decay

    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


class VisionDiffusionBC(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("VisionDiffusionBC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        *,
        horizon_steps: int = 4,
        cond_steps: int = 1,
        denoising_steps: int = 20,
        mlp_dims: Optional[Sequence[int]] = None,
        activation_fn: Optional[Activation] = "relu",
        residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
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
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        ema_start_step: Optional[int] = None,
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
            raise TypeError(
                "VisionDiffusionBC requires a Dict observation space; use "
                "DiffusionBC for Box (state-only) observations."
            )
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.dataset_path = dataset_path
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.denoising_steps = denoising_steps
        self.mlp_dims = list(mlp_dims) if mlp_dims is not None else [512, 512, 512]
        self.activation_fn = activation_fn
        self.residual_style = residual_style
        self.time_dim = time_dim
        self.kernel_init = kernel_init
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
        self._state_key = state_key if state_key is not None else "state"
        self._use_proprio = use_proprio if use_proprio is not None else True
        self._proprio_latent_dim = proprio_latent_dim if proprio_latent_dim is not None else 64
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
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self._ema_start_step_arg = ema_start_step
        self.num_traj = num_traj

        self._setup_model()
        self._load_dataset()
        if self._ema_start_step_arg is None:
            # Same epoch->step warmup conversion as DiffusionBC -- see that
            # class's docstring for the reasoning.
            steps_per_epoch = max(1, self._dataset_size // self.batch_size)
            self.ema_start_step = 20 * steps_per_epoch
        else:
            self.ema_start_step = self._ema_start_step_arg

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_steps": self.horizon_steps,
            "cond_steps": self.cond_steps,
            "denoising_steps": self.denoising_steps,
            "mlp_dims": self.mlp_dims,
            "activation_fn": self.activation_fn,
            "residual_style": self.residual_style,
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "use_proprio": self._use_proprio,
            "proprio_latent_dim": self._proprio_latent_dim,
            "image_fusion_mode": self._image_fusion_mode,
            "enable_stacking": self._enable_stacking,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ],
            "ema_net_state_dict": self.ema_policy.net.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        for sched, sched_state in zip(
            self._lr_schedulers, state.get("lr_scheduler_states", [])
        ):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)
        ema_net_state_dict = state.get("ema_net_state_dict")
        if ema_net_state_dict is not None:
            self.ema_policy.net.load_state_dict(ema_net_state_dict)

    # --- model / data setup ---

    def _setup_model(self) -> None:
        features_extractor = CombinedExtractor(
            observation_space=self.env.single_observation_space,
            image_keys=self._image_keys,
            state_key=self._state_key,
            image_encoder_factory=self._image_encoder_factory,
            proprio_latent_dim=self._proprio_latent_dim,
            use_proprio=self._use_proprio,
            fusion_mode=self._image_fusion_mode,
            enable_stacking=self._enable_stacking,
        )
        self.policy = VisionDiffusionPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            denoising_steps=self.denoising_steps,
            mlp_dims=self.mlp_dims,
            activation_fn=self.activation_fn,
            residual_style=self.residual_style,
            time_dim=self.time_dim,
            kernel_init=self.kernel_init,
            denoised_clip_value=self.denoised_clip_value,
            randn_clip_value=self.randn_clip_value,
            final_action_clip_value=self.final_action_clip_value,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
        ).to(self.device)
        self.ema_policy = copy.deepcopy(self.policy)
        for p in self.ema_policy.parameters():
            p.requires_grad_(False)

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
        self._ema = _EMA(self.ema_decay)

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
                "VisionDiffusionBC requires a Dict-shaped H5 dataset (nested "
                "obs/<key> groups); got a flat Box-shaped obs array. Use "
                "DiffusionBC for Box datasets."
            )
        self._obs_history = obs_history
        self._action_chunks = action_chunks
        self._dataset_size = action_chunks.shape[0]

    # --- training ---

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        loss_sum = 0.0
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            idx = torch.randint(
                0, self._dataset_size, (self.batch_size,), device=self._action_chunks.device
            )
            obs_history = index_obs(self._obs_history, idx)
            action_chunk = self._action_chunks[idx]

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss = self.policy.loss(obs_history, action_chunk)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            for sched in self._lr_schedulers:
                if sched is not None:
                    sched.step()

            if self._global_update % self.ema_update_every == 0:
                if self._global_update < self.ema_start_step:
                    self.ema_policy.load_state_dict(self.policy.state_dict())
                else:
                    self._ema.update(self.ema_policy, self.policy)

            loss_sum += float(loss.detach().item())

        return {"loss": loss_sum / gradient_steps}
