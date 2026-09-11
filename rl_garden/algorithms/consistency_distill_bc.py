"""ConsistencyDistillBC: fully offline Latent-Consistency-Model (LCM) style
distillation of a frozen ``DiffusionBC`` teacher into a one/few-step
consistency-model student, trained purely on the offline demo dataset --
no environment interaction, no RL, no rollout buffer.

Reuses the exact boundary-condition/DDIM-step LCM math
``DiffusionCMDistillOnline`` (``rl_garden/algorithms/diffusion_cm_distill.py``)
already ported faithfully from ``3rd_party/RL-100``'s ``compute_ddim2cm_loss``
(``_scalings_for_boundary_conditions``/``_gather_coef`` imported directly,
``_predicted_x0``/``_ddim_step`` reimplemented here against this class's own
frozen-teacher schedule buffers rather than coupling to ``DPPOPolicy``). The
online version distills against a just-collected on-policy rollout's clean
trajectory; this offline version distills against real dataset action
chunks instead -- the same "clean latent" role, translated from "just
produced by rollout" to "just loaded from demonstrations".

``self.policy`` stays the frozen multi-step teacher (a ``DiffusionPolicy``
loaded from ``bc_checkpoint``, matching ``DiffusionCMDistillOnline``'s own
"teacher stays ``self.policy``" convention) -- the trainable one-step
student lives in ``self.cm_student``/``self.cm_target``, separate attributes
meant for deployment-time export, not wired into ``predict()``.

The student/target networks must match the teacher's own denoiser class and
schedule exactly (``net_cls``/``net_kwargs``/``horizon_steps``/``cond_steps``/
``denoising_steps``, plus ``mlp_dims``/``activation_fn``/``residual_style``
when ``net_cls is DiffusionMLP``) since they are warm-started from the
teacher's ``ema_net_state_dict``. A ``net_cls`` mismatch alone would fail
loudly via a state-dict shape error, but the other fields (e.g.
``denoising_steps``, ``activation_fn``) carry no parameters and would
otherwise load silently with the wrong schedule/nonlinearity -- so
``_setup_model`` validates all of them explicitly against the checkpoint's
own recorded ``hyperparameters`` before loading, and raises ``ValueError``
naming every mismatched field. Fields absent from an older checkpoint's
recorded hyperparameters (e.g. ``net_cls``, added to
``DiffusionBC._checkpoint_metadata`` only once ``net_cls`` itself existed)
are skipped, not treated as a forced mismatch.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.diffusion_cm_distill import (
    _gather_coef,
    _scalings_for_boundary_conditions,
)
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.chunked_dataset import load_h5_dataset_as_chunks
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.networks import Activation, DiffusionMLP, KernelInit
from rl_garden.policies.diffusion_policy import DiffusionPolicy, build_diffusion_net

_MISSING = object()


class ConsistencyDistillBC(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("ConsistencyDistillBC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        bc_checkpoint: str,
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
        net_cls: type[nn.Module] = DiffusionMLP,
        net_kwargs: Optional[dict[str, Any]] = None,
        cm_lr: float = 1e-4,
        weight_decay: float = 1e-6,
        cm_ema_decay: float = 0.95,
        cm_grad_clip_norm: Optional[float] = 1.0,
        cm_sigma_data: float = 0.5,
        cm_timestep_scaling: float = 0.1,
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
        if not isinstance(self.env.single_observation_space, spaces.Box):
            raise TypeError(
                "ConsistencyDistillBC is state-only (Box observations), matching "
                "its DiffusionBC teacher; vision is out of scope."
            )
        if cm_grad_clip_norm is not None and cm_grad_clip_norm <= 0:
            raise ValueError(
                f"cm_grad_clip_norm must be positive or None, got {cm_grad_clip_norm}."
            )

        self.dataset_path = dataset_path
        self.bc_checkpoint = bc_checkpoint
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
        self.net_cls = net_cls
        self.net_kwargs = dict(net_kwargs) if net_kwargs is not None else None
        self.cm_lr = cm_lr
        self.weight_decay = weight_decay
        self.cm_ema_decay = cm_ema_decay
        self.cm_grad_clip_norm = cm_grad_clip_norm
        self.cm_sigma_data = cm_sigma_data
        self.cm_timestep_scaling = cm_timestep_scaling
        self.num_traj = num_traj

        self._setup_model()
        self._load_dataset()

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("cm_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "bc_checkpoint": self.bc_checkpoint,
            "horizon_steps": self.horizon_steps,
            "cond_steps": self.cond_steps,
            "denoising_steps": self.denoising_steps,
            "mlp_dims": self.mlp_dims,
            "activation_fn": self.activation_fn,
            "residual_style": self.residual_style,
            "net_cls": self.net_cls.__name__,
            "net_kwargs": self.net_kwargs,
            "cm_lr": self.cm_lr,
            "cm_ema_decay": self.cm_ema_decay,
            "cm_grad_clip_norm": self.cm_grad_clip_norm,
            "cm_sigma_data": self.cm_sigma_data,
            "cm_timestep_scaling": self.cm_timestep_scaling,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "cm_student_state_dict": self.cm_student.state_dict(),
            "cm_target_state_dict": self.cm_target.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        student_state = state.get("cm_student_state_dict")
        if student_state is not None:
            self.cm_student.load_state_dict(student_state)
        target_state = state.get("cm_target_state_dict")
        if target_state is not None:
            self.cm_target.load_state_dict(target_state)

    # --- model / data setup ---

    def _build_net(self) -> nn.Module:
        action_dim = self.env.single_action_space.shape[0]
        obs_dim = self.env.single_observation_space.shape[0]
        cond_dim = obs_dim * self.cond_steps
        return build_diffusion_net(
            self.net_cls,
            action_dim=action_dim,
            horizon_steps=self.horizon_steps,
            cond_dim=cond_dim,
            time_dim=self.time_dim,
            kernel_init=self.kernel_init,
            mlp_dims=self.mlp_dims,
            activation_fn=self.activation_fn,
            residual_style=self.residual_style,
            net_kwargs=self.net_kwargs,
        )

    def _validate_teacher_config(self, hyperparameters: dict[str, Any]) -> None:
        """Compare this run's teacher-matching config against the fields the
        ``bc_checkpoint`` teacher itself recorded (an explicit allowlist, not
        a whole-dict comparison -- ``hyperparameters`` also carries fields
        like ``seed``/``device``/``batch_size``/``gamma`` that legitimately
        differ between the teacher and this distillation run). A field
        absent from ``hyperparameters`` (e.g. an older checkpoint saved
        before ``net_cls`` existed) is skipped, never treated as a forced
        mismatch."""
        mismatches: list[str] = []

        def check(name: str, actual: Any) -> None:
            stored = hyperparameters.get(name, _MISSING)
            if stored is not _MISSING and stored != actual:
                mismatches.append(f"{name}: teacher checkpoint={stored!r} vs this run={actual!r}")

        check("horizon_steps", self.horizon_steps)
        check("cond_steps", self.cond_steps)
        check("denoising_steps", self.denoising_steps)
        check("net_cls", self.net_cls.__name__)
        check("net_kwargs", self.net_kwargs)
        if self.net_cls is DiffusionMLP:
            check("mlp_dims", list(self.mlp_dims))
            check("activation_fn", self.activation_fn)
            check("residual_style", self.residual_style)

        if mismatches:
            raise ValueError(
                "ConsistencyDistillBC's config does not match its --bc_checkpoint "
                "teacher's own training config (student/target networks are "
                "warm-started from the teacher's weights, so these must match "
                "exactly):\n  " + "\n  ".join(mismatches)
            )

    def _setup_model(self) -> None:
        checkpoint = load_checkpoint_file(self.bc_checkpoint, map_location=self.device)
        self._validate_teacher_config(checkpoint["metadata"]["hyperparameters"])

        self.policy = DiffusionPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
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
            net_cls=self.net_cls,
            net_kwargs=self.net_kwargs,
        ).to(self.device)
        ema_net_state_dict = checkpoint["state"]["extra"]["ema_net_state_dict"]
        self.policy.net.load_state_dict(ema_net_state_dict)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        self.cm_student = self._build_net().to(self.device)
        self.cm_student.load_state_dict(ema_net_state_dict)
        self.cm_target = self._build_net().to(self.device)
        self.cm_target.load_state_dict(ema_net_state_dict)
        for p in self.cm_target.parameters():
            p.requires_grad_(False)

        self.cm_optimizer = make_optimizer(
            list(self.cm_student.parameters()),
            lr=self.cm_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )

    def _load_dataset(self) -> None:
        obs_history, action_chunks = load_h5_dataset_as_chunks(
            self.dataset_path,
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            device=self.device,
            num_traj=self.num_traj,
        )
        self._obs_history = obs_history
        self._action_chunks = action_chunks
        self._dataset_size = action_chunks.shape[0]

    # --- LCM self-consistency math (``DiffusionCMDistillOnline``'s own
    # ``_predicted_x0``/``_ddim_step``, rebound to this class's frozen
    # teacher policy instead of a ``DPPOPolicy``) ---

    def _predicted_x0(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_recip = _gather_coef(self.policy.sqrt_recip_alphas_cumprod, t, x.dim())
        sqrt_recipm1 = _gather_coef(self.policy.sqrt_recipm1_alphas_cumprod, t, x.dim())
        return sqrt_recip * x - sqrt_recipm1 * noise

    def _ddim_step(
        self, pred_x0: torch.Tensor, pred_noise: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        alphas_cumprod = self.policy.sqrt_alphas_cumprod**2
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]
        )
        alpha_prev = _gather_coef(alphas_cumprod_prev, t_prev, pred_x0.dim())
        dir_xt = (1.0 - alpha_prev).sqrt() * pred_noise
        return alpha_prev.sqrt() * pred_x0 + dir_xt

    # --- training ---

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        loss_sum = 0.0
        self.cm_student.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            idx = torch.randint(
                0, self._dataset_size, (self.batch_size,), device=self._action_chunks.device
            )
            obs_history = self._obs_history[idx]
            action_chunk = self._action_chunks[idx]
            loss_sum += self._distill_step(obs_history, action_chunk)

        return {"loss": loss_sum / gradient_steps}

    def _distill_step(self, obs_history: torch.Tensor, action_chunk: torch.Tensor) -> float:
        batch = action_chunk.shape[0]
        device = action_chunk.device
        cond = {"state": obs_history.reshape(batch, -1)}
        noise = torch.randn_like(action_chunk)

        start_t = torch.randint(0, self.denoising_steps, (batch,), device=device)
        prev_t = (start_t - 1).clamp(min=0)

        x_t = self.policy.q_sample(action_chunk, start_t, noise)

        c_skip_start, c_out_start = _scalings_for_boundary_conditions(
            start_t, self.cm_sigma_data, self.cm_timestep_scaling
        )
        c_skip_prev, c_out_prev = _scalings_for_boundary_conditions(
            prev_t, self.cm_sigma_data, self.cm_timestep_scaling
        )
        c_skip_start = c_skip_start.view(batch, 1, 1)
        c_out_start = c_out_start.view(batch, 1, 1)
        c_skip_prev = c_skip_prev.view(batch, 1, 1)
        c_out_prev = c_out_prev.view(batch, 1, 1)

        student_noise_pred = self.cm_student(x_t, start_t, cond=cond)
        student_pred_x0 = self._predicted_x0(x_t, start_t, student_noise_pred)
        student_out = c_skip_start * x_t + c_out_start * student_pred_x0

        with torch.no_grad():
            teacher_noise_pred = self.policy.net(x_t, start_t, cond=cond)
            teacher_pred_x0 = self._predicted_x0(x_t, start_t, teacher_noise_pred)
            x_prev = self._ddim_step(teacher_pred_x0, teacher_noise_pred, prev_t)

            target_noise_pred = self.cm_target(x_prev, prev_t, cond=cond)
            target_pred_x0 = self._predicted_x0(x_prev, prev_t, target_noise_pred)
            target = c_skip_prev * x_prev + c_out_prev * target_pred_x0

        distill_loss = F.mse_loss(student_out, target)

        self.cm_optimizer.zero_grad(set_to_none=True)
        distill_loss.backward()
        if self.cm_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.cm_student.parameters(), self.cm_grad_clip_norm
            )
        self.cm_optimizer.step()
        polyak_update(
            self.cm_student.parameters(),
            self.cm_target.parameters(),
            tau=1.0 - self.cm_ema_decay,
        )

        return float(distill_loss.detach().item())
