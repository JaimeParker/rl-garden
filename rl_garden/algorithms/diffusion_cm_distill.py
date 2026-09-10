"""DiffusionCMDistillOnline: DPPO's online PPO fine-tuning fused with a
Latent-Consistency-Model (LCM) style one-step distillation step, run once
per training iteration right after that iteration's PPO update completes.

Ported from ``3rd_party/RL-100``'s ``distill_phase='online'`` mode
(``unidpg/uni_ppo.py::distill_update`` -> ``policy/rl100_3d.py::
compute_ddim2cm_loss``, verified against source directly), the *faithful*
LCM self-consistency path (boundary-condition ``c_skip``/``c_out``
parameterization + EMA target network), not RL-100's simplified
action-space-MSE shortcut.

**Not a subclass of ``PPO``.** ``DPPOCore``'s cooperative checkpoint-hook
``super()`` chain (``_checkpoint_metadata``/``_optimizer_names``/etc.)
would, under ``class X(DPPOCore, PPO)``, resolve into ``PPO``'s own hook
implementations -- which reference attributes (``clip_coef``, ``net_arch``,
``log_std_init``, ...) that only exist once ``PPO.__init__`` has actually
run. It must not run here: ``PPO``'s Gaussian-actor-critic construction is
irrelevant to a diffusion policy, exactly why ``DPPO`` itself never
inherits ``PPO``, only ``OnPolicyAlgorithm``. This class instead subclasses
``DPPO`` directly (not just its ``DPPOCore`` mixin) -- ``DPPO`` already
*is* "PPO's clip-ratio math, applied to a denoising chain treated as MDP
steps," so subclassing it wholesale reuses that machinery (rollout,
chain-buffer, ``train()``'s PPO update) with zero duplication, adding only
the distillation step on top via ``train()``/``_setup_model()`` overrides
that each call ``super()`` first.

At eval/inference time ``self.policy`` stays the multi-step teacher
(``DPPOPolicy``, unchanged from ``DPPO``'s own eval path) -- the one-step
CM student (``self.cm_student``) is a separate attribute meant for
deployment-time export once training is judged good enough, not swapped
into the online rollout/eval loop.
"""
from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.dppo import DPPO
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.optim import make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.networks.diffusion_mlp import DiffusionMLP


def _scalings_for_boundary_conditions(
    timestep: torch.Tensor, sigma_data: float = 0.5, timestep_scaling: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """RL-100's ``model/common/cm_util.py::scalings_for_boundary_conditions``,
    generalized to make its divisor tunable. Upstream's own function body
    hardcodes ``timestep / 0.1`` and ignores its own declared
    ``timestep_scaling=10.0`` default parameter -- a latent inconsistency in
    the source, not resolved here since which one is "correct" cannot be
    determined from the code alone. ``timestep_scaling=0.1`` (this
    function's default) reproduces that literal body exactly.

    **Schedule-length caveat**: upstream's discrete DDPM index runs to
    ~1000; this repo's ``denoising_steps`` is typically 5-20. With
    ``timestep_scaling=0.1``, ``c_skip`` is already ~0 (and ``c_out`` ~1,
    i.e. "ignore the skip connection, use the raw network prediction") for
    every ``t >= 1`` -- the boundary condition only does anything
    distinguishable exactly at ``t = 0``. This is not obviously wrong (LCM's
    boundary condition is *supposed* to bite hardest near ``t = 0``, and
    upstream's own sampled timesteps, which never go below its DDIM
    subsampling's step size, show the same "c_skip is negligible everywhere
    it actually samples" shape), but it does mean the mechanism is far more
    binary (t=0 vs. everything else) at this schedule length than upstream's
    smoother-looking curve over 1000 steps. Callers can widen this via
    ``DiffusionCMDistillOnline(cm_timestep_scaling=...)`` (larger values
    grade ``c_skip`` more smoothly across ``[0, denoising_steps)``) if that
    matters for a given run; the default stays at the literal ported value."""
    t = timestep.float() / timestep_scaling
    c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    c_out = t / (t**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def _gather_coef(buffer: torch.Tensor, t: torch.Tensor, x_ndim: int) -> torch.Tensor:
    out = buffer.gather(-1, t)
    return out.reshape(t.shape[0], *([1] * (x_ndim - 1)))


class DiffusionCMDistillOnline(DPPO):
    _compatible_checkpoint_algorithms = ("DiffusionCMDistillOnline",)

    def __init__(
        self,
        env: Any,
        *,
        bc_checkpoint: Optional[str] = None,
        cm_mlp_dims: Optional[Sequence[int]] = None,
        cm_lr: float = 1e-4,
        cm_ema_decay: float = 0.95,
        cm_grad_clip_norm: Optional[float] = 1.0,
        cm_sigma_data: float = 0.5,
        cm_timestep_scaling: float = 0.1,
        **dppo_kwargs: Any,
    ) -> None:
        self.cm_mlp_dims = list(cm_mlp_dims) if cm_mlp_dims is not None else None
        self.cm_lr = cm_lr
        self.cm_ema_decay = cm_ema_decay
        self.cm_grad_clip_norm = cm_grad_clip_norm
        self.cm_sigma_data = cm_sigma_data
        self.cm_timestep_scaling = cm_timestep_scaling
        super().__init__(env, bc_checkpoint=bc_checkpoint, **dppo_kwargs)
        if bc_checkpoint is not None:
            # DPPO.__init__ already loaded this checkpoint's weights into
            # self.policy.actor/actor_ft; RL-100's own teacher/distilled_model/
            # target_model all start as deepcopies of the *same* pretrained
            # network (rl100_3d.py:325-343), so cm_student/cm_target get the
            # same warm start here rather than a fresh random init.
            checkpoint = load_checkpoint_file(bc_checkpoint, map_location=self.device)
            ema_net_state_dict = checkpoint["state"]["extra"]["ema_net_state_dict"]
            self.cm_student.load_state_dict(ema_net_state_dict)
            self.cm_target.load_state_dict(ema_net_state_dict)

    def _setup_model(self) -> None:
        super()._setup_model()
        mlp_dims = self.cm_mlp_dims if self.cm_mlp_dims is not None else self.actor_mlp_dims
        self.cm_student = DiffusionMLP(
            action_dim=self.policy.action_dim,
            horizon_steps=self.horizon_steps,
            cond_dim=self.policy.features_extractor.features_dim,
            time_dim=self.time_dim,
            mlp_dims=mlp_dims,
            activation_fn=self.actor_activation_fn,
            residual_style=self.actor_residual_style,
            kernel_init=self.kernel_init,
        ).to(self.device)
        self.cm_target = copy.deepcopy(self.cm_student)
        for p in self.cm_target.parameters():
            p.requires_grad_(False)
        self.cm_optimizer = make_optimizer(
            list(self.cm_student.parameters()),
            lr=self.cm_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )

    # --- checkpoint hooks: extend DPPO's (no PPO in the MRO, see module
    # docstring -- these super() chains resolve exactly as they do for DPPO
    # today, through DPPOCore to OnPolicyAlgorithm/BaseAlgorithm) ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*super()._optimizer_names(), "cm_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "cm_mlp_dims": self.cm_mlp_dims,
            "cm_lr": self.cm_lr,
            "cm_ema_decay": self.cm_ema_decay,
            "cm_grad_clip_norm": self.cm_grad_clip_norm,
            "cm_sigma_data": self.cm_sigma_data,
            "cm_timestep_scaling": self.cm_timestep_scaling,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "cm_student_state_dict": self.cm_student.state_dict(),
            "cm_target_state_dict": self.cm_target.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        student_state = state.get("cm_student_state_dict")
        if student_state is not None:
            self.cm_student.load_state_dict(student_state)
        target_state = state.get("cm_target_state_dict")
        if target_state is not None:
            self.cm_target.load_state_dict(target_state)

    # --- training ---

    def train(self) -> dict[str, float]:
        metrics = super().train()
        metrics.update(self._distill_step())
        return metrics

    def _predicted_x0(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Unclamped predicted-x0 (RL-100's ``cm_util.py::predicted_origin``,
        epsilon parameterization). Deliberately not
        ``DiffusionProcess._predict_x0`` -- that method applies
        ``denoised_clip_value`` for DDPM-sampling fidelity, but RL-100's own
        ``compute_ddim2cm_loss`` has no such clamp."""
        policy = self.policy
        sqrt_recip = _gather_coef(policy.sqrt_recip_alphas_cumprod, t, x.dim())
        sqrt_recipm1 = _gather_coef(policy.sqrt_recipm1_alphas_cumprod, t, x.dim())
        return sqrt_recip * x - sqrt_recipm1 * noise

    def _ddim_step(
        self, pred_x0: torch.Tensor, pred_noise: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        """RL-100's ``cm_util.py::DDIMSolver.ddim_step``, simplified: this
        repo's diffusion chain already samples at "full resolution" (one
        schedule step per index, no separate train/inference-step split), so
        the ``DDIMSolver``'s subsampled-timestep table collapses to plain
        indexing into ``alphas_cumprod`` -- the ddim_step math itself
        (``x_prev = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev) *
        pred_noise``) is unchanged."""
        policy = self.policy
        alphas_cumprod = policy.sqrt_alphas_cumprod**2
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]
        )
        alpha_prev = _gather_coef(alphas_cumprod_prev, t_prev, pred_x0.dim())
        dir_xt = (1.0 - alpha_prev).sqrt() * pred_noise
        return alpha_prev.sqrt() * pred_x0 + dir_xt

    def _distill_step(self) -> dict[str, float]:
        """One LCM self-consistency gradient step per training iteration,
        using this iteration's just-collected on-policy rollout data (the
        chain buffer's final/clean trajectory, same data DPPO's own PPO
        update just trained on). Runs after ``super().train()`` completes
        (RL-100's ``distill_phase='online'`` ordering) -- the teacher forward
        below is under ``torch.no_grad()`` so this step cannot leak gradient
        into the PPO-trained teacher (``actor``/``actor_ft``)."""
        obs_flat = flatten_leading_dims(self.rollout_buffer.obs)
        # Detached: this distillation step must not train the shared
        # features_extractor (only the critic loss does, see DPPO._dppo_loss's
        # own comment) -- cm_optimizer doesn't include it anyway (so this was
        # harmless-but-wasted compute, not a correctness bug), but detaching
        # here makes that explicit rather than relying on optimizer-grouping.
        cond = self.policy._cond(obs_flat, stop_gradient=True)
        latents = self._chain_buffer.chains[:, :, -1].reshape(
            -1, self.horizon_steps, self.policy.action_dim
        )
        batch = latents.shape[0]
        device = latents.device
        noise = torch.randn_like(latents)

        start_t = torch.randint(0, self.denoising_steps, (batch,), device=device)
        prev_t = (start_t - 1).clamp(min=0)

        x_t = self.policy.q_sample(latents, start_t, noise)

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
            teacher_noise_pred = self.policy._predict_noise_mixed(x_t, start_t, cond)
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

        return {"cm_distill_loss": float(distill_loss.detach().item())}
