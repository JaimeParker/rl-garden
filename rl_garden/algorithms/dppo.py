"""DPPO: Diffusion PPO fine-tuning (phase 2, after ``DiffusionBC``).

Ported from ``3rd_party/dppo/agent/finetune/train_ppo_diffusion_agent.py`` +
``model/diffusion/diffusion_ppo.py::PPODiffusion.loss``, verified against
source directly. DDPM only; the reference's optional BC-regularization loss
term (``use_bc_loss``) is out of scope for this port -- it requires a full
non-subsampled ``get_logprobs`` pass over a *second*, freshly-sampled chain
per training step (roughly doubling the update's cost) for a term the
reference itself treats as optional and off in its own example configs.
Likewise the reference's ``entropy_loss = -eta.mean()`` term is omitted: for
DDPM, ``eta`` is hardcoded to ``torch.ones_like(mu)`` (a DDIM-only quantity),
making ``entropy_loss`` an algebraic constant with zero gradient -- and the
reference's own ``ent_coef`` defaults to 0 regardless.

``OnPolicyAlgorithm.learn()`` (the rollout loop, GAE call, checkpointing) is
reused completely unmodified -- see ``DPPOPolicy``'s module docstring:
DPPO's critic is a pure function of ``obs`` (exactly like standard PPO's),
so the rollout step can compute real bootstrap values inline, and
``compute_returns_and_advantage`` needs no override. The chain rollout step,
the pair-minibatch PPO update, the two optimizers and their schedulers are
``ChainPPOCore``'s (shared with ``FlowPPO``); this class supplies the DDPM
sampling (``_sample_rollout_chain``) and the denoising-step-aware ratio,
advantage discount and clip schedule (``_chain_policy_terms``).

As in the reference, advantages are normalized and quantile-clamped per
minibatch of ``(env step, denoising step)`` pairs
(``advantage_normalization = "minibatch"``), then discounted per pair by
the denoising discount.

Action chunking (``horizon_steps``/``act_steps``) is handled entirely by
``rl_garden.envs.wrappers.ActionChunkWrapper`` around the env passed in --
this algorithm does not know or care that the env is chunked; it only knows
``self.env.single_action_space.shape == (act_steps, action_dim)``.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.chain_ppo_core import (
    ChainMinibatch,
    ChainPolicyTerms,
    ChainPPOCore,
)
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.logger import Logger
from rl_garden.common.types import Obs
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks import Activation, KernelInit
from rl_garden.observations import ObsGroups
from rl_garden.policies.dppo_policy import DPPOPolicy


class DPPO(ChainPPOCore):
    """PPO over the fine-tuned tail of a DDPM denoising chain (``K =
    ft_denoising_steps`` steps per env action), on ``ChainPPOCore``.

    DPPO-specific terms (``_chain_policy_terms``): per-element log-probs
    clamped to ``[-5, 2]`` and averaged over the first ``reward_horizon``
    actions of the chunk; each pair's advantage discounted by
    ``gamma_denoising ** (K - k - 1)`` (``k`` its denoising-step index, so
    the step closest to the clean action is undiscounted); and a clip
    coefficient annealed from ``clip_ploss_coef_base`` at ``k = 0`` to
    ``clip_ploss_coef`` at ``k = K - 1`` (rate ``clip_ploss_coef_rate``).
    The reference's ``entropy_loss = -eta.mean()`` term is omitted (a
    constant for DDPM, see the module docstring)."""

    _compatible_checkpoint_algorithms = ("DPPO",)
    # diffusion_ppo.py::PPODiffusion.loss normalizes and clamps the
    # minibatch's advantages, before the denoising discount.
    advantage_normalization = "minibatch"

    def __init__(
        self,
        env: Any,
        bc_checkpoint: Optional[str] = None,
        eval_env: Optional[Any] = None,
        num_steps: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon_steps: int = 4,
        act_steps: int = 4,
        denoising_steps: int = 20,
        ft_denoising_steps: int = 10,
        actor_mlp_dims: Optional[Sequence[int]] = None,
        actor_activation_fn: Optional[Activation] = "relu",
        actor_residual_style: bool = True,
        critic_mlp_dims: Optional[Sequence[int]] = None,
        critic_activation_fn: Optional[Activation] = "mish",
        critic_residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        critic_warmup_updates: int = 0,
        update_epochs: int = 5,
        update_batch_size: int = 50_000,
        norm_adv: bool = True,
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.01,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        vf_coef: float = 0.5,
        target_kl: Optional[float] = 1.0,
        reward_horizon: Optional[int] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        image_augmentation_seed: Optional[int] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        bootstrap_at_done: str = "truncated",
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        if not (1 <= act_steps <= horizon_steps):
            raise ValueError(f"act_steps must be in [1, horizon_steps], got {act_steps}.")
        if not (1 <= ft_denoising_steps <= denoising_steps):
            raise ValueError(
                f"ft_denoising_steps must be in [1, denoising_steps], got {ft_denoising_steps}."
            )
        self._image_augmentation_seed = image_augmentation_seed
        super().__init__(
            env,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
            actor_mlp_dims=actor_mlp_dims if actor_mlp_dims is not None else [512, 512, 512],
            actor_activation_fn=actor_activation_fn,
            critic_mlp_dims=critic_mlp_dims if critic_mlp_dims is not None else [256, 256, 256],
            critic_activation_fn=critic_activation_fn,
            kernel_init=kernel_init,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            critic_warmup_updates=critic_warmup_updates,
            update_epochs=update_epochs,
            update_batch_size=update_batch_size,
            norm_adv=norm_adv,
            clip_vloss_coef=clip_vloss_coef,
            clip_advantage_lower_quantile=clip_advantage_lower_quantile,
            clip_advantage_upper_quantile=clip_advantage_upper_quantile,
            vf_coef=vf_coef,
            target_kl=target_kl,
            eval_env=eval_env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            finite_horizon_gae=finite_horizon_gae,
            bootstrap_at_done=bootstrap_at_done,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.denoising_steps = denoising_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.actor_residual_style = actor_residual_style
        self.critic_residual_style = critic_residual_style
        self.time_dim = time_dim
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.reward_horizon = reward_horizon if reward_horizon is not None else act_steps

        expected_action_shape = (self.act_steps,) + env.single_action_space.shape[1:]
        if env.single_action_space.shape != expected_action_shape:
            raise ValueError(
                "DPPO expects env.single_action_space to already be chunked by "
                "ActionChunkWrapper(env, act_steps=...); got shape "
                f"{env.single_action_space.shape}, expected {expected_action_shape}."
            )

        self._setup_model()
        if bc_checkpoint is not None:
            checkpoint = load_checkpoint_file(bc_checkpoint, map_location=self.device)
            # Every DiffusionBC checkpoint's recorded observation_space is
            # Dict now (boundary normalization always normalizes a bare Box
            # env into Dict({"state": Box})), so "type == Dict" alone no
            # longer distinguishes vision from state-only -- check the key
            # set instead: DPPOPolicy's actor is
            # state-only, so any key other than "state" (an rgb_<cam>/
            # depth_<cam> image key) means this checkpoint was vision-trained.
            ckpt_obs_space = checkpoint["metadata"]["observation_space"]
            ckpt_obs_keys = set(ckpt_obs_space.get("spaces", {}).keys())
            if ckpt_obs_space.get("type") == "Dict" and ckpt_obs_keys != {"state"}:
                raise ValueError(
                    "--bc_checkpoint was trained with Dict (vision) observations "
                    f"(its recorded observation_space keys are {sorted(ckpt_obs_keys)}, "
                    "not just 'state'); DPPO's --bc_checkpoint path requires a "
                    "state-only DiffusionBC checkpoint."
                )
            ema_net_state_dict = checkpoint["state"]["extra"]["ema_net_state_dict"]
            self.policy.load_actor_weights(ema_net_state_dict)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_steps": self.horizon_steps,
            "act_steps": self.act_steps,
            "denoising_steps": self.denoising_steps,
            "ft_denoising_steps": self.ft_denoising_steps,
            "image_augmentation_seed": self._image_augmentation_seed,
        }

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        raw_action_space = spaces.Box(
            low=self.env.single_action_space.low[0],
            high=self.env.single_action_space.high[0],
            shape=self.env.single_action_space.shape[1:],
            dtype=self.env.single_action_space.dtype,
        )
        extractor_kwargs = self._policy_extractor_kwargs(
            obs_space, augmentation_seed=self._image_augmentation_seed
        )
        self.policy = DPPOPolicy(
            observation_space=obs_space,
            action_space=raw_action_space,
            **extractor_kwargs,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            denoising_steps=self.denoising_steps,
            ft_denoising_steps=self.ft_denoising_steps,
            actor_mlp_dims=self.actor_mlp_dims,
            actor_activation_fn=self.actor_activation_fn,
            actor_residual_style=self.actor_residual_style,
            critic_mlp_dims=self.critic_mlp_dims,
            critic_activation_fn=self.critic_activation_fn,
            critic_residual_style=self.critic_residual_style,
            time_dim=self.time_dim,
            kernel_init=self.kernel_init,
            denoised_clip_value=self.denoised_clip_value,
            randn_clip_value=self.randn_clip_value,
            final_action_clip_value=self.final_action_clip_value,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
        ).to(self.device)
        self._build_chain_training_state(
            self.policy.actor_parameters(),
            self.policy.critic_and_encoder_parameters(),
            chain_steps=self.ft_denoising_steps,
            horizon_steps=self.horizon_steps,
            action_dim=self.policy.action_dim,
        )

    # --- ChainPPOCore hooks ---

    def _sample_rollout_chain(
        self, obs: Obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DDPM sampling of the fine-tuned tail, its per-step log-probs and
        the first ``act_steps`` actions of the sampled chunk."""
        chain, action_chunk = self.policy.sample_rollout_chain(obs)
        old_log_probs = self.policy.get_logprobs(self.policy._cond(obs), chain)
        return chain, old_log_probs, action_chunk[:, : self.act_steps]

    def _chain_policy_terms(self, batch: ChainMinibatch) -> ChainPolicyTerms:
        # Gradient isolation: the critic loss (ChainPPOCore, through
        # policy.predict_values) trains a shared encoder. The actor's
        # log-prob path must not also backprop into it under
        # encoder_sharing="shared_critic_grad" (see
        # BasePolicy.actor_features_detached), or the PPO policy loss would
        # additionally train it every step -- detach the copy fed to the
        # actor in that case.
        cond = self.policy._cond(batch.obs)
        if self.policy.actor_features_detached:
            cond = {key: value.detach() for key, value in cond.items()}
        new_log_probs = self.policy.get_logprobs_subsample(
            cond, batch.chains_prev, batch.chains_next, batch.chain_step_inds
        ).clamp(min=-5, max=2)
        old_log_probs = batch.old_log_probs.clamp(min=-5, max=2)
        new_log_probs = new_log_probs[:, : self.reward_horizon, :].mean(dim=(-1, -2))
        old_log_probs = old_log_probs[:, : self.reward_horizon, :].mean(dim=(-1, -2))
        logratio = new_log_probs - old_log_probs

        step = batch.chain_step_inds.float()
        discount = self.gamma_denoising ** (self.ft_denoising_steps - step - 1)
        if self.ft_denoising_steps > 1:
            t = step / (self.ft_denoising_steps - 1)
            clip_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            # The reference's value for a single fine-tuned step: the step
            # index itself, always 0.
            clip_coef = step
        return ChainPolicyTerms(
            logratio=logratio,
            ratio=logratio.exp(),
            advantages=batch.advantages * discount,
            clip_coef=clip_coef,
        )
