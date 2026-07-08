"""RLPD: SAC + high UTD + REDQ-style critic ensemble subsampling + LayerNorm
+ offline/online prior-data mixing from step 0.

High UTD, critic LayerNorm, and REDQ-style ensemble subsampling are already
first-class ``SAC``/``SACPolicy`` constructor knobs (``utd``, ``n_critics``,
``critic_subsample_size``, ``critic_use_layer_norm``) -- RLPD only needs to
(a) mix in a static prior-data buffer via ``PriorDataReplayMixin`` and
(b) forward a few knobs (dropout/kernel_init/backbone_type via ``SACPolicy``,
``use_pnorm`` via ``RLPDPolicy``) that plain ``SAC`` doesn't expose, with
RLPD-recommended defaults.
"""
from __future__ import annotations

from typing import Any, Optional

from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, KernelInit
from rl_garden.policies.rlpd_policy import RLPDPolicy


class RLPD(PriorDataReplayMixin, SAC):
    """RLPD (Ball et al. 2023) as a thin extension of ``SAC``.

    Defaults (``n_critics=10``, ``critic_subsample_size=2``,
    ``critic_use_layer_norm=True``) match the RLPD paper's recipe. Load a
    static prior-data buffer with ``load_offline_replay_buffer(...)`` to mix
    it into every training batch at ``offline_data_ratio`` from the first
    gradient step -- there is no separate offline-only pretraining phase.
    """

    _compatible_checkpoint_algorithms = ("RLPD",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        critic_use_layer_norm: bool = True,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        use_pnorm: bool = False,
        exclude_bias_from_decay: bool = False,
        **sac_kwargs: Any,
    ) -> None:
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.use_pnorm = use_pnorm
        self.exclude_bias_from_decay = exclude_bias_from_decay
        self._init_prior_data_params()
        super().__init__(
            env,
            eval_env,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            critic_use_layer_norm=critic_use_layer_norm,
            **sac_kwargs,
        )

    def _build_policy(self, features_extractor: BaseFeaturesExtractor) -> RLPDPolicy:
        return RLPDPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_impl=self.critic_impl,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            use_pnorm=self.use_pnorm,
            log_std_min=self.actor_log_std_min,
            log_std_mode=self.actor_log_std_mode,
            actor_feature_dim=self.actor_feature_dim,
            critic_spatial_emb_dim=self.critic_spatial_emb_dim,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        if not (self.exclude_bias_from_decay and self.weight_decay > 0):
            return
        # SAC._setup_model() already built self.q_optimizer/self.actor_optimizer
        # (uniform weight decay) and self._lr_schedulers bound to those exact
        # objects. Rebuilding just the optimizers here would leave the
        # schedulers pointing at now-discarded objects -- so the schedulers
        # must be rebuilt too, not just the optimizers.
        self.q_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.q_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
            exclude_bias_from_decay=True,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.policy_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
            exclude_bias_from_decay=True,
        )
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.q_optimizer, self.actor_optimizer)
        ]

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "actor_dropout_rate": self.actor_dropout_rate,
            "critic_dropout_rate": self.critic_dropout_rate,
            "kernel_init": self.kernel_init,
            "backbone_type": self.backbone_type,
            "use_pnorm": self.use_pnorm,
            "exclude_bias_from_decay": self.exclude_bias_from_decay,
        }
