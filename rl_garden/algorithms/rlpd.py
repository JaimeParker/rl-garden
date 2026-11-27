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

from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.sac import RLPDParams
from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.networks import BackboneType
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
    policy_class = RLPDPolicy

    params_class = RLPDParams

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        critic_backbone_type: Optional[BackboneType] = None,
        params: RLPDParams | None = None,
        **sac_kwargs: Any,
    ) -> None:
        parameter_names = self.params_class.__dataclass_fields__
        overrides = {name: sac_kwargs.pop(name) for name in tuple(sac_kwargs)
                     if name in parameter_names}
        params = resolve_params(self.params_class, params, overrides)
        actor_dropout_rate = params.actor_dropout_rate
        critic_dropout_rate = params.critic_dropout_rate
        kernel_init = params.kernel_init
        backbone_type = params.backbone_type
        use_pnorm = params.use_pnorm
        exclude_bias_from_decay = params.exclude_bias_from_decay
        std_parameterization = params.std_parameterization
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.use_pnorm = use_pnorm
        self.exclude_bias_from_decay = exclude_bias_from_decay
        self.std_parameterization = std_parameterization
        self._init_prior_data_params()
        # critic_backbone_type is forwarded to (and stored by) SAC.__init__
        # rather than set here directly -- SAC.__init__ runs inside this
        # super().__init__() call and would otherwise overwrite a
        # pre-set value back to its own default.
        super().__init__(
            env,
            eval_env,
            critic_backbone_type=critic_backbone_type,
            params=params,
            **sac_kwargs,
        )

    def _policy_kwargs(self, observation_space) -> dict[str, Any]:
        # RLPDPolicy names every keyword, so the extras' position in the dict
        # does not matter.
        return {
            **super()._policy_kwargs(observation_space),
            "actor_dropout_rate": self.actor_dropout_rate,
            "critic_dropout_rate": self.critic_dropout_rate,
            "kernel_init": self.kernel_init,
            "backbone_type": self.backbone_type,
            "use_pnorm": self.use_pnorm,
            "std_parameterization": self.std_parameterization,
        }

    def _make_optimizers(self, *, exclude_bias_from_decay: bool = False):
        # Without weight decay there is nothing to exclude: keep one group.
        return super()._make_optimizers(
            exclude_bias_from_decay=exclude_bias_from_decay
            or (self.exclude_bias_from_decay and self.weight_decay > 0)
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
        }
