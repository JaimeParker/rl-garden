"""SACFlow: SAC with a flow-matching actor instead of a Gaussian actor.

Ports RLinf's SACFlow (`RLinf/rlinf/models/embodiment/flow_policy/`)
into rl-garden's SAC hierarchy. RLinf's actor loss is verified identical to
plain SAC's -- `alpha*log_pi - Q`, unmodified
(`RLinf/rlinf/workers/actor/fsdp_sac_policy_worker.py:524`) -- so
this subclass only swaps the actor network (`FlowMatchingActor`, see
`rl_garden/networks/flow_actor.py`) via `_build_policy`. `SACCore` (critic
loss, target Q, actor loss, alpha tuning, train loop) is inherited from `SAC`
unmodified, same shape as `SequenceSAC` swapping in a recurrent actor.

Flat Box observations only for this version -- Dict/RGBD structured obs are
rejected, mirroring `SequenceSAC._build_policy`'s guard.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from rl_garden.algorithms.sac import SAC
from rl_garden.policies.sac_flow_policy import SACFlowPolicy


class SACFlow(SAC):
    _compatible_checkpoint_algorithms = ("SACFlow",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        denoising_steps: int = 4,
        noise_std: float = 0.3,
        flow_hidden_dims: Sequence[int] = (256, 256, 256),
        flow_use_layer_norm: bool = False,
        **sac_kwargs: Any,
    ) -> None:
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_policy override below reads
        # these attributes.
        self.denoising_steps = denoising_steps
        self.noise_std = noise_std
        self.flow_hidden_dims = list(flow_hidden_dims)
        self.flow_use_layer_norm = flow_use_layer_norm
        super().__init__(env, eval_env, **sac_kwargs)

    def _build_policy(self, features_extractor) -> SACFlowPolicy:
        if features_extractor.structured_feature_config() is not None:
            raise NotImplementedError(
                f"{type(self).__name__} only supports flat-latent feature "
                "extractors this round (Dict/RGBD structured obs untested)."
            )
        return SACFlowPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_use_layer_norm=self.critic_use_layer_norm,
            critic_impl=self.critic_impl,
            flow_hidden_dims=self.flow_hidden_dims,
            denoising_steps=self.denoising_steps,
            noise_std=self.noise_std,
            flow_use_layer_norm=self.flow_use_layer_norm,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "denoising_steps": self.denoising_steps,
            "noise_std": self.noise_std,
            "flow_hidden_dims": self.flow_hidden_dims,
            "flow_use_layer_norm": self.flow_use_layer_norm,
        }
