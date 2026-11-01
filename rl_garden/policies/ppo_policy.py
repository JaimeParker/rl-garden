"""Actor-critic policy for PPO with pluggable feature extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    BackboneType,
    DiagGaussianActor,
    KernelInit,
    ValueNetwork,
)
from rl_garden.policies.base import BasePolicy, EncoderSharing

if TYPE_CHECKING:
    from rl_garden.buffers.rollout_buffer import ExtraSpecs

#: What an evaluate method returns last when called with ``return_outputs=True``.
EvaluationOutputs = dict[str, Any]


def get_ppo_arch(
    net_arch: Sequence[int] | dict[str, Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Resolve PPO actor/value hidden dims from an SB3-style net_arch spec."""
    if isinstance(net_arch, dict):
        if "pi" not in net_arch or "vf" not in net_arch:
            raise ValueError("PPO net_arch dict must contain both 'pi' and 'vf' keys.")
        return list(net_arch["pi"]), list(net_arch["vf"])
    shared = list(net_arch)
    return shared, list(shared)


class PPOPolicy(BasePolicy):
    """Continuous-action actor-critic policy used by PPO.

    Extension points for a subclass (all default to plain PPO behavior):
    ``_build_actor`` (the actor module), ``rollout_extra_specs`` (per-step
    ``infos`` entries the rollout buffer stores) and ``evaluation_outputs``
    (the update asks the evaluate methods for a trailing outputs dict)."""

    #: Extension point: when True, ``PPO``'s update calls the evaluate
    #: method it uses (``evaluate_actions``/``evaluate_actions_with_dist_params``,
    #: or the ``_sequence`` pair of a recurrent policy) with
    #: ``return_outputs=True`` and stores the trailing dict in
    #: ``MinibatchEval.outputs``, where ``PPO._aux_loss`` reads it. The dict
    #: of this class (and ``RecurrentPPOPolicy``) is ``{"latent": <the
    #: features the actor reads, one row per minibatch row>}``, detached
    #: exactly when the actor's input is (``actor_features_detached``).
    evaluation_outputs: ClassVar[bool] = False

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        features_dim: Optional[int] = None,
        log_std_init: float = -0.5,
        log_std_init_per_dim: Optional[Sequence[float]] = None,
        action_mean_bias_init: Optional[Sequence[float]] = None,
        action_mean_head_gain: Optional[float] = None,
        actor_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        critic_backbone_type: Optional[BackboneType] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
    ) -> None:
        if not isinstance(action_space, spaces.Box):
            raise TypeError("PPOPolicy only supports Box action spaces.")
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        actor_arch, value_arch = get_ppo_arch(net_arch)
        fd = features_dim if features_dim is not None else actor_extractor.features_dim
        critic_fd = (
            fd
            if self.critic_extractor is None
            else self.critic_extractor.features_dim
        )
        self.actor = self._build_actor(
            fd,
            action_space,
            actor_arch,
            log_std_init=log_std_init,
            log_std_init_per_dim=log_std_init_per_dim,
            action_mean_bias_init=action_mean_bias_init,
            action_mean_head_gain=action_mean_head_gain,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.value_net = ValueNetwork(
            critic_fd,
            value_arch,
            use_layer_norm=value_use_layer_norm,
            use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=critic_backbone_type or backbone_type,
        )

    def _build_actor(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        **actor_options: Any,
    ) -> nn.Module:
        """Extension point: build the actor head. Called once from
        ``__init__``, after the extractors and before ``value_net``, so the
        modules it creates consume the RNG at that position. ``features_dim``
        is the width of the features the actor reads, ``hidden_dims`` the
        resolved ``pi`` widths, ``actor_options`` ``DiagGaussianActor``'s
        keyword options (``log_std_init``, ``log_std_init_per_dim``,
        ``action_mean_bias_init``, ``action_mean_head_gain``,
        ``use_layer_norm``, ``use_group_norm``, ``num_groups``,
        ``dropout_rate``, ``kernel_init``, ``backbone_type``). The module
        must offer ``DiagGaussianActorBase``'s interface (``forward`` -> a
        ``Normal``, ``action_log_prob``, ``evaluate_action_log_prob``,
        ``deterministic_action``, ``clamp_action`` and a ``log_std``
        parameter); every method of this class reads it as ``self.actor``."""
        return DiagGaussianActor(features_dim, action_space, hidden_dims=hidden_dims, **actor_options)

    def rollout_extra_specs(self) -> ExtraSpecs:
        """Extension point: per-step ``infos`` entries the rollout buffer
        stores, ``{key: (per-env shape, dtype)}``. ``PPO`` reads
        ``infos[key]`` (a ``(num_envs, *shape)`` tensor; ``(num_envs,)`` is
        accepted for shape ``(1,)``) on every rollout step, raising
        ``ValueError`` when it is missing or mismatched, and each minibatch
        carries it in ``data.extras[key]`` (rows of a flat minibatch,
        ``(T, B, *shape)`` for a sequence minibatch). Read once, when the
        algorithm builds its rollout buffer. Default: none."""
        return {}

    @property
    def is_stateful(self) -> bool:
        return False

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Raw actor-extractor access with an explicit ``stop_gradient`` --
        an escape hatch for diagnostics/back-compat callers. The rollout/
        train paths below use ``extract_actor_features`` (``BasePolicy``) by
        default, which applies the ``encoder_sharing`` rule automatically."""
        return self.actor_extractor.extract(obs, stop_gradient=stop_gradient)

    def _actor_role_features(self, obs: Obs, stop_gradient_actor: Optional[bool]) -> torch.Tensor:
        if stop_gradient_actor is None:
            return self.extract_actor_features(obs)
        return self.extract_features(obs, stop_gradient=stop_gradient_actor)

    def forward(
        self,
        obs: Obs,
        deterministic: bool = False,
        *,
        stop_gradient_actor: Optional[bool] = None,
        sum_dims: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = self._actor_role_features(obs, stop_gradient_actor)
        value_features = self.extract_critic_features(obs)
        actions, log_prob, entropy = self.actor.action_log_prob(
            actor_features, deterministic=deterministic, sum_dims=sum_dims
        )
        values = self.value_net(value_features)
        return actions, values, log_prob, entropy

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_actor_features(obs)
        if deterministic:
            return self.actor.clamp_action(self.actor.deterministic_action(features))
        action, _, _ = self.actor.action_log_prob(features, deterministic=False)
        return self.actor.clamp_action(action)

    def predict_values(self, obs: Obs) -> torch.Tensor:
        return self.value_net(self.extract_critic_features(obs))

    def evaluate_actions(
        self,
        obs: Obs,
        actions: torch.Tensor,
        *,
        stop_gradient_actor: Optional[bool] = None,
        sum_dims: bool = True,
        return_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """``(values, log_prob, entropy)``; with ``return_outputs`` an
        ``EvaluationOutputs`` dict ``{"latent": <the features the actor
        read>}`` is appended (see ``evaluation_outputs``)."""
        actor_features = self._actor_role_features(obs, stop_gradient_actor)
        value_features = self.extract_critic_features(obs)
        log_prob, entropy = self.actor.evaluate_action_log_prob(
            actor_features, actions, sum_dims=sum_dims
        )
        values = self.value_net(value_features)
        if return_outputs:
            return values, log_prob, entropy, {"latent": actor_features}
        return values, log_prob, entropy

    def act_with_value_and_logprob(
        self, obs: Obs, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """Single-pass rollout call: action, value, log-prob, entropy, state.

        ``state`` is always ``None`` in and out for plain PPO -- a no-op
        passthrough. It exists so a future ``SequencePPO``-contract policy
        (RecurrentPPO/TransformerPPO) can implement the same method name
        while actually threading hidden state through it, letting an RLinf
        rollout worker call one method name regardless of which contract the
        concrete policy implements. See docs/design/rlinf-integration.md,
        "PPO contract" / "SequencePPO contract".
        """
        actions, values, log_prob, entropy = self.forward(obs, deterministic=False)
        return actions, values, log_prob, entropy, state

    def clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.clamp_action(actions)

    def act_with_value_logprob_and_dist_params(
        self,
        obs: Obs,
        deterministic: bool = False,
        *,
        stop_gradient_actor: Optional[bool] = None,
        sum_dims: bool = True,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Like ``act_with_value_and_logprob``, but also returns the rollout
        Gaussian's ``(mean, log_std)`` -- the "old" distribution params the
        adaptive-KL LR schedule (``lr_schedule="adaptive_kl"``) needs. Single
        actor forward pass, no duplicate compute. ``sum_dims=False`` keeps
        ``log_prob``/``entropy`` per action dimension, ``(N, A)``."""
        actor_features = self._actor_role_features(obs, stop_gradient_actor)
        value_features = self.extract_critic_features(obs)
        dist = self.actor(actor_features)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if sum_dims:
            log_prob = log_prob.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        values = self.value_net(value_features)
        log_std = self.actor.log_std.expand_as(dist.mean)
        return action, values, log_prob, entropy, dist.mean, log_std

    def evaluate_actions_with_dist_params(
        self,
        obs: Obs,
        actions: torch.Tensor,
        *,
        stop_gradient_actor: Optional[bool] = None,
        sum_dims: bool = True,
        return_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """Like ``evaluate_actions``, but also returns the current policy's
        ``(mean, log_std)`` -- the "new" distribution params the adaptive-KL
        LR schedule needs. Single actor forward pass, no duplicate compute.
        ``sum_dims=False`` keeps ``log_prob``/``entropy`` per action
        dimension, ``(N, A)``. ``return_outputs`` appends the outputs dict
        like ``evaluate_actions``."""
        actor_features = self._actor_role_features(obs, stop_gradient_actor)
        value_features = self.extract_critic_features(obs)
        dist = self.actor(actor_features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if sum_dims:
            log_prob = log_prob.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        values = self.value_net(value_features)
        log_std = self.actor.log_std.expand_as(dist.mean)
        if return_outputs:
            return values, log_prob, entropy, dist.mean, log_std, {"latent": actor_features}
        return values, log_prob, entropy, dist.mean, log_std
