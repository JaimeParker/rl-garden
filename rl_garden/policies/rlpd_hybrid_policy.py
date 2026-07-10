"""RLPDHybrid policy: RLPDPolicy's continuous actor/critic restricted to the
non-gripper action dims, plus an independent :class:`DiscreteCritic` for a
discrete gripper action (open/hold/close). Mirrors HIL-SERL's
``sac_hybrid_single``: the two halves never share a joint distribution --
``predict()`` just concatenates the continuous sample and the discrete
argmax into one flat action, matching the env's existing action convention.
"""
from __future__ import annotations

from typing import Sequence

import torch
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.discrete_critic import DiscreteCritic
from rl_garden.policies.rlpd_policy import RLPDPolicy

# Discrete gripper action index -> numeric value appended to the continuous
# action vector, matching FrankaRealEnv's existing gripper convention
# (open=+1, close=-1). "hold" (0.0) lets the discrete critic choose to leave
# the gripper state unchanged, matching HIL-SERL's 3-way GraspCritic.
_DISCRETE_ACTION_VALUES = (-1.0, 0.0, 1.0)


class RLPDHybridPolicy(RLPDPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        discrete_hidden_dims: Sequence[int] = (256,),
        discrete_use_layer_norm: bool = True,
        n_discrete_actions: int = 3,
        **rlpd_kwargs,
    ) -> None:
        if n_discrete_actions != len(_DISCRETE_ACTION_VALUES):
            raise ValueError(
                f"n_discrete_actions must be {len(_DISCRETE_ACTION_VALUES)}, "
                f"got {n_discrete_actions}"
            )
        continuous_action_space = spaces.Box(
            low=action_space.low[:-1],
            high=action_space.high[:-1],
            dtype=action_space.dtype,
        )
        super().__init__(
            observation_space, continuous_action_space, features_extractor, **rlpd_kwargs
        )
        self.discrete_critic = DiscreteCritic(
            features_extractor.features_dim,
            discrete_hidden_dims,
            n_actions=n_discrete_actions,
            use_layer_norm=discrete_use_layer_norm,
        )
        self.discrete_target_critic = DiscreteCritic(
            features_extractor.features_dim,
            discrete_hidden_dims,
            n_actions=n_discrete_actions,
            use_layer_norm=discrete_use_layer_norm,
        )
        self.discrete_target_critic.load_state_dict(self.discrete_critic.state_dict())
        for p in self.discrete_target_critic.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "_discrete_action_values", torch.tensor(_DISCRETE_ACTION_VALUES)
        )

    def discrete_action_from_index(self, index: torch.Tensor) -> torch.Tensor:
        return self._discrete_action_values[index].unsqueeze(-1)

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        actor_input = self._transform_features_for_actor(features)
        if deterministic:
            continuous_action = self.actor.deterministic_action(actor_input)
        else:
            continuous_action, _ = self.actor.action_log_prob(actor_input)
        discrete_index = self.discrete_critic(features).argmax(dim=-1)
        discrete_action = self.discrete_action_from_index(discrete_index)
        return torch.cat([continuous_action, discrete_action], dim=-1)

    def discrete_critic_parameters(self):
        yield from self.discrete_critic.parameters()
