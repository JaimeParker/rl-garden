"""RLPDHybrid policy: RLPDPolicy's continuous actor/critic restricted to the
non-gripper action dims, plus an independent :class:`DiscreteCritic` for the
discrete gripper actions (open/hold/close per gripper). Mirrors HIL-SERL's
``sac_hybrid_single``: the two halves never share a joint distribution --
``predict()`` just concatenates the continuous sample and the decoded discrete
argmax into one flat action, matching the env's existing action convention.

Action layout: the last ``num_grippers`` env action dims are the grippers, in
order; every dim before them is continuous. The discrete critic has one output
per joint gripper choice, ``3 ** num_grippers`` in all; ``encode_gripper_labels``
and ``decode_gripper_index`` map between the gripper dims and that index.
"""
from __future__ import annotations

from typing import Optional, Sequence

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


def _gripper_place_values(num_grippers: int, device: torch.device) -> torch.Tensor:
    """``(3**(k-1), ..., 3, 1)``: the weight of each gripper's digit in the joint index."""
    base = len(_DISCRETE_ACTION_VALUES)
    return base ** torch.arange(num_grippers - 1, -1, -1, device=device)


def gripper_index_digits(index: torch.Tensor, num_grippers: int) -> torch.Tensor:
    """``(..., k)`` per-gripper value indices (into ``_DISCRETE_ACTION_VALUES``) of a
    ``(...,)`` joint index: ``d_i = (index // 3**(k-1-i)) % 3``."""
    place_values = _gripper_place_values(num_grippers, index.device)
    digits = torch.div(index.unsqueeze(-1), place_values, rounding_mode="floor")
    return digits % len(_DISCRETE_ACTION_VALUES)


def encode_gripper_labels(gripper_actions: torch.Tensor) -> torch.Tensor:
    """Joint discrete index ``(...,)`` of ``(..., k)`` gripper actions (the last ``k``
    env action dims, in order).

    Each gripper action ``a_i`` is mapped to the index ``d_i`` of its nearest value in
    ``_DISCRETE_ACTION_VALUES`` (``-1 -> 0``, ``0 -> 1``, ``+1 -> 2``; a tie goes to the
    lower value, values outside ``[-1, 1]`` to the nearest end), and the label is
    ``idx = sum_i d_i * 3**(k-1-i)``: a base-3 number whose first gripper is the most
    significant digit. For ``k = 1`` the label is ``d_0``, the nearest-of-three index.
    """
    values = gripper_actions.new_tensor(_DISCRETE_ACTION_VALUES)
    digits = (gripper_actions.unsqueeze(-1) - values).abs().argmin(dim=-1)
    place_values = _gripper_place_values(gripper_actions.shape[-1], digits.device)
    return (digits * place_values).sum(dim=-1)


def decode_gripper_index(index: torch.Tensor, num_grippers: int) -> torch.Tensor:
    """Inverse of ``encode_gripper_labels``: the ``(..., k)`` gripper values (each one of
    ``_DISCRETE_ACTION_VALUES``) of a ``(...,)`` joint index, gripper ``i`` taking
    ``_DISCRETE_ACTION_VALUES[(index // 3**(k-1-i)) % 3]``."""
    values = torch.tensor(_DISCRETE_ACTION_VALUES, device=index.device)
    return values[gripper_index_digits(index, num_grippers)]


class RLPDHybridPolicy(RLPDPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
        discrete_hidden_dims: Sequence[int] = (256,),
        discrete_use_layer_norm: bool = True,
        n_discrete_actions: Optional[int] = None,
        num_grippers: int = 1,
        **rlpd_kwargs,
    ) -> None:
        if num_grippers < 1:
            raise ValueError(f"num_grippers must be >= 1, got {num_grippers}")
        joint_actions = len(_DISCRETE_ACTION_VALUES) ** num_grippers
        if n_discrete_actions is None:
            n_discrete_actions = joint_actions
        if n_discrete_actions != joint_actions:
            raise ValueError(
                f"n_discrete_actions must be {joint_actions} "
                f"({len(_DISCRETE_ACTION_VALUES)}**num_grippers), got {n_discrete_actions}"
            )
        continuous_action_space = spaces.Box(
            low=action_space.low[:-num_grippers],
            high=action_space.high[:-num_grippers],
            dtype=action_space.dtype,
        )
        super().__init__(
            observation_space, continuous_action_space, actor_extractor, **rlpd_kwargs
        )
        self.num_grippers = int(num_grippers)
        # discrete_critic is architecturally a third critic head -- follows
        # the critic role's extractor (falls back to actor_extractor in the
        # default shared case).
        self.discrete_critic = DiscreteCritic(
            self.critic_features_dim,
            discrete_hidden_dims,
            n_actions=n_discrete_actions,
            use_layer_norm=discrete_use_layer_norm,
        )
        self.discrete_target_critic = DiscreteCritic(
            self.critic_features_dim,
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
        """The ``(B, num_grippers)`` gripper values of each joint discrete index."""
        return self._discrete_action_values[gripper_index_digits(index, self.num_grippers)]

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        actor_input = self._transform_features_for_actor(features)
        if deterministic:
            continuous_action = self.actor.deterministic_action(actor_input)
        else:
            continuous_action, _ = self.actor.action_log_prob(actor_input)
        discrete_features = self.critic_features_for(obs, features, stop_gradient=True)
        discrete_index = self.discrete_critic(discrete_features).argmax(dim=-1)
        discrete_action = self.discrete_action_from_index(discrete_index)
        return torch.cat([continuous_action, discrete_action], dim=-1)

    def discrete_critic_parameters(self):
        yield from self.discrete_critic.parameters()
