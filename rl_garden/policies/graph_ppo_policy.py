"""PPO policy whose trunk is a declared graph.

The YAML reference (node types, edge fields, rules) is the module docstring
of ``rl_garden.networks.graph.spec``. The graph decides everything between
the observation and the two PPO heads, including gradient routing (per-edge
``detach``); the Gaussian actor head and the scalar value head stay the
``PPOPolicy`` contract and read the inputs of the ``actor``/``value`` sinks.
The user guide, with worked examples, is ``docs/guides/policy-graphs.md``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.graph.executor import (
    GraphState,
    StateLayout,
    run_flat,
    run_sequence,
    run_step,
)
from rl_garden.networks.graph.losses import TargetSpecs, info_target_specs
from rl_garden.networks.graph.nodes import build_graph_nodes
from rl_garden.networks.graph.spec import GraphSource, GraphSpec
from rl_garden.policies.ppo_policy import PPOPolicy

#: Node name -> output of one graph run (a sink maps to its head's input).
GraphOutputs = dict[str, torch.Tensor]


class GraphFeaturesStandIn(BaseFeaturesExtractor):
    """Parameter-free stand-in that only reports a ``features_dim``: the
    ``actor_extractor`` of a graph policy with no single encoder on its actor
    path, and the carrier that sizes ``value_net`` at the value sink's width.
    Features come from the graph; calling it is an error."""

    def forward(self, obs: Obs) -> torch.Tensor:
        raise RuntimeError(
            "GraphPPOPolicy computes features through its graph; use "
            "extract_actor_features / extract_critic_features."
        )


class GraphPPOPolicy(PPOPolicy):
    """``PPOPolicy`` whose trunk is a ``GraphSpec``.

    Construction order (decides RNG consumption): graph nodes in
    ``GraphSpec.topological_order()`` (ties broken by name), then ``actor``,
    then ``value_net``. The nodes live in ``graph_nodes`` (state_dict keys
    ``graph_nodes.<name>.``). ``actor_extractor`` is the one encoder node on
    the actor sink's path (a ``GraphFeaturesStandIn`` when there is none or
    more than one) and is not registered a second time, so it adds no
    state_dict keys; ``critic_extractor`` is ``None``. ``value_net`` is
    built directly at the value sink's width.

    Edges decide gradient flow: ``actor_features_detached`` is ``False`` and
    every ``stop_gradient_actor`` argument is accepted and ignored. A
    stateful graph serves only the recurrent methods (``act_recurrent``,
    ``evaluate_actions_sequence``, ...), a stateless graph only the flat ones
    (``forward``, ``evaluate_actions``, ...); the other family raises
    ``TypeError``. The recurrent hidden state is a flat tuple of tensors
    (batch at dim 1), see ``rl_garden.networks.graph.executor``.

    Parameter order (it fixes the summation order of the global grad-norm
    clip) is ``actor``, ``value_net``, then the graph nodes; reference
    policies put ``actor_extractor`` first, which matters only when an
    encoder has parameters.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *,
        graph: GraphSource,
        **ppo_head_kwargs: Any,
    ) -> None:
        spec = GraphSpec.parse(graph)
        modules, dims = build_graph_nodes(spec, observation_space, action_space)
        self._reject_structured_features_into_stateful_nodes(spec, modules)
        actor_dim, value_dim = dims[spec.actor_sink], dims[spec.value_sink]
        actor_extractor = self._actor_path_encoder(spec, modules) or GraphFeaturesStandIn(
            observation_space, actor_dim
        )
        value_dim_carrier = (
            None if value_dim == actor_dim else GraphFeaturesStandIn(observation_space, value_dim)
        )
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            features_dim=actor_dim,
            critic_extractor=value_dim_carrier,
            encoder_sharing="shared",
            **ppo_head_kwargs,
        )
        self._keep_unregistered("actor_extractor", actor_extractor)
        self._keep_unregistered("critic_extractor", None)
        self.graph_nodes = nn.ModuleDict(modules)
        self.graph_spec = spec
        self._actor_dim = actor_dim
        self._value_dim = value_dim
        self._state_layout = StateLayout.from_nodes(spec, self.graph_nodes)
        #: Per-env layout of the losses' ``info.<key>`` targets, keyed by
        #: ``<key>`` (what the rollout buffer stores as extras).
        self.info_target_specs: TargetSpecs = info_target_specs(spec, dims)

    @staticmethod
    def _reject_structured_features_into_stateful_nodes(
        spec: GraphSpec, modules: dict[str, nn.Module]
    ) -> None:
        """``SequencePPO``'s guard, per stateful node: sequence stages take
        flat latents only."""
        for stateful in spec.stateful_nodes:
            for name in sorted(spec.ancestors(stateful)):
                module = modules.get(name)
                if (
                    isinstance(module, BaseFeaturesExtractor)
                    and module.structured_feature_config() is not None
                ):
                    raise NotImplementedError(
                        f"GraphPPOPolicy: encoder node {name!r} feeds stateful node "
                        f"{stateful!r}, which only supports flat-latent feature "
                        "extractors this round (ViT token_and_prop layouts untested)."
                    )

    @staticmethod
    def _actor_path_encoder(spec: GraphSpec, modules: dict[str, nn.Module]):
        encoders = [
            modules[name]
            for name in sorted(spec.ancestors(spec.actor_sink))
            if isinstance(modules.get(name), BaseFeaturesExtractor)
        ]
        return encoders[0] if len(encoders) == 1 else None

    def _keep_unregistered(self, name: str, value: Optional[nn.Module]) -> None:
        """Keep ``value`` reachable as ``self.<name>`` without registering it
        as a submodule (the graph already owns it)."""
        self._modules.pop(name, None)
        object.__setattr__(self, name, value)

    # --- extractor contract ---------------------------------------------------

    @property
    def is_stateful(self) -> bool:
        return self.graph_spec.is_stateful

    @property
    def actor_features_dim(self) -> int:
        return self._actor_dim

    @property
    def critic_features_dim(self) -> int:
        return self._value_dim

    @property
    def actor_features_detached(self) -> bool:
        return False

    def update_normalizer(self, obs: Obs) -> None:
        for module in self.graph_nodes.values():
            if isinstance(module, BaseFeaturesExtractor):
                module.update_normalizer(obs)

    def _require_stateless(self, method: str) -> None:
        if self.is_stateful:
            raise TypeError(
                f"GraphPPOPolicy.{method} needs a stateless graph, but this graph has "
                f"stateful node(s) {list(self.graph_spec.stateful_nodes)}; use the "
                "recurrent methods (act_recurrent, evaluate_actions_sequence, ...)."
            )

    def _require_stateful(self, method: str) -> None:
        if not self.is_stateful:
            raise TypeError(
                f"GraphPPOPolicy.{method} needs a stateful graph (a gru/lstm/gtrxl "
                "node), but this graph has none; use the flat methods (forward, "
                "evaluate_actions, ...)."
            )

    def _flat_sink_input(self, obs: Obs, sink: str, method: str) -> torch.Tensor:
        self._require_stateless(method)
        return run_flat(self.graph_spec, self.graph_nodes, obs, (sink,))[sink]

    def _flat_evaluation_latents(
        self, obs: Obs, method: str
    ) -> tuple[torch.Tensor, torch.Tensor, GraphOutputs]:
        """Actor and value sink inputs plus the graph outputs, computed as
        ``PPOPolicy`` computes its features: one run for the actor sink, then
        one for the value sink (so training stays bit-identical to it). The
        actor run also computes the loss nodes; the outputs dict is that
        run's, plus the value sink's input."""
        self._require_stateless(method)
        spec = self.graph_spec
        outputs = run_flat(spec, self.graph_nodes, obs, (spec.actor_sink, *spec.loss_nodes))
        value_latent = self.extract_critic_features(obs)
        outputs[spec.value_sink] = value_latent
        return outputs[spec.actor_sink], value_latent, outputs

    def extract_actor_features(self, obs: Obs) -> torch.Tensor:
        return self._flat_sink_input(obs, self.graph_spec.actor_sink, "extract_actor_features")

    def extract_critic_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        features = self._flat_sink_input(
            obs, self.graph_spec.value_sink, "extract_critic_features"
        )
        return features.detach() if stop_gradient else features

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        features = self.extract_actor_features(obs)
        return features.detach() if stop_gradient else features

    def _actor_role_features(self, obs: Obs, stop_gradient_actor: Optional[bool]) -> torch.Tensor:
        del stop_gradient_actor  # edges decide
        return self.extract_actor_features(obs)

    # --- flat evaluation (PPOPolicy contract plus the graph outputs) ---------------

    def evaluate_actions(
        self,
        obs: Obs,
        actions: torch.Tensor,
        *,
        stop_gradient_actor: bool | None = None,
        sum_dims: bool = True,
        return_graph_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """``PPOPolicy.evaluate_actions``; with ``return_graph_outputs`` the
        graph outputs dict is appended to the returned tuple."""
        if not return_graph_outputs:
            return super().evaluate_actions(
                obs, actions, stop_gradient_actor=stop_gradient_actor, sum_dims=sum_dims
            )
        actor_latent, value_latent, outputs = self._flat_evaluation_latents(
            obs, "evaluate_actions"
        )
        log_prob, entropy = self.actor.evaluate_action_log_prob(
            actor_latent, actions, sum_dims=sum_dims
        )
        values = self.value_net(value_latent)
        return values, log_prob, entropy, outputs

    def evaluate_actions_with_dist_params(
        self,
        obs: Obs,
        actions: torch.Tensor,
        *,
        stop_gradient_actor: bool | None = None,
        sum_dims: bool = True,
        return_graph_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """``PPOPolicy.evaluate_actions_with_dist_params``; with
        ``return_graph_outputs`` the graph outputs dict is appended."""
        if not return_graph_outputs:
            return super().evaluate_actions_with_dist_params(
                obs, actions, stop_gradient_actor=stop_gradient_actor, sum_dims=sum_dims
            )
        actor_latent, value_latent, outputs = self._flat_evaluation_latents(
            obs, "evaluate_actions_with_dist_params"
        )
        dist = self.actor(actor_latent)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if sum_dims:
            log_prob = log_prob.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        values = self.value_net(value_latent)
        log_std = self.actor.log_std.expand_as(dist.mean)
        return values, log_prob, entropy, dist.mean, log_std, outputs

    # --- hidden state -------------------------------------------------------------

    def initial_hidden_state(self, batch_size: int, device) -> GraphState:
        self._require_stateful("initial_hidden_state")
        return self._state_layout.initial(self.graph_nodes, batch_size, device)

    def mask_hidden_state(self, state: GraphState, keep_mask: torch.Tensor) -> GraphState:
        return self._state_layout.mask(self.graph_nodes, state, keep_mask)

    def index_hidden_state(self, state: GraphState, indices: torch.Tensor) -> GraphState:
        return self._state_layout.index(self.graph_nodes, state, indices)

    def _step_latents(
        self, obs: Obs, hidden: GraphState, episode_starts: torch.Tensor, method: str
    ) -> tuple[torch.Tensor, torch.Tensor, GraphState]:
        self._require_stateful(method)
        spec = self.graph_spec
        outputs, new_state = run_step(
            spec,
            self.graph_nodes,
            obs,
            self._state_layout.split(hidden),
            episode_starts,
            (spec.actor_sink, spec.value_sink),
        )
        return (
            outputs[spec.actor_sink],
            outputs[spec.value_sink],
            self._state_layout.join(new_state),
        )

    def _sequence_latents(
        self,
        obs: Obs,
        initial_hidden: GraphState,
        episode_starts: torch.Tensor,
        method: str,
        *,
        with_loss_nodes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, GraphOutputs]:
        """Actor and value sink inputs of a ``(T, B)`` window, flattened to
        ``(T * B, D)``, and the outputs dict of that run (T-major rows); the
        loss nodes run too ``with_loss_nodes``."""
        self._require_stateful(method)
        spec = self.graph_spec
        loss_nodes = spec.loss_nodes if with_loss_nodes else ()
        outputs, _ = run_sequence(
            spec,
            self.graph_nodes,
            obs,
            self._state_layout.split(initial_hidden),
            episode_starts,
            (spec.actor_sink, spec.value_sink, *loss_nodes),
        )
        return outputs[spec.actor_sink], outputs[spec.value_sink], outputs

    # --- recurrent methods (same contract as RecurrentPPOPolicy) -------------------

    def act_recurrent(
        self,
        obs: Obs,
        hidden: GraphState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
        *,
        stop_gradient_actor: bool = False,
        sum_dims: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, GraphState]:
        del stop_gradient_actor  # edges decide
        actor_latent, value_latent, new_hidden = self._step_latents(
            obs, hidden, episode_starts, "act_recurrent"
        )
        actions, log_prob, entropy = self.actor.action_log_prob(
            actor_latent, deterministic=deterministic, sum_dims=sum_dims
        )
        values = self.value_net(value_latent)
        return actions, values, log_prob, entropy, new_hidden

    def act_recurrent_with_dist_params(
        self,
        obs: Obs,
        hidden: GraphState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
        *,
        stop_gradient_actor: bool = False,
        sum_dims: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        GraphState,
        torch.Tensor,
        torch.Tensor,
    ]:
        del stop_gradient_actor  # edges decide
        actor_latent, value_latent, new_hidden = self._step_latents(
            obs, hidden, episode_starts, "act_recurrent_with_dist_params"
        )
        dist = self.actor(actor_latent)
        actions = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if sum_dims:
            log_prob = log_prob.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        values = self.value_net(value_latent)
        log_std = self.actor.log_std.expand_as(dist.mean)
        return actions, values, log_prob, entropy, new_hidden, dist.mean, log_std

    def predict_recurrent(
        self,
        obs: Obs,
        hidden: GraphState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, GraphState]:
        actor_latent, _, new_hidden = self._step_latents(
            obs, hidden, episode_starts, "predict_recurrent"
        )
        if deterministic:
            action = self.actor.clamp_action(self.actor.deterministic_action(actor_latent))
        else:
            action, _, _ = self.actor.action_log_prob(actor_latent, deterministic=False)
            action = self.actor.clamp_action(action)
        return action, new_hidden

    def predict_values_recurrent(
        self, obs: Obs, hidden: GraphState, episode_starts: torch.Tensor
    ) -> tuple[torch.Tensor, GraphState]:
        _, value_latent, new_hidden = self._step_latents(
            obs, hidden, episode_starts, "predict_values_recurrent"
        )
        return self.value_net(value_latent), new_hidden

    def evaluate_actions_sequence(
        self,
        obs: Obs,
        actions: torch.Tensor,
        initial_hidden: GraphState,
        episode_starts: torch.Tensor,
        *,
        stop_gradient_actor: bool = False,
        sum_dims: bool = True,
        return_graph_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """BPTT update path. obs/actions: (T, B, ...). Returns (values,
        log_prob, entropy), each (T, B, 1); with ``sum_dims=False``
        ``log_prob``/``entropy`` are (T, B, A). With ``return_graph_outputs``
        the graph outputs dict (``(T * B, ...)`` rows, T-major) is appended."""
        del stop_gradient_actor  # edges decide
        num_steps, num_envs = actions.shape[0], actions.shape[1]
        actor_latent, value_latent, outputs = self._sequence_latents(
            obs,
            initial_hidden,
            episode_starts,
            "evaluate_actions_sequence",
            with_loss_nodes=return_graph_outputs,
        )
        flat_actions = actions.reshape(num_steps * num_envs, -1)
        log_prob, entropy = self.actor.evaluate_action_log_prob(
            actor_latent, flat_actions, sum_dims=sum_dims
        )
        values = self.value_net(value_latent)
        evaluated = (
            values.reshape(num_steps, num_envs, -1),
            log_prob.reshape(num_steps, num_envs, -1),
            entropy.reshape(num_steps, num_envs, -1),
        )
        return (*evaluated, outputs) if return_graph_outputs else evaluated

    def evaluate_actions_sequence_with_dist_params(
        self,
        obs: Obs,
        actions: torch.Tensor,
        initial_hidden: GraphState,
        episode_starts: torch.Tensor,
        *,
        stop_gradient_actor: bool = False,
        sum_dims: bool = True,
        return_graph_outputs: bool = False,
    ) -> tuple[Any, ...]:
        """``evaluate_actions_sequence`` plus the ``(mean, log_std)`` of the
        current policy, each (T, B, A)."""
        del stop_gradient_actor  # edges decide
        num_steps, num_envs = actions.shape[0], actions.shape[1]
        actor_latent, value_latent, outputs = self._sequence_latents(
            obs,
            initial_hidden,
            episode_starts,
            "evaluate_actions_sequence_with_dist_params",
            with_loss_nodes=return_graph_outputs,
        )
        flat_actions = actions.reshape(num_steps * num_envs, -1)
        dist = self.actor(actor_latent)
        log_prob = dist.log_prob(flat_actions)
        entropy = dist.entropy()
        if sum_dims:
            log_prob = log_prob.sum(-1, keepdim=True)
            entropy = entropy.sum(-1, keepdim=True)
        values = self.value_net(value_latent)
        log_std = self.actor.log_std.expand_as(dist.mean)
        evaluated = (
            values.reshape(num_steps, num_envs, -1),
            log_prob.reshape(num_steps, num_envs, -1),
            entropy.reshape(num_steps, num_envs, -1),
            dist.mean.reshape(num_steps, num_envs, -1),
            log_std.reshape(num_steps, num_envs, -1),
        )
        return (*evaluated, outputs) if return_graph_outputs else evaluated
