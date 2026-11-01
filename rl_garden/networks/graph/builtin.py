"""The built-in policy graphs: the trunks of ``PPOPolicy`` and
``RecurrentPPOPolicy`` written as graphs.

Built with the same seed and head arguments, ``GraphPPOPolicy`` on these
graphs has the same parameters as the reference policy under the key
renames ``graph_nodes.encoder. -> actor_extractor.`` (``actor_encoder``/
``critic_encoder`` -> ``actor_extractor``/``critic_extractor`` for
``"separate"``) and ``graph_nodes.<gru|lstm|gtrxl>. -> recurrent_encoder.``,
and computes the same outputs. Selectable on the CLI as
``--policy-graph builtin:<mlp|recurrent|gru|lstm|gtrxl>``: ``recurrent``
takes the algorithm's own stage; ``gru``/``lstm``/``gtrxl`` must agree with
it.

``"shared_critic_grad"`` becomes ``detach: true`` on the edge into the
actor sink, a full detach of that latent. ``PPOPolicy`` with an image
``CombinedExtractor`` detaches only the image branches there (the proprio /
vector branches still get the actor gradient), so the built-in ``mlp`` graph
reproduces it exactly only for state observations. On image observations
under ``"shared_critic_grad"`` the two objectives differ, and ``PPO`` warns
when it builds ``builtin:mlp`` there.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from rl_garden.common.effective_config import ConfigError
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.graph.spec import BUILTIN_GRAPH_PREFIX, GraphSpec

SEQUENCE_KINDS = ("gru", "lstm", "gtrxl")
BUILTIN_GRAPHS = ("mlp",) + SEQUENCE_KINDS
#: ``builtin:recurrent`` takes the sequence stage the algorithm is configured
#: for (``RecurrentPPO``'s ``rnn_type``, ``TransformerPPO``'s gtrxl).
CONFIGURED_STAGE = "recurrent"
BUILTIN_NAMES = BUILTIN_GRAPHS + (CONFIGURED_STAGE,)


def _encoder_node(encoder_config: Optional[EncoderConfig]) -> dict[str, Any]:
    config = encoder_config if encoder_config is not None else EncoderConfig()
    return {"type": "encoder", "keys": None, "config": dataclasses.asdict(config)}


def mlp_actor_critic(
    encoder_config: Optional[EncoderConfig] = None,
    encoder_sharing: str = "shared_critic_grad",
) -> GraphSpec:
    """``PPOPolicy``'s trunk: one encoder feeding both heads, or two
    (``actor_encoder``/``critic_encoder``) for ``"separate"``."""
    if encoder_sharing == "separate":
        nodes = {
            "obs": {"type": "input"},
            "actor_encoder": _encoder_node(encoder_config),
            "critic_encoder": _encoder_node(encoder_config),
            "actor": {"type": "actor"},
            "value": {"type": "value"},
        }
        edges = [
            {"from": "obs", "to": "actor_encoder"},
            {"from": "obs", "to": "critic_encoder"},
            {"from": "actor_encoder", "to": "actor"},
            {"from": "critic_encoder", "to": "value"},
        ]
        return GraphSpec.parse({"version": 1, "nodes": nodes, "edges": edges})
    return _single_trunk_graph(encoder_config, encoder_sharing, stage=None)


def recurrent_actor_critic(
    kind: str,
    encoder_config: Optional[EncoderConfig] = None,
    encoder_sharing: str = "shared_critic_grad",
    **stage_params: Any,
) -> GraphSpec:
    """``RecurrentPPOPolicy``'s trunk: encoder -> ``kind`` stage -> both
    heads. ``stage_params`` are the ``gru``/``lstm``/``gtrxl`` node
    parameters (defaults when omitted)."""
    if kind not in SEQUENCE_KINDS:
        raise ConfigError(f"unknown sequence stage {kind!r}; choose from {list(SEQUENCE_KINDS)}.")
    if encoder_sharing == "separate":
        raise ConfigError(
            "the recurrent built-in graph has one sequence stage shared by both heads; "
            "encoder_sharing='separate' needs a custom graph."
        )
    return _single_trunk_graph(encoder_config, encoder_sharing, stage=(kind, stage_params))


def _single_trunk_graph(
    encoder_config: Optional[EncoderConfig],
    encoder_sharing: str,
    stage: Optional[tuple[str, dict[str, Any]]],
) -> GraphSpec:
    if encoder_sharing not in ("shared_critic_grad", "shared"):
        raise ConfigError(f"unknown encoder_sharing {encoder_sharing!r}.")
    nodes: dict[str, Any] = {
        "obs": {"type": "input"},
        "encoder": _encoder_node(encoder_config),
        "actor": {"type": "actor"},
        "value": {"type": "value"},
    }
    edges = [{"from": "obs", "to": "encoder"}]
    head_input = "encoder"
    if stage is not None:
        kind, params = stage
        nodes[kind] = {"type": kind, **params}
        edges.append({"from": "encoder", "to": kind})
        head_input = kind
    edges += [
        {"from": head_input, "to": "actor", "detach": encoder_sharing == "shared_critic_grad"},
        {"from": head_input, "to": "value"},
    ]
    return GraphSpec.parse({"version": 1, "nodes": nodes, "edges": edges})


def builtin_graph_name(source: str) -> str:
    """``"builtin:gru"`` -> ``"gru"``; raises for an unknown name."""
    name = source[len(BUILTIN_GRAPH_PREFIX):]
    if name not in BUILTIN_NAMES:
        raise ConfigError(
            f"unknown built-in policy graph {source!r}; choose from "
            f"{[BUILTIN_GRAPH_PREFIX + graph for graph in BUILTIN_NAMES]}."
        )
    return name


def builtin_graph_is_stateful(name: str) -> bool:
    return name != "mlp"


def resolve_builtin_stage(
    name: str, configured: Optional[tuple[str, str]], algorithm: str
) -> Optional[str]:
    """Sequence-stage kind of the built-in graph ``name`` (``None`` for
    ``mlp``) under ``algorithm``, whose own stage is ``configured`` (``(kind,
    setting that decides it)``, ``None`` for a stateless algorithm). A
    built-in name that disagrees with the algorithm's stage raises."""
    if name == "mlp":
        return None
    if configured is None:
        if name == CONFIGURED_STAGE:
            raise ConfigError(
                f"policy graph 'builtin:{CONFIGURED_STAGE}' takes the algorithm's sequence "
                f"stage, but {algorithm} has none (use recurrent_ppo/transformer_ppo)."
            )
        return name
    kind, setting = configured
    if name not in (CONFIGURED_STAGE, kind):
        raise ConfigError(
            f"policy graph 'builtin:{name}' builds a {name} stage, but {algorithm} is "
            f"configured for a {kind} stage ({setting}); use "
            f"'builtin:{CONFIGURED_STAGE}' or make the two agree."
        )
    return kind


def builtin_graph(
    name: str,
    *,
    encoder_config: Optional[EncoderConfig] = None,
    encoder_sharing: str = "shared_critic_grad",
    stage_params: Optional[dict[str, Any]] = None,
) -> GraphSpec:
    if name == "mlp":
        return mlp_actor_critic(encoder_config, encoder_sharing)
    return recurrent_actor_critic(name, encoder_config, encoder_sharing, **(stage_params or {}))
