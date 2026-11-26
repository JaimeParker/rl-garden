"""--help text must belong to the field it is shown for.

tyro attaches a comment block above a dataclass field to every following field
until the next blank line, comment or docstring, so a comment written for one
field silently becomes the help text of its neighbours. Fields may share a help
text only when the comment really describes all of them (listed below).
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import typing

from tyro import _docstrings

from rl_garden.training.evaluation import EvalCheckpointArgs
from rl_garden.training.off2on.registry import registry as off2on_registry
from rl_garden.training.offline.registry import registry as offline_registry
from rl_garden.training.online.registry import registry as online_registry

_SAME_TOPIC = "one comment describes the whole group"

# (class that declares the first field, fields sharing one help text) -> reason
ALLOWED_SHARED_HELP: dict[tuple[str, tuple[str, ...]], str] = {
    ("SACTrainingArgs", ("target_policy_noise", "target_noise_clip", "actor_q_agg")): (
        "the TD3-shaped options of SAC's update are described together"
    ),
    ("RLPDArgs", ("reward_scale", "reward_bias")): (
        "one affine reward transform shared by the live env and the offline loader"
    ),
    ("Off2OnCommonArgs", ("offline_eval_freq", "online_eval_freq")): (
        "the two phase cadences are described together"
    ),
    ("CQLOff2OnArgs", ("use_sarsa_reference", "sarsa_hidden_dims", "sarsa_lr")): (
        "the SARSA reference network and its knobs"
    ),
    ("OfflineCalQLArgs", ("use_sarsa_reference", "sarsa_hidden_dims", "sarsa_lr")): (
        "the SARSA reference network and its knobs"
    ),
    ("OfflineFloQArgs", ("r_min", "r_max")): "the reward range",
    ("OfflineValueFlowsArgs", ("min_reward", "max_reward")): "the reward range",
    (
        "OfflineQGFArgs",
        ("actor_use_layer_norm", "critic_use_layer_norm", "value_use_layer_norm"),
    ): "the reference applies layer norm to every network",
    ("OfflineQGFArgs", ("robust_critic_lr", "robust_critic_t_emb_size")): (
        "both configure the RobustQ critic"
    ),
    (
        "OfflineQAMArgs",
        ("actor_use_layer_norm", "critic_use_layer_norm", "value_use_layer_norm"),
    ): "the per-network layer-norm flags mirror one reference setting",
    ("TDMPC2MultitaskTrainingArgs", ("tasks", "max_episodes_per_task")): (
        "both limit which manifest data is loaded"
    ),
    (
        "CalQLOff2OnArgs",
        (
            "hidden_dim",
            "actor_hidden_layers",
            "critic_hidden_layers",
            "policy_log_std_multiplier",
            "policy_log_std_offset",
        ),
    ): "the official Cal-QL network shape and log-std transform",
    ("WSRLOff2OnArgs", ("hidden_dim", "actor_hidden_layers", "critic_hidden_layers")): (
        "the asymmetric depth knobs replace net_arch together"
    ),
    ("FACOff2OnArgs", ("fac_alpha", "fac_lambda", "fac_threshold")): (
        "defaults documented together in the FAC module docstring"
    ),
    ("FINOOff2OnArgs", ("noise_scale", "beta", "num_samples")): (
        "defaults documented together in the FINO module docstring"
    ),
    (
        "FloQOff2OnArgs",
        (
            "r_min",
            "r_max",
            "flow_num_ensembles",
            "noise_samples",
            "noise_coverage",
            "critic_flow_steps",
            "train_at_zero_only",
            "embed_time",
            "time_embed_dim",
            "use_prob_embed",
            "num_bins",
            "sigma",
            "reward_offset",
        ),
    ): "defaults documented together in the FloQ module docstring",
    (
        "ValueFlowsOff2OnArgs",
        (
            "min_reward",
            "max_reward",
            "ret_agg",
            "confidence_weight_temp",
            "dcfm_lambda",
            "bcfm_lambda",
            "clip_flow_returns",
            "num_samples",
            "policy_extraction",
        ),
    ): "defaults documented together in the Value Flows module docstring",
    ("EvalCheckpointArgs", ("env_backend", "env_id")): (
        "None keeps the run's value for both"
    ),
}


def _dataclasses_reachable_from(root: type) -> list[type]:
    found: dict[type, None] = {}

    def visit(cls: type) -> None:
        if cls in found or not dataclasses.is_dataclass(cls):
            return
        found[cls] = None
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            hint = hints[field.name]
            for candidate in (hint, *typing.get_args(hint)):
                if isinstance(candidate, type):
                    visit(candidate)

    visit(root)
    return list(found)


@functools.cache
def _all_args_classes() -> list[type]:
    roots: list[type] = [EvalCheckpointArgs]
    for registry in (online_registry, offline_registry, off2on_registry):
        registry.discover()
        roots.extend(entry.args_cls for entry in registry.entries().values())
    classes: dict[type, None] = {}
    for root in roots:
        for cls in _dataclasses_reachable_from(root):
            classes[cls] = None
    return list(classes)


def _declaring_class(cls: type, field_name: str) -> type:
    return next(
        base
        for base in cls.__mro__
        if field_name in base.__dict__.get("__annotations__", {})
    )


@functools.cache
def _help_text(cls: type, field_name: str) -> str | None:
    return _docstrings.get_field_docstring(cls, field_name, ())


@functools.cache
def _shared_help_groups() -> dict[tuple[str, tuple[str, ...]], tuple[str, set[str]]]:
    """(declaring class, field names) -> (help text, leaf classes showing it)."""
    groups: dict[tuple[str, tuple[str, ...]], tuple[str, set[str]]] = {}
    for cls in _all_args_classes():
        by_text: dict[str, list[str]] = collections.defaultdict(list)
        for field in dataclasses.fields(cls):
            text = _help_text(cls, field.name)
            if text:
                by_text[text].append(field.name)
        for text, names in by_text.items():
            if len(names) < 2:
                continue
            key = (_declaring_class(cls, names[0]).__name__, tuple(names))
            groups.setdefault(key, (text, set()))[1].add(cls.__name__)
    return groups


def test_no_help_text_is_shared_by_unrelated_fields():
    unexpected = {
        key: (text[:80], sorted(users)[:3])
        for key, (text, users) in _shared_help_groups().items()
        if key not in ALLOWED_SHARED_HELP
    }
    assert not unexpected, (
        "fields share one help text; put a blank line after the last field the "
        f"comment describes, or allow the group with a reason: {unexpected}"
    )


def test_allowed_shared_help_groups_are_current():
    present = set(_shared_help_groups())
    stale = sorted(set(ALLOWED_SHARED_HELP) - present)
    assert not stale, f"allowlist entries that no longer match a group: {stale}"


def test_section_headings_are_not_help_text():
    headings = [
        (cls.__name__, field.name, _help_text(cls, field.name))
        for cls in _all_args_classes()
        for field in dataclasses.fields(cls)
        if (_help_text(cls, field.name) or "").startswith("---")
    ]
    assert not headings, f"section heading shown as help text: {headings}"


def test_eval_cadence_fields_document_their_unit():
    missing = [
        (cls.__name__, name)
        for cls in _all_args_classes()
        for name in ("eval_freq", "num_eval_steps")
        if name in {field.name for field in dataclasses.fields(cls)}
        and not _help_text(cls, name)
    ]
    assert not missing, f"eval_freq/num_eval_steps without help text: {missing}"
