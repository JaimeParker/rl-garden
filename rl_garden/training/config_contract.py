"""Declarative ownership and consumption contracts for training arguments."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

from rl_garden.common.cli_args import CheckpointArgs, LoggingArgs, VisionArgs
from rl_garden.common.effective_config import ConfigError, FieldProvenance, json_value
from rl_garden.common.env_args import EnvBackendArgs

FieldOwner = Literal[
    "environment",
    "agent",
    "runner",
    "logging",
    "checkpoint",
    "derived",
    "unused",
]

_BACKENDS = {
    field.name for field in fields(EnvBackendArgs) if field.name != "env_backend"
}
# Reflected off the canonical mixins instead of hand-copied, so a field added
# to LoggingArgs/CheckpointArgs is classified automatically.
_LOGGING = {field.name for field in fields(LoggingArgs)}
_CHECKPOINT = {field.name for field in fields(CheckpointArgs)} | {
    # Declared directly on SACTrainingArgs, not on the CheckpointArgs mixin.
    "load_actor_checkpoint"
}
_RUNNER = {
    "total_timesteps",
    "num_offline_steps",
    "num_online_steps",
    "save_filename",
    "offline_dataset_path",
    "dataset_dir",
    "dataset_source",
    "num_eval_episodes",
    "eval_episode_horizon",
    "offline_eval_freq",
    "online_eval_freq",
    "offline_num_traj",
    "offline_buffer_size",
    "offline_data_ratio",
    "online_replay_mode",
    "success_key",
}
_ENVIRONMENT = {
    "env_backend",
    "env_id",
    "num_envs",
    "num_eval_envs",
    "spec_num_envs",
    "obs_mode",
    "include_state",
    # frame_stack is dual-consumption: EnvRequest.frame_stack on the
    # environment side (owned here) AND, independently,
    # enable_stacking=args.frame_stack > 1 on the agent side of most visual
    # builders (absent entirely from off2on/wsrl.py's builder -- the agent-side
    # effect isn't even uniform across algorithms). The single-owner
    # FieldRule.consumption model can't represent "one field, two
    # destinations, effect differs per algorithm" without a bigger change; the
    # agent-side effect is intentionally not cross-checked in Batch 2.
    "frame_stack",
    "camera_width",
    "camera_height",
    "per_camera_rgbd",
    "image_keys",
    "control_mode",
    "render_mode",
    "capture_video",
    "video_fps",
    "eval_output_dir",
    "reward_scale",
    "reward_bias",
    "action_low",
    "action_high",
}
# obs_mode drives the active-condition check itself, so it can't gate on its
# own value; everything else declared on VisionArgs is visual-only.
_VISUAL = {field.name for field in fields(VisionArgs)} - {"obs_mode"}
_RESNET_ONLY = {"pretrained_weights", "freeze_resnet_encoder", "freeze_resnet_backbone"}
_PLAIN_CONV_ONLY = {
    "plain_conv_weight_init",
    "plain_conv_last_act",
    "plain_conv_pooling",
}


@dataclass(frozen=True)
class ActiveCondition:
    """Closed set of active-when conditions -- construction fails on any kind
    not listed below instead of a check silently defaulting to active."""

    kind: Literal[
        "always",
        "env_backend_is",
        "visual_observation",
        "encoder_starts_with",
        "encoder_is",
    ]
    value: str | None = None

    def __post_init__(self) -> None:
        value_required = self.kind in {
            "env_backend_is",
            "encoder_starts_with",
            "encoder_is",
        }
        if self.kind not in {
            "always",
            "env_backend_is",
            "visual_observation",
            "encoder_starts_with",
            "encoder_is",
        }:
            raise ValueError(f"Unknown ActiveCondition kind {self.kind!r}")
        if value_required and self.value is None:
            raise ValueError(f"ActiveCondition {self.kind!r} requires a value")
        if not value_required and self.value is not None:
            raise ValueError(f"ActiveCondition {self.kind!r} does not accept a value")

    def evaluate(self, args: Any) -> bool:
        if self.kind == "always":
            return True
        if self.kind == "env_backend_is":
            return getattr(args, "env_backend", None) == self.value
        if self.kind == "visual_observation":
            return getattr(args, "obs_mode", "state") != "state"
        if self.kind == "encoder_starts_with":
            return getattr(args, "obs_mode", "state") != "state" and str(
                getattr(args, "encoder", "")
            ).startswith(self.value)
        if self.kind == "encoder_is":
            return (
                getattr(args, "obs_mode", "state") != "state"
                and getattr(args, "encoder", None) == self.value
            )
        raise AssertionError(f"Unhandled ActiveCondition kind {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "always":
            return "always"
        if self.kind == "env_backend_is":
            return f"env_backend == {self.value!r}"
        if self.kind == "visual_observation":
            return "obs_mode != 'state'"
        if self.kind == "encoder_starts_with":
            return f"obs_mode != 'state' and encoder starts with {self.value!r}"
        if self.kind == "encoder_is":
            return f"obs_mode != 'state' and encoder == {self.value!r}"
        raise AssertionError(f"Unhandled ActiveCondition kind {self.kind!r}")


@dataclass(frozen=True)
class DirectKwarg:
    """Field value reaches the constructor unchanged, under this kwarg name."""

    name: str


@dataclass(frozen=True)
class NestedKwarg:
    """Field value ends up at constructor_kwargs[container][key]."""

    container: str
    key: str


@dataclass(frozen=True)
class BuilderDerived:
    """Field feeds a builder-local computation (e.g. an encoder factory
    closure) that collapses several Args fields into kwargs whose names don't
    map 1:1 back to any single field. Checked by presence of any kwarg name in
    the named cluster, never by value -- see _CONSUMPTION_CLUSTERS."""

    cluster: str


Consumption = DirectKwarg | NestedKwarg | BuilderDerived

# Presence-only kwarg-name clusters for BuilderDerived fields. Enumerated by
# reading image_encoder_factory_from_args/image_keys_from_env/
# vit_sac_kwargs_from_args (rl_garden/common/cli_args.py) and every visual
# builder's image_kwargs dict literal, plus sac_initial_training_phase_from_args
# (rl_garden/training/online/_args.py) and initial_training_phase_from_args
# (rl_garden/training/off2on/_args.py).
_CONSUMPTION_CLUSTERS: dict[str, frozenset[str]] = {
    "visual_encoder": frozenset(
        {
            "image_encoder_factory",
            "image_keys",
            "image_fusion_mode",
            "enable_stacking",
            "image_augmentation",
            "random_shift_pad",
            "image_augmentation_seed",
            "state_key",
            "use_proprio",
            "policy_kwargs",
            "actor_feature_dim",
            "critic_spatial_emb_dim",
        }
    ),
    "initial_training_phase": frozenset({"initial_training_phase"}),
    "net_arch": frozenset({"net_arch"}),
    "base_action_provider": frozenset({"base_action_provider"}),
}

# Algorithms confirmed (by reading every builder) to fold actor_hidden_layers/
# critic_hidden_layers/hidden_dim into a single net_arch={"pi": [...], "qf":
# [...]} constructor kwarg, replacing the previous "SAC" in target substring
# heuristic (which happened to not misfire today only because DrQv2's target,
# rl_garden.algorithms.ddpg.DDPG, doesn't contain "SAC" -- but nothing enforced
# that).
_NET_ARCH_FOLD_TARGETS = frozenset(
    {
        "rl_garden.algorithms.SAC",
        "rl_garden.algorithms.RLPD",
        "rl_garden.algorithms.RLPDHybrid",
        "rl_garden.algorithms.ResidualSAC",
        "rl_garden.algorithms.RecurrentSAC",
        "rl_garden.algorithms.TransformerSAC",
        "rl_garden.algorithms.CQL",
        "rl_garden.algorithms.CalQL",
        "rl_garden.algorithms.WSRL",
        "rl_garden.algorithms.Off2OnCalQL",
    }
)

_BASE_ACTION_PROVIDER_FIELDS = frozenset(
    {
        "base_policy",
        "base_ckpt_path",
        "base_act_temporal_agg",
        "base_act_temporal_agg_k",
        "base_sac_deterministic",
        "base_sac_encoder",
        "base_sac_encoder_features_dim",
        "base_sac_image_fusion_mode",
        "debug",
    }
)

# Fields inherited for CLI compatibility but not consumed by these algorithms.
# Keeping the exceptions target-scoped avoids pretending they are constructor
# parameters while preserving the existing public CLI surface.
_TARGET_UNUSED_FIELDS: dict[str, frozenset[str]] = {
    "rl_garden.algorithms.Off2OnAWAC": frozenset(
        {"critic_subsample_size", "std_parameterization"}
    ),
    "rl_garden.algorithms.AWAC": frozenset({"utd"}) | _VISUAL,
    "rl_garden.algorithms.TD3BC": frozenset({"utd"}) | _VISUAL,
}

# critic_only_* only exists on the online SAC family, combined into a single
# InitialTrainingPhase constructor kwarg by sac_initial_training_phase_from_args.
# warmup_steps is the off2on-only analogue, combined by
# initial_training_phase_from_args -- distinct field, same target cluster.
_INITIAL_TRAINING_PHASE_FIELDS = frozenset(
    {
        "critic_only_steps",
        "critic_only_freeze_encoder",
        "critic_only_random_action_prob",
        "warmup_steps",
    }
)


def check_constructor_coverage(
    consumption_by_path: Mapping[str, Consumption],
    constructor_kwargs: Mapping[str, Any],
    *,
    inactive_clusters: frozenset[str] = frozenset(),
) -> list[str]:
    """Compare declared field consumption against constructor_kwargs actually
    captured by construct_agent(). Builder clusters known to be inapplicable to
    the materialized environment may be excluded explicitly. Returns
    human-readable violation messages; empty means every declaration matched."""
    violations: list[str] = []
    for path, consumption in consumption_by_path.items():
        if isinstance(consumption, DirectKwarg):
            ok = consumption.name in constructor_kwargs
        elif isinstance(consumption, NestedKwarg):
            container = constructor_kwargs.get(consumption.container)
            ok = isinstance(container, Mapping) and consumption.key in container
        elif isinstance(consumption, BuilderDerived):
            if consumption.cluster in inactive_clusters:
                continue
            try:
                cluster = _CONSUMPTION_CLUSTERS[consumption.cluster]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown builder-derived consumption cluster "
                    f"{consumption.cluster!r}"
                ) from exc
            ok = any(name in constructor_kwargs for name in cluster)
        else:
            raise TypeError(f"Unhandled consumption type {consumption!r}")
        if not ok:
            violations.append(
                f"{path!r} declared {consumption!r} but was not found in the "
                "captured constructor kwargs"
            )
    return violations


@dataclass(frozen=True)
class FieldRule:
    path: str
    owner: FieldOwner
    mapped_to: str
    active_when: ActiveCondition = ActiveCondition("always")
    consumption: Consumption | None = None


@dataclass(frozen=True)
class ConfigContract:
    target: str
    fields: Mapping[str, FieldRule]
    derived_parameters: frozenset[str] = frozenset()
    mode: Literal["strict", "passthrough"] = "strict"

    def __post_init__(self) -> None:
        if self.mode not in {"strict", "passthrough"}:
            raise ValueError(f"Unknown config contract mode {self.mode!r}")

    def target_type(self) -> type:
        module_name, _, attribute = self.target.rpartition(".")
        target = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(target, type):
            raise TypeError(f"Config contract target {self.target!r} is not a type")
        return target

    @classmethod
    def for_args(
        cls,
        args_cls: type,
        *,
        target: str,
        derived_parameters: frozenset[str] = frozenset(),
        mode: Literal["strict", "passthrough"] = "strict",
        agent_fields: Mapping[str, str] | None = None,
    ) -> ConfigContract:
        instance = args_cls()
        rules: dict[str, FieldRule] = {}
        explicit_agent_fields = dict(agent_fields or {})
        passthrough = mode == "passthrough"

        def visit(value: Any, prefix: str = "") -> None:
            for field in fields(value):
                path = f"{prefix}.{field.name}" if prefix else field.name
                item = getattr(value, field.name)
                root = path.split(".", 1)[0]
                if root in _BACKENDS or field.name in _ENVIRONMENT:
                    owner: FieldOwner = "environment"
                    mapped_to = f"environment.{path}"
                    active_when = (
                        ActiveCondition("env_backend_is", root)
                        if root in _BACKENDS
                        else ActiveCondition("always")
                    )
                elif field.name in _LOGGING:
                    owner = "logging"
                    mapped_to = f"logger.{field.name}"
                    active_when = ActiveCondition("always")
                elif field.name in _CHECKPOINT:
                    owner = "checkpoint"
                    mapped_to = f"checkpoint.{field.name}"
                    active_when = ActiveCondition("always")
                elif field.name in _RUNNER:
                    owner = "runner"
                    mapped_to = f"runner.{field.name}"
                    active_when = ActiveCondition("always")
                elif field.name in _TARGET_UNUSED_FIELDS.get(target, frozenset()):
                    owner = "unused"
                    mapped_to = f"unused.{target}.{field.name}"
                    active_when = ActiveCondition("always")
                elif path in explicit_agent_fields or not passthrough:
                    owner = "agent"
                    consumption: Consumption | None
                    if path in explicit_agent_fields:
                        mapped_to = explicit_agent_fields[path]
                        consumption = DirectKwarg(mapped_to.rsplit(".", 1)[-1])
                    elif field.name in _VISUAL:
                        mapped_to = f"{target}.visual_encoder"
                        consumption = BuilderDerived("visual_encoder")
                    elif field.name in _INITIAL_TRAINING_PHASE_FIELDS:
                        mapped_to = f"{target}.initial_training_phase"
                        consumption = BuilderDerived("initial_training_phase")
                    elif field.name in _BASE_ACTION_PROVIDER_FIELDS:
                        mapped_to = f"{target}.base_action_provider"
                        consumption = BuilderDerived("base_action_provider")
                    elif field.name == "actor_hidden_layers":
                        mapped_to = f"{target}.net_arch.pi"
                        consumption = (
                            NestedKwarg("net_arch", "pi")
                            if target in _NET_ARCH_FOLD_TARGETS
                            else DirectKwarg("actor_hidden_layers")
                        )
                    elif field.name == "critic_hidden_layers":
                        mapped_to = f"{target}.net_arch.qf"
                        consumption = (
                            NestedKwarg("net_arch", "qf")
                            if target in _NET_ARCH_FOLD_TARGETS
                            else DirectKwarg("critic_hidden_layers")
                        )
                    elif field.name == "hidden_dim" and target in _NET_ARCH_FOLD_TARGETS:
                        mapped_to = f"{target}.net_arch.pi,qf"
                        # Feeds both net_arch["pi"] and net_arch["qf"] at once;
                        # BuilderDerived's presence-only check is the coarsest
                        # type that can represent "one field, two nested
                        # destinations" without a dedicated multi-key type.
                        consumption = BuilderDerived("net_arch")
                    else:
                        mapped_to = f"{target}.{field.name}"
                        consumption = DirectKwarg(field.name)
                    active_when = ActiveCondition("always")
                else:
                    rules[path] = FieldRule(
                        path,
                        "unused",
                        f"unregistered.{path}",
                        ActiveCondition("always"),
                    )
                    if is_dataclass(item) and not isinstance(item, type):
                        visit(item, path)
                    continue
                if field.name in _VISUAL:
                    active_when = ActiveCondition("visual_observation")
                if field.name in _RESNET_ONLY:
                    active_when = ActiveCondition("encoder_starts_with", "resnet")
                elif field.name in _PLAIN_CONV_ONLY:
                    active_when = ActiveCondition("encoder_is", "plain_conv")
                elif field.name.startswith("vit_"):
                    active_when = ActiveCondition("encoder_is", "vit")
                if owner != "agent":
                    consumption = None
                rules[path] = FieldRule(path, owner, mapped_to, active_when, consumption)
                if is_dataclass(item) and not isinstance(item, type):
                    visit(item, path)

        visit(instance)
        return cls(
            target=target,
            fields=rules,
            derived_parameters=derived_parameters,
            mode=mode,
        )

    def apply(
        self, provenance: Mapping[str, FieldProvenance]
    ) -> dict[str, FieldProvenance]:
        from dataclasses import replace

        if set(provenance) != set(self.fields):
            missing = sorted(set(provenance) - set(self.fields))
            stale = sorted(set(self.fields) - set(provenance))
            raise ValueError(
                f"Config contract mismatch: missing={missing}, stale={stale}"
            )
        return {
            path: replace(
                field,
                owner=self.fields[path].owner,
                mapped_to=self.fields[path].mapped_to,
                active_when=self.fields[path].active_when.describe(),
            )
            for path, field in provenance.items()
        }

    def validate_active(
        self,
        args: Any,
        provenance: Mapping[str, FieldProvenance],
    ) -> dict[str, FieldProvenance]:
        from dataclasses import replace

        result: dict[str, FieldProvenance] = {}
        for path, field in provenance.items():
            condition = self.fields[path].active_when
            active = condition.evaluate(args)
            if not active and field.source.kind in {
                "preset",
                "RLG_*",
                "launcher",
                "CLI",
            }:
                raise ConfigError(
                    f"Configuration field {path!r} is inactive unless "
                    f"{condition.describe()}."
                )
            result[path] = replace(field, active=active)
        return result

    def constructor_defaults(self) -> dict[str, Any]:
        from rl_garden.training._constructor_introspection import (
            inspect_constructor_parameters,
        )

        parameters = inspect_constructor_parameters(self.target_type())
        mapped_names: set[str] = set()
        declaration_errors: list[str] = []
        for path, rule in self.fields.items():
            if rule.owner != "agent" or rule.consumption is None:
                continue
            consumption = rule.consumption
            if isinstance(consumption, DirectKwarg):
                declared_names = {consumption.name}
            elif isinstance(consumption, NestedKwarg):
                declared_names = {consumption.container}
            elif isinstance(consumption, BuilderDerived):
                try:
                    cluster = _CONSUMPTION_CLUSTERS[consumption.cluster]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown builder-derived consumption cluster "
                        f"{consumption.cluster!r}"
                    ) from exc
                declared_names = set(cluster)
            else:
                raise TypeError(f"Unhandled consumption type {consumption!r}")
            names = declared_names & set(parameters)
            if not names:
                declaration_errors.append(
                    f"{path!r} declares {consumption!r}, which does not match "
                    f"any constructor parameter on {self.target}"
                )
            mapped_names.update(names)
        if declaration_errors:
            raise ValueError("Invalid config contract:\n" + "\n".join(declaration_errors))

        defaults: dict[str, Any] = {}
        for name, parameter in parameters.items():
            if name in mapped_names or name in self.derived_parameters:
                continue
            if parameter.default is parameter.empty:
                raise ValueError(
                    f"Required constructor parameter {self.target}.{name} is not mapped "
                    "by the config contract."
                )
            defaults[name] = json_value(parameter.default)
        return defaults

    def field_mappings(self, active_paths: set[str] | None = None) -> dict[str, str]:
        return {
            path: rule.mapped_to
            for path, rule in self.fields.items()
            if rule.owner == "agent" and (active_paths is None or path in active_paths)
        }

    def consumption_map(
        self, active_paths: set[str] | None = None
    ) -> dict[str, Consumption]:
        return {
            path: rule.consumption
            for path, rule in self.fields.items()
            if rule.owner == "agent"
            and rule.consumption is not None
            and (active_paths is None or path in active_paths)
        }
