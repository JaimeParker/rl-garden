"""Evaluate a saved rl-garden checkpoint in a live environment.

This is the generic checkpoint evaluation entrypoint (console script
``rl-garden-eval``; ``examples/eval_checkpoint.py`` wraps it): it rebuilds the
agent through the registry entry that trained it, loads model weights without
replay/optimizer state, then runs policy evaluation until the requested number
of completed episodes is collected.

Example:

    rl-garden-eval \
      --phase offline --algorithm calql \
      --checkpoint-path runs/.../calql_offline_pretrained.pt \
      --num-eval-envs 16 --num-eval-episodes 50 \
      --capture-video false --output-json /tmp/calql_eval.json

Training args are rebuilt in layers, each overriding the previous: the run
configuration (``inputs``, then the selected backend's sub-config from
``active_environment``), the checkpoint's hyperparameters, the eval CLI's
environment/device fields, and finally any backend sub-config field set on
the eval CLI to a non-default value (e.g. ``--gymnasium.device cuda:1``).
``--env-backend``, ``--env-id`` and ``--control-mode`` override the run's
values only when passed; left out, the run's values stand. ``--obs.*``
overrides the run's observation config only when the result differs from the
default ``ObservationConfig()``. Backend sub-config fields left at their
default on the eval CLI keep the run's values.

The run configuration is the file ``--config-path`` names, else the one the
checkpoint embeds (``metadata["run_config"]``), else the ``config.json`` of the
checkpoint's run directory. Without any, ``--env-backend`` and ``--env-id`` are
required and every setting the checkpoint does not record takes its Args
default, with a warning.

A checkpoint holds no absolute path. The embedded configuration is therefore
completed from that ``config.json`` when the same run wrote it; a removed field
is otherwise as if never set, which is an error for the env backend's fields
unless their flag is passed.

Evaluation needs the checkpoint and one evaluation environment, nothing else a
run was given: the entry's
``training_only_inputs`` are set to their absent values and no encoder reads a
pretrained-weights file, since the checkpoint holds every weight.

The agent is built by the registry entry's ``build_agent``; an entry
registered without one is looked up by module path
(``rl_garden.training.<phase>.entrypoints.<algorithm>``, its ``build_*``
function).
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.checkpoint import (
    REMOVED_PATH,
    _canonical_algorithm_class,
    holds_removed_path,
    load_checkpoint_file,
    without_local_paths,
)
from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.common.effective_config import config_digest
from rl_garden.common.env_args import (
    EnvBackendArgs,
    ManiSkillConfig,
    MinariConfig,
    MujocoConfig,
    MujocoWarpConfig,
)
from rl_garden.common.eval_metrics import EVAL_METRIC_ALIASES, EVAL_METRIC_DROP
from rl_garden.common.resolved_config import resolved_config_json
from rl_garden.envs.backend_registry import (
    EnvRequest,
    _get_backend,
    make_evaluation_env,
)
from rl_garden.observations import ObservationConfig
from rl_garden.training.inspection import CheckpointEvaluation, checkpoint_evaluation

Phase = Literal["auto", "online", "offline", "off2on"]

_logger = logging.getLogger(__name__)


@dataclass
class EvalCheckpointArgs(EnvBackendArgs):
    checkpoint_path: str = ""
    phase: Phase = "auto"
    algorithm: str = "auto"
    config_path: str | None = None

    # None keeps the run's value (config.json, then checkpoint metadata);
    # a value replaces it.
    env_backend: str | None = None
    env_id: str | None = None
    # Replaces the run's value only when it differs from ObservationConfig():
    # an Optional dataclass would turn ``--obs.*`` into tyro subcommands.
    obs: ObservationConfig = field(default_factory=ObservationConfig)

    control_mode: str | None = None
    render_mode: str = "rgb_array"
    num_eval_envs: int = 16
    num_eval_episodes: int = 50
    # Steps of the env the agent acts in: with a builder that wraps it in action
    # chunking, one step covers a whole chunk. None allows 1000 per episode.
    max_eval_steps: int | None = None

    seed: int = 1
    # Device of the agent, for every algorithm whose constructor takes one.
    device: str = "auto"
    # Evaluation reads no replay buffer, so it is built without storage ("meta").
    # "cpu" or "cuda" allocate the run's full buffer there, for a custom buffer
    # that cannot be built on "meta".
    buffer_device: str = "meta"

    reward_scale: float = 1.0
    reward_bias: float = 0.0

    capture_video: bool = False
    video_fps: int = 30
    eval_output_dir: str | None = None
    output_json: str | None = None
    strict: bool = True

    log_type: str = "none"
    std_log: bool = True

    # QGF-only: sweep the test-time guidance weight without retraining
    # (the paper's entire premise -- train once, sweep this at eval).
    # None leaves whatever the checkpoint's own training run used.
    # No-op for every other algorithm (_set_existing_attr no-ops via hasattr).
    # TODO: EvalCheckpointArgs is accreting algorithm-specific fields like
    # this one directly onto one flat dataclass (already true of several
    # fields above it too). Once enough algorithms need their own eval-time
    # knobs, split this into a shared base + per-algorithm/per-env eval
    # extensions instead of growing this dataclass indefinitely.
    guidance_weight: float | None = None

    # SUPE-only: the OPAL file of a checkpoint that embeds no VAE. None uses the
    # path in the run's configuration.
    opal_checkpoint: str | None = None

    maniskill: ManiSkillConfig = field(default_factory=ManiSkillConfig)
    minari: MinariConfig = field(default_factory=MinariConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)


_PHASES: tuple[str, ...] = ("online", "offline", "off2on")


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _algorithm_from_class(algorithm_class: Any) -> str | None:
    if not isinstance(algorithm_class, str):
        return None
    canonical = _canonical_algorithm_class(algorithm_class)
    name = str(canonical)
    if name.startswith("Off2On"):
        name = name.removeprefix("Off2On")
    known = {
        "AWAC": "awac",
        "BC": "bc",
        "CalQL": "calql",
        "CQL": "cql",
        "DrQv2": "drqv2",
        "FlashSAC": "flash_sac",
        "IQL": "iql",
        "PPO": "ppo",
        "RLPD": "rlpd",
        "SAC": "sac",
        "TD3": "td3",
        "TD3BC": "td3_bc",
        "TDMPC2": "tdmpc2",
        "WSRL": "wsrl",
    }
    if name in known:
        return known[name]
    return _snake_case(name)


def _registry_for_phase(phase: str):
    module = importlib.import_module(f"rl_garden.training.{phase}.registry")
    registry = module.registry
    registry.discover()
    return registry


def _registered_by_class_name(
    phases: tuple[str, ...], algorithm_class: Any
) -> list[tuple[str, str]]:
    """``(phase, name)`` of every registered entry whose algorithm class is
    named ``algorithm_class``; entries without an ``algorithm_cls`` factory
    cannot be matched this way."""
    if not isinstance(algorithm_class, str):
        return []
    class_name = _canonical_algorithm_class(algorithm_class)
    matches: list[tuple[str, str]] = []
    for phase in phases:
        for name, entry in _registry_for_phase(phase).entries().items():
            algorithm_cls = getattr(entry, "algorithm_cls", None)
            if algorithm_cls is not None and algorithm_cls().__name__ == class_name:
                matches.append((phase, name))
    return matches


def _matches_of_training_stage(
    matches: list[tuple[str, str]], checkpoint: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """The ``matches`` that build the checkpoint's own class; of several, those
    of the phase that trains the stage it was saved in (the only phase its
    class is registered in, when it records no stage)."""
    registered = _registered_by_class_name(
        _PHASES, checkpoint.get("metadata", {}).get("algorithm_class")
    )
    own_class = [match for match in matches if match in registered] or matches
    if len(own_class) == 1:
        return own_class
    extra = (checkpoint.get("state") or {}).get("extra") or {}
    if "online_start_step" in extra:
        stage_phases = {"offline" if extra["online_start_step"] is None else "off2on"}
    else:
        stage_phases = {phase for phase, _ in registered}
    return [match for match in own_class if {match[0]} == stage_phases]


def _resolve_phase_and_algorithm(
    requested_phase: str,
    requested_algorithm: str,
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Pick the registry entry that rebuilds the checkpoint's agent.

    Where a flag is ``auto``, the phase and algorithm the embedded run
    configuration selected decide, when they name a registered entry. Otherwise
    the checkpoint's ``algorithm_class`` is matched against the class each
    registered entry builds; only when no entry declares that class does the
    name table in ``_algorithm_from_class`` apply. The recorded training stage
    decides between an offline and an off2on entry that both match.
    """
    metadata = checkpoint.get("metadata", {})
    algorithm_class = metadata.get("algorithm_class")
    selection = (metadata.get("run_config") or {}).get("selection") or {}
    selected = (selection.get("training_phase"), selection.get("algorithm"))
    if (
        selected[0] in _PHASES
        and requested_phase in ("auto", selected[0])
        and requested_algorithm in ("auto", selected[1])
        and selected[1] in _registry_for_phase(selected[0]).entries()
    ):
        return selected
    phases = _PHASES if requested_phase == "auto" else (requested_phase,)

    if requested_algorithm != "auto":
        algorithm = requested_algorithm
        matches = [
            (phase, algorithm)
            for phase in phases
            if algorithm in _registry_for_phase(phase).entries()
        ]
    else:
        matches = _registered_by_class_name(phases, algorithm_class)
        algorithm = matches[0][1] if matches else _algorithm_from_class(algorithm_class)
        if algorithm is None:
            raise SystemExit(
                "Could not infer algorithm from checkpoint metadata; pass --algorithm."
            )
        if not matches:
            matches = [
                (phase, algorithm)
                for phase in phases
                if algorithm in _registry_for_phase(phase).entries()
            ]

    if not matches:
        raise SystemExit(
            f"Algorithm {algorithm!r} is not registered for phase {requested_phase!r}."
        )
    if len(matches) > 1:
        matches = _matches_of_training_stage(matches, checkpoint) or matches
    if len(matches) > 1:
        choices = ", ".join(f"{phase}:{alg}" for phase, alg in matches)
        raise SystemExit(
            f"Checkpoint algorithm {algorithm!r} is ambiguous across registered "
            f"entries ({choices}); pass --phase and --algorithm explicitly."
        )
    return matches[0]


# Inverse of rl_garden.common.effective_config.json_value()'s non-finite-
# float encoding: config.json is written with json.dumps(..., allow_nan=
# False), so json_value() cannot emit raw Infinity/-Infinity/NaN JSON tokens
# and instead writes these string sentinels for any non-finite float field
# (e.g. CalQLArgs.cql_clip_diff_min/max defaulting to float("-inf")/float(
# "inf")). Reading such a config.json back here must invert that encoding
# before assigning onto a float-typed field, or it ends up holding the
# Python str "-Infinity"/"Infinity" -- observed as `TypeError: clamp()
# received (Tensor, str, str)` in rl_garden/algorithms/cql.py's
# `_cql_regularizer` when replaying a run whose cql_clip_diff_min/max were
# left at their +-inf default (no config yaml overrides them).
_JSON_FLOAT_SENTINELS: dict[str, float] = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan"),
}


def _set_existing_attr(target: Any, key: str, value: Any) -> None:
    if not hasattr(target, key):
        return
    current = getattr(target, key)
    nested = is_dataclass(current) and isinstance(value, Mapping)
    if holds_removed_path(value) and not nested:
        # A policy graph or an unrecorded import path is used without its path,
        # unless the complete text is already there.
        usable = isinstance(value, str) and value != REMOVED_PATH
        if not usable or without_local_paths(current) == value:
            return
    if (
        isinstance(current, float)
        and isinstance(value, str)
        and value in _JSON_FLOAT_SENTINELS
    ):
        value = _JSON_FLOAT_SENTINELS[value]
    if is_dataclass(current) and value is None:
        # Checkpoint metadata stores resolved (collapsed) hyperparameters:
        # a structured dataclass field that was left at its sentinel default
        # (e.g. ObsGroups()/EncoderConfig()) is serialized as a bare `None`
        # (see resolve_obs_groups_config/resolve_critic_encoder_config in
        # rl_garden/common/cli_args.py). Applying that `None` back onto the
        # dataclass-typed attribute would replace the sentinel instance with
        # a literal None, breaking downstream code that expects the
        # dataclass shape. Leave the current (default) value in place.
        return
    if nested:
        if getattr(type(current), "__dataclass_params__").frozen:
            field_names = {f.name for f in fields(current)}
            updates = {
                k: v
                for k, v in value.items()
                if k in field_names and not holds_removed_path(v)
            }
            setattr(target, key, replace(current, **updates))
        else:
            _apply_mapping_to_args(current, value)
    else:
        setattr(target, key, value)


def _apply_mapping_to_args(args: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        _set_existing_attr(args, key, value)


def _run_config(
    eval_args: EvalCheckpointArgs, checkpoint: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """The run configuration to evaluate with (see the module docstring)."""
    if eval_args.config_path is not None:
        config_path = Path(eval_args.config_path)
    else:
        metadata = checkpoint.get("metadata", {})
        config_path = _default_config_path(Path(eval_args.checkpoint_path))
        embedded = metadata.get("run_config")
        if embedded is not None:
            # With its paths, if the run that wrote the checkpoint wrote it.
            beside = _read_config(config_path)
            recorded = metadata.get("run_config_digest")
            same_run = beside is not None and config_digest(beside) == recorded
            return beside if same_run else embedded
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def _read_config(path: Path) -> Mapping[str, Any] | None:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, Mapping) else None


def _require_run_config_or_environment(
    eval_args: EvalCheckpointArgs, checkpoint: Mapping[str, Any]
) -> None:
    if eval_args.config_path is not None and not Path(eval_args.config_path).exists():
        raise SystemExit(f"--config-path {eval_args.config_path} does not exist.")
    if _run_config(eval_args, checkpoint) is not None:
        return
    beside = _default_config_path(Path(eval_args.checkpoint_path))
    missing = f"The checkpoint embeds no run configuration and {beside} does not exist"
    if eval_args.env_backend is None or eval_args.env_id is None:
        raise SystemExit(
            f"{missing}; pass --config-path, or --env-backend and --env-id (with any --obs.* "
            "and backend flags the run used)."
        )
    _logger.warning(
        "%s: every setting not recorded in the checkpoint's hyperparameters or passed here "
        "takes its default, which may differ from the run's",
        missing,
    )


def _config_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
    args = config.get("inputs", config.get("args", config))
    if not isinstance(args, Mapping):
        raise SystemExit("The run configuration does not contain an args mapping.")
    return args


def _config_backend_section(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The run's env-backend sub-config as ``{section name: values}``.

    ``inputs`` in the run configuration drops every backend sub-config; the selected
    backend's values live under ``active_environment``. Configs without that
    block (older or hand-written ones) yield nothing.
    """
    active_environment = config.get("active_environment")
    if not isinstance(active_environment, Mapping):
        return {}
    backend = active_environment.get("backend")
    values = active_environment.get("config")
    if not isinstance(backend, str) or not isinstance(values, Mapping):
        return {}
    return {backend: values}


def _backend_overrides_from_cli(
    eval_args: EvalCheckpointArgs,
) -> dict[str, dict[str, Any]]:
    """Per backend sub-config, only the fields that differ from the dataclass
    default, so a sub-config left untouched on the eval CLI does not overwrite
    what the run used. A value set on the CLI equal to the dataclass default
    cannot override the run's non-default one."""
    overrides: dict[str, dict[str, Any]] = {}
    for backend_field in fields(EnvBackendArgs):
        if backend_field.name == "env_backend":
            continue
        value = getattr(eval_args, backend_field.name)
        default = backend_field.default_factory()
        changed = {
            f.name: getattr(value, f.name)
            for f in fields(value)
            if getattr(value, f.name) != getattr(default, f.name)
        }
        if changed:
            overrides[backend_field.name] = changed
    return overrides


def _default_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent.parent / "config.json"


def _load_training_args(
    phase: str,
    algorithm: str,
    eval_args: EvalCheckpointArgs,
    checkpoint: Mapping[str, Any],
) -> Any:
    registry = _registry_for_phase(phase)
    entry = registry.entries()[algorithm]
    args = entry.args_cls()

    config = _run_config(eval_args, checkpoint)
    if config is not None:
        _apply_mapping_to_args(args, _config_args(config))
        _apply_mapping_to_args(args, _config_backend_section(config))

    metadata = checkpoint.get("metadata", {})
    hyperparameters = metadata.get("hyperparameters", {})
    if isinstance(hyperparameters, Mapping):
        _apply_mapping_to_args(args, hyperparameters)
        _apply_recorded_net_arch(args, hyperparameters)
        _log_restored_policy_modules(args, hyperparameters)

    buffer_size = _evaluation_buffer_size(args, eval_args)
    _apply_mapping_to_args(args, _evaluation_overrides(eval_args))
    if buffer_size is not None:
        args.buffer_size = buffer_size
    _apply_mapping_to_args(args, _backend_overrides_from_cli(eval_args))
    _apply_mapping_to_args(args, entry.training_only_inputs)
    removed = _removed_fields(args, entry, eval_args, checkpoint)
    _require_environment_fields(args, eval_args, removed)
    return args


def _evaluation_buffer_size(args: Any, eval_args: EvalCheckpointArgs) -> int | None:
    """The training run's rows per env, times ``num_eval_envs``: a replay buffer
    split over the eval envs then passes the same size checks (window length,
    checkpoint stride) as in training. ``None`` when ``args`` has no integer
    ``buffer_size`` and ``num_envs``."""
    buffer_size, num_envs = getattr(args, "buffer_size", None), getattr(args, "num_envs", None)
    if not (isinstance(buffer_size, int) and isinstance(num_envs, int) and num_envs > 0):
        return None
    return max(buffer_size // num_envs, 1) * eval_args.num_eval_envs


def _evaluation_overrides(eval_args: EvalCheckpointArgs) -> dict[str, Any]:
    environment_overrides = {
        "env_backend": eval_args.env_backend,
        "env_id": eval_args.env_id,
        "obs": None if eval_args.obs == ObservationConfig() else eval_args.obs,
        "control_mode": eval_args.control_mode,
    }
    overrides = {
        **{key: value for key, value in environment_overrides.items() if value is not None},
        "render_mode": eval_args.render_mode,
        "num_envs": eval_args.num_eval_envs,
        "num_eval_envs": eval_args.num_eval_envs,
        # Eval never updates, and a sequence PPO requires num_envs to be
        # divisible by num_minibatches, which num_eval_envs need not be.
        "num_minibatches": 1,
        "num_eval_steps": eval_args.max_eval_steps
        or max(eval_args.num_eval_episodes * 1000, 1),
        "seed": eval_args.seed,
        "device": eval_args.device,
        "buffer_device": eval_args.buffer_device,
        # Disk-backed or lazily-read replay storage needs a real CPU buffer.
        "mmap_dir": None,
        "replay_lazy_next_obs": False,
        "reward_scale": eval_args.reward_scale,
        "reward_bias": eval_args.reward_bias,
        "capture_video": eval_args.capture_video,
        "video_fps": eval_args.video_fps,
        "eval_output_dir": eval_args.eval_output_dir,
        "eval_freq": 1,
        "log_type": eval_args.log_type,
        "std_log": eval_args.std_log,
        "load_checkpoint": None,
        "load_replay_buffer": False,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": False,
        "save_final_checkpoint": False,
    }
    for name in ("guidance_weight", "opal_checkpoint"):
        value = getattr(eval_args, name)
        if value is not None:
            overrides[name] = value
    return overrides


def _leaves(values: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in values.items():
        if isinstance(value, Mapping) and value:
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _removed_fields(
    args: Any, entry: Any, eval_args: EvalCheckpointArgs, checkpoint: Mapping[str, Any]
) -> list[str]:
    """The fields of ``args`` whose absolute path was removed from the
    checkpoint and neither the run's ``config.json`` nor the eval CLI gives."""
    config = _run_config(eval_args, checkpoint) or {}
    layers = [_config_args(config), _config_backend_section(config)]
    hyperparameters = checkpoint.get("metadata", {}).get("hyperparameters", {})
    if isinstance(hyperparameters, Mapping):
        layers.append(hyperparameters)
    removed, given = set(), set()
    for name, value in (leaf for layer in layers for leaf in _leaves(layer)):
        if holds_removed_path(value):
            removed.add(name)
        elif value is not None and not (isinstance(value, str) and value == ""):
            given.add(name)
    for layer in (
        _evaluation_overrides(eval_args),
        _backend_overrides_from_cli(eval_args),
        entry.training_only_inputs,
    ):
        given.update(name for name, _ in _leaves(layer))
    return sorted(
        name
        for name in removed - given
        if name.partition(".")[0] not in given and hasattr(args, name.partition(".")[0])
    )


def _where_to_give(names: list[str], eval_args: EvalCheckpointArgs) -> str:
    """``names``, each with its eval flag if it has one, and ``--config-path``."""

    def described(name: str) -> str:
        owner = eval_args
        for part in name.split("."):
            owner = getattr(owner, part, _NO_FLAG)
        flag = "--" + name.replace("_", "-")
        return name if owner is _NO_FLAG else f"{name} ({flag})"

    beside = _default_config_path(Path(eval_args.checkpoint_path))
    return (
        f"{', '.join(described(name) for name in names)}: a checkpoint holds no absolute paths, "
        f"and {beside} is not the config.json of the run that saved it. Pass the flag, or "
        "--config-path with that run's config.json."
    )


_NO_FLAG = object()


def _require_environment_fields(
    args: Any, eval_args: EvalCheckpointArgs, removed: list[str]
) -> None:
    """Exit when the evaluation environment would be made without a value the
    run's had: another environment, possibly without an error."""
    sections = ("obs.", f"{getattr(args, 'env_backend', None)}.")
    names = [
        name for name in removed if name in ("env_id", "control_mode") or name.startswith(sections)
    ]
    if names:
        raise SystemExit(f"The run's environment needs {_where_to_give(names, eval_args)}")


def _apply_recorded_net_arch(args: Any, hyperparameters: Mapping[str, Any]) -> None:
    """The CQL family's builders make ``net_arch`` from a width and two depths:
    set those fields to the recorded networks'."""
    net_arch = hyperparameters.get("net_arch")
    if "use_cql_loss" not in hyperparameters or net_arch is None:
        return
    if not isinstance(net_arch, Mapping):
        net_arch = {"pi": net_arch, "qf": net_arch}
    actor, critic = net_arch.get("pi"), net_arch.get("qf")
    if not actor or not critic or len({*actor, *critic}) != 1:
        return
    _apply_mapping_to_args(
        args,
        {
            "hidden_dim": actor[0],
            "actor_hidden_layers": len(actor),
            "critic_hidden_layers": len(critic),
        },
    )


def _reject_other_net_arch(agent: Any, hyperparameters: Mapping[str, Any]) -> None:
    """Exit when the builder gave a CQL-family agent other networks than the
    checkpoint records, which its load would report as a shape mismatch."""

    def per_network(net_arch: Any) -> dict[str, list[int]]:
        if not isinstance(net_arch, Mapping):
            net_arch = {"pi": net_arch, "qf": net_arch}
        return {name: list(layers) for name, layers in net_arch.items()}

    recorded, built = hyperparameters.get("net_arch"), getattr(agent, "net_arch", None)
    if "use_cql_loss" not in hyperparameters or recorded is None or built is None:
        return
    if per_network(recorded) != per_network(built):
        raise SystemExit(
            f"The checkpoint's networks are {per_network(recorded)}, but the selected entry "
            f"builds {per_network(built)}; its arguments cannot express the recorded ones."
        )


def _training_stage(agent: Any, checkpoint: Mapping[str, Any]) -> str | None:
    """The stage of the evaluated off2on agent, ``"unknown"`` when a checkpoint
    of an off2on class records none."""
    stage = getattr(agent, "training_stage", None)
    if stage is None or "online_start_step" in checkpoint["state"].get("extra", {}):
        return stage
    registered = _registered_by_class_name(
        _PHASES, checkpoint["metadata"].get("algorithm_class")
    )
    return "unknown" if any(phase == "off2on" for phase, _ in registered) else stage


def _log_restored_policy_modules(args: Any, hyperparameters: Mapping[str, Any]) -> None:
    """Log each ``PolicyModuleArgs`` import path the checkpoint metadata put
    on ``args``; the builder imports it (``policy_module_kwargs``)."""
    for module_field in fields(PolicyModuleArgs):
        spec = hyperparameters.get(module_field.name)
        if isinstance(spec, str) and getattr(args, module_field.name, None) == spec:
            _logger.info(
                "importing %s %s from checkpoint metadata", module_field.name, spec
            )


def _eval_record_dir(eval_args: EvalCheckpointArgs) -> str | None:
    if not eval_args.capture_video:
        return None
    if eval_args.eval_output_dir is not None:
        return eval_args.eval_output_dir
    checkpoint_path = Path(eval_args.checkpoint_path)
    return str(checkpoint_path.parent / "eval_videos")


def _env_request(args: Any, eval_args: EvalCheckpointArgs) -> EnvRequest:
    observation = getattr(args, "obs", None) or ObservationConfig()
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        observation=observation,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        reward_scale=getattr(args, "reward_scale", 1.0),
        reward_bias=getattr(args, "reward_bias", 0.0),
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=_eval_record_dir(eval_args),
        capture_video=eval_args.capture_video,
        video_fps=eval_args.video_fps,
        num_eval_steps=args.num_eval_steps,
        create_eval_env=True,
        backend_config=args.resolve_backend_config(),
    )


def _registered_builder(phase: str, algorithm: str):
    """The builder the registry entry declares; for an entry registered
    without one, the ``build_*`` function of its module in the phase's
    ``entrypoints`` package. Every in-tree entry with a builder declares it,
    so the module lookup serves only entries registered elsewhere without
    one."""
    entry = _registry_for_phase(phase).entries()[algorithm]
    if entry.build_agent is not None:
        return entry.build_agent
    return _builder_from_module(_training_module(phase, algorithm), algorithm)


def _training_module(phase: str, algorithm: str):
    return importlib.import_module(
        f"rl_garden.training.{phase}.entrypoints.{algorithm.replace('-', '_')}"
    )


def _builder_from_module(module: Any, algorithm: str):
    expected = f"build_{algorithm.replace('-', '_')}"
    if hasattr(module, expected):
        return getattr(module, expected)
    builders = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("build_") and callable(getattr(module, name))
    ]
    if len(builders) != 1:
        raise SystemExit(
            f"Could not identify a unique build_* function in {module.__name__}."
        )
    return builders[0]


@contextmanager
def _encoder_weights_from_checkpoint() -> Iterator[None]:
    """Encoders built in this scope skip their pretrained-weights file
    (``--encoder.pretrained-weights``, or the same field of a policy graph's
    encoder node): the checkpoint loaded next holds those weights."""
    from rl_garden.encoders.resnet import ResNetEncoder

    load_pretrained = ResNetEncoder.load_pretrained
    ResNetEncoder.load_pretrained = lambda self, name, strict=False: []
    try:
        yield
    finally:
        ResNetEncoder.load_pretrained = load_pretrained


def _build_agent(phase: str, algorithm: str, args: Any, eval_env: Any) -> Any:
    """The agent the entry's builder makes with ``eval_env`` as its only
    environment; builders and constructors read its spaces and ``num_envs``
    and never step it."""
    logger = Logger(log_type="none")
    build_agent = _registered_builder(phase, algorithm)
    if phase == "offline":
        env_spec = OfflineEnvSpec(
            eval_env.single_observation_space,
            eval_env.single_action_space,
            num_envs=1,
        )
        return build_agent(args, env_spec, logger, eval_env)
    return build_agent(args, eval_env, eval_env, logger, None)


def _policy_eval_env(agent: Any, eval_env: Any) -> Any:
    """The env the agent's policy acts in: the one its builder gave it, which
    may wrap the backend's ``eval_env`` (action chunking, skill actions), or
    ``eval_env`` itself when the agent holds none."""
    agent_eval_env = getattr(agent, "eval_env", None)
    return eval_env if agent_eval_env is None else agent_eval_env


def _reset_env(env: Any, seed: int):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _to_cpu_1d(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)
    return torch.as_tensor(value).detach().cpu().reshape(-1)


def _done_mask(
    infos: Mapping[str, Any], terminations: Any, truncations: Any
) -> torch.Tensor:
    if "_final_info" in infos:
        return _to_cpu_1d(infos["_final_info"]).bool()
    return _to_cpu_1d(terminations).bool() | _to_cpu_1d(truncations).bool()


def _append_metric_values(
    metrics: dict[str, list[float]],
    episode: Mapping[str, Any],
    mask: torch.Tensor,
    remaining: int,
) -> int:
    appended = 0
    for raw_key, value in episode.items():
        if (
            raw_key.startswith("_")
            or raw_key in EVAL_METRIC_DROP
            or isinstance(value, Mapping)
        ):
            continue
        key = EVAL_METRIC_ALIASES.get(raw_key, raw_key)
        values = _to_cpu_1d(value)
        if values.numel() == mask.numel():
            values = values[mask]
        take = min(values.numel(), remaining)
        if take <= 0:
            continue
        metrics.setdefault(key, []).extend(float(v) for v in values[:take])
        appended = max(appended, take)
    return appended


def _to_agent_device(agent: Any, value: Any) -> Any:
    device = getattr(agent, "device", None)
    if device is None or not isinstance(value, torch.Tensor):
        return value
    return value.to(device)


def evaluate_agent(
    agent: Any,
    eval_env: Any,
    *,
    seed: int,
    num_eval_episodes: int,
    max_eval_steps: int,
) -> dict[str, float]:
    if eval_env is None:
        raise SystemExit("Evaluation environment was not created.")

    agent.policy.eval()
    obs, _ = _reset_env(eval_env, seed)
    agent._eval_start_hook()

    metrics: dict[str, list[float]] = {}
    running_returns = torch.zeros(eval_env.num_envs, dtype=torch.float32)
    completed = 0
    steps = 0

    try:
        while completed < num_eval_episodes and steps < max_eval_steps:
            with torch.no_grad():
                env_action, critic_action = agent._eval_action_and_critic_action(obs)
                obs_before = obs
                obs, rewards, terminations, truncations, infos = eval_env.step(
                    env_action
                )
                # The env may live on another device than the agent.
                agent._eval_step_hook(
                    obs_before,
                    critic_action,
                    _to_agent_device(agent, rewards),
                    _to_agent_device(agent, terminations),
                    _to_agent_device(agent, truncations),
                    infos,
                )

            rewards_cpu = _to_cpu_1d(rewards).float()
            running_returns[: rewards_cpu.numel()] += rewards_cpu
            mask = _done_mask(infos, terminations, truncations)
            remaining = num_eval_episodes - completed

            appended = 0
            if isinstance(infos, Mapping) and "final_info" in infos:
                final_info = infos["final_info"]
                if isinstance(final_info, Mapping) and "episode" in final_info:
                    episode = final_info["episode"]
                    if isinstance(episode, Mapping):
                        appended = _append_metric_values(
                            metrics, episode, mask, remaining
                        )

            if appended == 0 and mask.any():
                done_returns = running_returns[mask[: running_returns.numel()]]
                take = min(done_returns.numel(), remaining)
                metrics.setdefault("return", []).extend(
                    float(v) for v in done_returns[:take]
                )
                appended = take

            if mask.any():
                running_returns[mask[: running_returns.numel()]] = 0.0
            completed += appended
            steps += 1
    finally:
        agent.policy.train()

    if completed < num_eval_episodes:
        raise SystemExit(
            f"Only completed {completed}/{num_eval_episodes} episodes within "
            f"max_eval_steps={max_eval_steps}."
        )

    out = {
        key: float(torch.tensor(values, dtype=torch.float32).mean().item())
        for key, values in sorted(metrics.items())
        if values
    }
    out["episodes_completed"] = float(completed)
    out["eval_steps"] = float(steps)
    out.update(agent._eval_finalize_hook())
    return out


def _headline_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "return" in metrics:
        out["average_return"] = metrics["return"]
    for key in ("success_at_end", "success_once", "success"):
        if key in metrics:
            out["success_rate"] = metrics[key]
            break
    for key in ("success_at_end", "success_once", "success", "episodes_completed"):
        if key in metrics:
            out[key] = metrics[key]
    return out


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(resolved_config_json(payload) + "\n", encoding="utf-8")


def _reject_unevaluable_algorithm(phase: str, algorithm: str) -> None:
    """Exit with the algorithm's ``checkpoint_eval_unsupported_reason`` when
    the registry entry's class declares one (a pretraining stage, or a
    policy that cannot act in the environment evaluation builds)."""
    entry = _registry_for_phase(phase).entries()[algorithm]
    algorithm_cls_factory = getattr(entry, "algorithm_cls", None)
    if algorithm_cls_factory is None:
        return
    algorithm_cls = algorithm_cls_factory()
    reason = getattr(algorithm_cls, "checkpoint_eval_unsupported_reason", None)
    if reason is not None:
        raise SystemExit(
            f"{phase}:{algorithm} ({algorithm_cls.__name__}) checkpoints are not "
            f"evaluable by rl-garden-eval: {reason}"
        )


def _reject_unknown_env_backend(env_backend: str) -> None:
    try:
        _get_backend(env_backend)
    except KeyError as exc:
        raise SystemExit(
            f"{exc.args[0]} Pass --env-backend to evaluate the checkpoint on another backend."
        ) from exc


def evaluate_checkpoint(eval_args: EvalCheckpointArgs) -> dict[str, Any]:
    if not eval_args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required.")
    if eval_args.num_eval_episodes <= 0:
        raise SystemExit("--num-eval-episodes must be positive.")

    checkpoint_path = Path(eval_args.checkpoint_path)
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    _require_run_config_or_environment(eval_args, checkpoint)
    phase, algorithm = _resolve_phase_and_algorithm(
        eval_args.phase,
        eval_args.algorithm,
        checkpoint,
    )
    _reject_unevaluable_algorithm(phase, algorithm)
    args = _load_training_args(phase, algorithm, eval_args, checkpoint)
    _reject_unknown_env_backend(args.env_backend)
    seed_everything(args.seed)
    enable_fast_math()

    evaluation = CheckpointEvaluation(checkpoint_path, checkpoint, eval_args.device)
    eval_env = make_evaluation_env(args.env_backend, _env_request(args, eval_args))
    try:
        with checkpoint_evaluation(evaluation), _encoder_weights_from_checkpoint():
            try:
                agent = _build_agent(phase, algorithm, args, eval_env)
            except (Exception, SystemExit):
                entry = _registry_for_phase(phase).entries()[algorithm]
                removed = _removed_fields(args, entry, eval_args, checkpoint)
                if removed:
                    _logger.error(
                        "The agent was built without %s", _where_to_give(removed, eval_args)
                    )
                raise
        _reject_other_net_arch(agent, checkpoint["metadata"].get("hyperparameters", {}))
        agent.load(
            checkpoint_path,
            strict=eval_args.strict,
            load_replay_buffer=False,
            load_optimizers=False,
        )
        max_eval_steps = eval_args.max_eval_steps or max(
            eval_args.num_eval_episodes * 1000, 1
        )
        metrics = evaluate_agent(
            agent,
            _policy_eval_env(agent, eval_env),
            seed=eval_args.seed,
            num_eval_episodes=eval_args.num_eval_episodes,
            max_eval_steps=max_eval_steps,
        )
    finally:
        eval_env.close()

    result = {
        "checkpoint_path": str(checkpoint_path),
        "phase": phase,
        "algorithm": algorithm,
        "env_backend": args.env_backend,
        "env_id": args.env_id,
        "obs": repr(getattr(args, "obs", None)),
        "control_mode": args.control_mode,
        "num_eval_envs": args.num_eval_envs,
        "num_eval_episodes": eval_args.num_eval_episodes,
        "timestamp": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        "headline": _headline_metrics(metrics),
        "metrics": metrics,
    }
    training_stage = _training_stage(agent, checkpoint)
    if training_stage is not None:
        result["training_stage"] = training_stage
    if eval_args.output_json:
        result["output_json"] = eval_args.output_json
        _write_json(eval_args.output_json, result)
    return result


def main() -> None:
    result = evaluate_checkpoint(tyro.cli(EvalCheckpointArgs))
    print("=== Evaluation Results ===", flush=True)
    if "training_stage" in result:
        print(f"training_stage: {result['training_stage']}", flush=True)
    for key, value in result["headline"].items():
        print(f"{key}: {value:.4f}", flush=True)
    print("--- all metrics ---", flush=True)
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.4f}", flush=True)
    if "output_json" in result:
        print(f"output_json: {result['output_json']}", flush=True)


if __name__ == "__main__":
    main()
