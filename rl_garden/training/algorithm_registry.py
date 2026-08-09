"""Shared algorithm registry and CLI dispatch support."""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from rl_garden.common.cli_args import (
    apply_log_env_defaults,
    logging_args_from,
    resolve_num_eval_steps,
)
from rl_garden.common.effective_config import (
    ConfigError,
    FieldProvenance,
    apply_strict_mapping,
    default_provenance,
    effective_config_json,
    json_value,
    load_preset,
    override_provenance,
    resolve_effective_config,
    runtime_metadata,
)
from rl_garden.training.config_contract import ConfigContract


@dataclass(frozen=True)
class AlgorithmEntry:
    args_cls: type
    run_fn: Callable
    contract: ConfigContract
    validate_config: Callable[[object], None] | None = None


@dataclass(frozen=True)
class ParsedCommand:
    args: object
    algorithm: str
    action: str
    explain_param: str | None
    provenance: dict[str, FieldProvenance]
    argv: tuple[str, ...]
    derived: dict[str, object]


class BaseAlgorithmRegistry:
    """Discover, register, and dispatch algorithms within one training phase."""

    package_name: str
    phase_name: str
    supports_dry_run: bool = True

    def __init__(self) -> None:
        self._entries: dict[str, AlgorithmEntry] = {}
        self._discovered = False

    def register(
        self,
        name: str,
        args_cls: type,
        run_fn: Callable,
        *,
        target: str | None = None,
        derived_parameters: frozenset[str] = frozenset(),
        validate_config: Callable[[object], None] | None = None,
        contract_mode: Literal["strict", "passthrough"] = "strict",
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Algorithm {name!r} already registered")
        if any(entry.args_cls is args_cls for entry in self._entries.values()):
            raise ValueError(f"Args type {args_cls.__name__!r} already registered")
        contract_target = target or f"{run_fn.__module__}.{run_fn.__qualname__}"
        self._entries[name] = AlgorithmEntry(
            args_cls,
            run_fn,
            ConfigContract.for_args(
                args_cls,
                target=contract_target,
                derived_parameters=derived_parameters,
                mode=contract_mode,
            ),
            validate_config,
        )

    def entries(self) -> dict[str, AlgorithmEntry]:
        return dict(self._entries)

    def discover(self) -> None:
        if self._discovered:
            return
        package = importlib.import_module(self.package_name)
        for info in pkgutil.iter_modules(package.__path__):
            if not info.name.startswith("_"):
                importlib.import_module(f"{self.package_name}.{info.name}")
        self._discovered = True

    def _defaults(self, *, apply_logging_env: bool = True) -> dict[str, object]:
        self.discover()
        if not self._entries:
            raise RuntimeError(f"No algorithms registered in {self.package_name!r}")
        defaults: dict[str, object] = {}
        for name, entry in self._entries.items():
            default = entry.args_cls()
            logging_args = logging_args_from(default)
            if apply_logging_env and logging_args is not None:
                apply_log_env_defaults(logging_args)
            defaults[name] = default
        return defaults

    def parse_args(self, args: Sequence[str] | None = None):
        defaults = self._defaults()
        cli_type = tyro.extras.subcommand_type_from_defaults(defaults)
        return tyro.cli(
            cli_type,
            args=args,
        )

    @staticmethod
    def _pop_control_args(
        cli_args: list[str],
    ) -> tuple[list[str], str, str | None, str | None]:
        remaining: list[str] = []
        actions: list[str] = []
        preset_path: str | None = None
        explain_param: str | None = None

        def select_preset(candidate: str) -> None:
            nonlocal preset_path
            if preset_path is None:
                preset_path = candidate
                return
            launcher_default = os.getenv("RLG_LAUNCHER_PRESET")
            if (
                launcher_default
                and Path(preset_path).resolve() == Path(launcher_default).resolve()
            ):
                preset_path = candidate
                return
            raise ConfigError("Only one --config preset may be supplied.")

        index = 0
        while index < len(cli_args):
            token = cli_args[index]
            if token in {"--print-config", "--dry-run"}:
                actions.append(token[2:].replace("-", "_"))
                index += 1
                continue
            if token == "--config":
                if index + 1 >= len(cli_args):
                    raise ConfigError("--config requires a YAML path.")
                select_preset(cli_args[index + 1])
                index += 2
                continue
            if token.startswith("--config="):
                select_preset(token.split("=", 1)[1])
                index += 1
                continue
            if token == "--explain-param":
                if index + 1 >= len(cli_args):
                    raise ConfigError("--explain-param requires a dotted field path.")
                actions.append("explain_param")
                explain_param = cli_args[index + 1]
                index += 2
                continue
            if token.startswith("--explain-param="):
                actions.append("explain_param")
                explain_param = token.split("=", 1)[1]
                index += 1
                continue
            remaining.append(token)
            index += 1
        if len(actions) > 1:
            raise ConfigError(
                "--print-config, --dry-run, and --explain-param are mutually exclusive."
            )
        return remaining, actions[0] if actions else "run", preset_path, explain_param

    @staticmethod
    def _algorithm_token(cli_args: Sequence[str]) -> str:
        for token in cli_args:
            if not token.startswith("-"):
                return token
        raise ConfigError("An algorithm subcommand is required.")

    @staticmethod
    def _cli_override_paths(cli_args: Sequence[str]) -> set[str]:
        paths: set[str] = set()
        for token in cli_args:
            if not token.startswith("--"):
                continue
            option = token[2:].split("=", 1)[0]
            if option.startswith(("no-", "no_")):
                option = option[3:]
            paths.add(".".join(part.replace("-", "_") for part in option.split(".")))
        return paths

    @staticmethod
    def _value_at_path(args: object, path: str) -> object:
        value = args
        for part in path.split("."):
            try:
                value = getattr(value, part)
            except AttributeError as exc:
                raise ConfigError(f"Unknown configuration field {path!r}.") from exc
        return value

    def parse_command(self, args: Sequence[str] | None = None) -> ParsedCommand:
        cli_args = list(sys.argv[1:] if args is None else args)
        original_argv = tuple(cli_args)
        cli_args, action, preset_path, explain_param = self._pop_control_args(cli_args)
        algorithm = self._algorithm_token(cli_args)
        defaults = self._defaults(apply_logging_env=False)
        if algorithm not in defaults:
            raise ConfigError(
                f"Unknown algorithm {algorithm!r}. Available: {sorted(defaults)}."
            )
        provenance = default_provenance(defaults[algorithm])
        if preset_path is not None:
            preset = load_preset(preset_path)
            apply_strict_mapping(defaults[algorithm], preset.values)
            launcher = os.getenv("RLG_LAUNCHER")
            launcher_preset = os.getenv("RLG_LAUNCHER_PRESET")
            is_launcher_default = bool(
                launcher
                and launcher_preset
                and Path(launcher_preset).resolve() == Path(preset.path).resolve()
            )
            override_provenance(
                provenance,
                preset.paths,
                kind="launcher" if is_launcher_default else "preset",
                detail=launcher if is_launcher_default else preset.path,
            )
        for default in defaults.values():
            logging_args = logging_args_from(default)
            if logging_args is not None:
                apply_log_env_defaults(logging_args)
        logging_env = {
            "std_log": "RLG_STD_LOG",
            "log_type": "RLG_LOG_TYPE",
            "log_keywords": "RLG_LOG_KEYWORDS",
            "wandb_project": "RLG_WANDB_PROJECT",
            "wandb_entity": "RLG_WANDB_ENTITY",
            "wandb_group": "RLG_WANDB_GROUP",
        }
        for field_name, env_name in logging_env.items():
            if os.getenv(env_name) is not None:
                override_provenance(
                    provenance,
                    {field_name},
                    kind="RLG_*",
                    detail=env_name,
                )
        cli_type = tyro.extras.subcommand_type_from_defaults(defaults)
        parsed = tyro.cli(cli_type, args=cli_args)
        override_provenance(
            provenance,
            self._cli_override_paths(cli_args),
            kind="CLI",
            detail="argv",
        )
        normalized, runtime_derived = self._normalize_runtime(parsed, provenance)
        return ParsedCommand(
            normalized,
            algorithm,
            action,
            explain_param,
            provenance,
            original_argv,
            runtime_derived,
        )

    def _normalize_runtime(
        self,
        parsed: object,
        provenance: dict[str, FieldProvenance],
    ) -> tuple[object, dict[str, object]]:
        """Return the exact Args object consumed by the runner."""
        normalized = deepcopy(parsed)
        derived: dict[str, object] = {}

        def set_derived(path: str, value: object, reason: str) -> None:
            before = getattr(normalized, path)
            if before == value:
                return
            setattr(normalized, path, value)
            derived[path] = {
                "before": json_value(before),
                "after": json_value(value),
                "reason": reason,
            }
            override_provenance(
                provenance,
                {path},
                kind="runtime-derived",
                detail=reason,
            )

        if getattr(normalized, "buffer_device", None) == "cuda":
            import torch

            if not torch.cuda.is_available():
                warnings.warn(
                    "CUDA not available; falling back to CPU buffer.",
                    stacklevel=2,
                )
                set_derived("buffer_device", "cpu", "CUDA is unavailable")

        if self.phase_name in {"offline", "off2on"} and hasattr(
            normalized, "num_eval_steps"
        ):
            resolved_steps = resolve_num_eval_steps(
                num_eval_steps=normalized.num_eval_steps,
                num_eval_episodes=getattr(normalized, "num_eval_episodes", None),
                eval_episode_horizon=getattr(normalized, "eval_episode_horizon", None),
                default=50,
            )
            set_derived(
                "num_eval_steps",
                resolved_steps,
                "resolved evaluation step budget",
            )

        if (
            self.phase_name == "off2on"
            and getattr(normalized, "dataset_source", None) == "minari"
            and getattr(normalized, "env_id", None) == "PickCube-v1"
        ):
            set_derived(
                "env_id",
                normalized.offline_dataset_path,
                "derived live environment from Minari dataset id",
            )
        return normalized, derived

    def _validate_config(self, command: ParsedCommand) -> None:
        args = command.args
        if self.phase_name == "offline":
            if command.algorithm == "tdmpc2_multitask":
                if not getattr(args, "dataset_dir", None):
                    raise ConfigError("--dataset_dir is required for tdmpc2_multitask.")
                if not getattr(args, "mmap_dir", None):
                    raise ConfigError("--mmap_dir is required for tdmpc2_multitask.")
            elif not getattr(args, "offline_dataset_path", None):
                raise ConfigError(
                    "--offline_dataset_path is required for offline pretraining."
                )
            if getattr(args, "num_offline_steps", 1) <= 0:
                raise ConfigError("--num_offline_steps must be positive.")
        if (
            self.phase_name == "off2on"
            and getattr(args, "num_offline_steps", 0) > 0
            and not getattr(args, "offline_dataset_path", None)
        ):
            raise ConfigError(
                "--offline_dataset_path is required when --num_offline_steps > 0."
            )
        load_checkpoint = getattr(args, "load_checkpoint", None)
        if load_checkpoint is not None and not Path(load_checkpoint).is_file():
            raise ConfigError(f"Checkpoint does not exist: {load_checkpoint}")
        entry = self._entries[command.algorithm]
        if entry.validate_config is not None:
            entry.validate_config(args)

    def _preflight_config(self, command: ParsedCommand):
        args = command.args
        entry = self._entries[command.algorithm]
        applied_provenance = entry.contract.apply(command.provenance)
        command.provenance.clear()
        command.provenance.update(
            entry.contract.validate_active(args, applied_provenance)
        )
        active_paths = {
            path for path, field in command.provenance.items() if field.active
        }
        active_environment: dict[str, object] = {}
        if hasattr(args, "env_backend"):
            backend_name = args.env_backend
            try:
                backend_config = getattr(args, backend_name)
            except AttributeError as exc:
                available = sorted(
                    field
                    for field in vars(args)
                    if hasattr(getattr(args, field), "__dataclass_fields__")
                )
                raise ConfigError(
                    f"Unknown env backend {backend_name!r}. Available: {available}."
                ) from exc
            backend_values = json_value(backend_config)
            if isinstance(backend_values, dict):
                for key, value in backend_values.items():
                    if key.endswith("_kwargs_json") and value:
                        try:
                            parsed_json = json.loads(value)
                        except (TypeError, json.JSONDecodeError) as exc:
                            raise ConfigError(
                                f"Invalid {backend_name}.{key}: expected a JSON object."
                            ) from exc
                        if not isinstance(parsed_json, dict):
                            raise ConfigError(
                                f"Invalid {backend_name}.{key}: expected a JSON object."
                            )
            active_environment = {
                "backend": backend_name,
                "config": backend_values,
            }
        try:
            implicit_defaults = (
                entry.contract.constructor_defaults()
                if entry.contract.mode == "strict"
                else {}
            )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid config contract: {exc}") from exc
        return resolve_effective_config(
            args,
            training_phase=self.phase_name,
            algorithm=command.algorithm,
            provenance=command.provenance,
            active_environment=active_environment,
            algorithm_config={
                "target": entry.contract.target,
                "mode": entry.contract.mode,
                "constructor_kwargs": {},
                "field_mappings": entry.contract.field_mappings(active_paths),
                "implicit_defaults": implicit_defaults,
            },
            derived=command.derived,
            runtime=runtime_metadata(argv=[sys.argv[0], *command.argv]),
        )

    def entry_for_args(self, args) -> tuple[str, AlgorithmEntry]:
        for name, entry in self._entries.items():
            if type(args) is entry.args_cls:
                return name, entry
        raise TypeError(
            f"No registered algorithm accepts args type {type(args).__name__!r}"
        )

    def dispatch(self, args) -> None:
        _, entry = self.entry_for_args(args)
        entry.run_fn(args)

    def run_cli(self, args: Sequence[str] | None = None) -> None:
        try:
            cli_args = list(sys.argv[1:] if args is None else args)
            if any(token in {"-h", "--help"} for token in cli_args):
                print(
                    "Inspection options:\n"
                    "  --config PATH          Load a strict YAML preset.\n"
                    "  --print-config         Validate and print preflight config.\n"
                    "  --dry-run              Materialize env and agent without training.\n"
                    "  --explain-param PATH   Explain one parameter's value and provenance.\n"
                )
                remaining, _, _, _ = self._pop_control_args(cli_args)
                if not any(not token.startswith("-") for token in remaining):
                    self.parse_args(
                        [token for token in remaining if token in {"-h", "--help"}]
                    )
            self._run_cli(args)
        except ConfigError as exc:
            raise SystemExit(f"Configuration error: {exc}") from exc

    def _run_cli(self, args: Sequence[str] | None = None) -> None:
        command = self.parse_command(args)
        self._validate_config(command)
        if command.action == "print_config":
            print(effective_config_json(self._preflight_config(command)))
            return
        if command.action == "explain_param":
            assert command.explain_param is not None
            self._preflight_config(command)
            field_path = command.explain_param.replace("-", "_")
            try:
                field = command.provenance[field_path]
            except KeyError as exc:
                raise ConfigError(
                    f"Unknown configuration field {field_path!r}."
                ) from exc
            value = self._value_at_path(command.args, field_path)
            payload = {
                "path": field_path,
                "value": json_value(value),
                "type": type(value).__name__,
                **json_value(field),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        if command.action == "dry_run":
            if not self.supports_dry_run:
                raise ConfigError(
                    f"--dry-run is not supported for {self.phase_name!r} "
                    "training: these entrypoints drive physical hardware and "
                    "have no way to materialize config without doing so."
                )
            from rl_garden.training.inspection import config_session

            with config_session(
                self._preflight_config(command),
                dry_run=True,
                contract=self._entries[command.algorithm].contract,
            ):
                self.dispatch(command.args)
            return
        from rl_garden.training.inspection import config_session

        with config_session(
            self._preflight_config(command),
            dry_run=False,
            contract=self._entries[command.algorithm].contract,
        ):
            self.dispatch(command.args)
