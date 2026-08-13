"""Shared algorithm registry and CLI dispatch support."""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import tyro

from rl_garden.common.cli_args import resolve_num_eval_steps
from rl_garden.common.effective_config import (
    ConfigError,
    FieldSource,
    apply_strict_mapping,
    effective_config_json,
    inactive_config_paths,
    json_value,
    load_preset,
    override_sources,
    resolve_active_environment,
    resolve_effective_config,
    runtime_metadata,
)


@dataclass(frozen=True)
class AlgorithmEntry:
    args_cls: type
    run_fn: Callable


@dataclass(frozen=True)
class ParsedCommand:
    args: object
    algorithm: str
    action: str
    explain_param: str | None
    sources: dict[str, FieldSource]
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
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Algorithm {name!r} already registered")
        if any(entry.args_cls is args_cls for entry in self._entries.values()):
            raise ValueError(f"Args type {args_cls.__name__!r} already registered")
        self._entries[name] = AlgorithmEntry(args_cls, run_fn)

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

    def _defaults(self) -> dict[str, object]:
        self.discover()
        if not self._entries:
            raise RuntimeError(f"No algorithms registered in {self.package_name!r}")
        defaults: dict[str, object] = {}
        for name, entry in self._entries.items():
            defaults[name] = entry.args_cls()
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
        defaults = self._defaults()
        if algorithm not in defaults:
            raise ConfigError(
                f"Unknown algorithm {algorithm!r}. Available: {sorted(defaults)}."
            )
        sources: dict[str, FieldSource] = {}
        if preset_path is not None:
            preset = load_preset(preset_path)
            apply_strict_mapping(defaults[algorithm], preset.values)
            override_sources(
                sources,
                preset.paths,
                kind="preset",
                detail=preset.path,
            )
        cli_type = tyro.extras.subcommand_type_from_defaults(defaults)
        parsed = tyro.cli(cli_type, args=cli_args)
        override_sources(
            sources,
            self._cli_override_paths(cli_args),
            kind="CLI",
            detail="argv",
        )
        normalized, runtime_derived = self._normalize_runtime(parsed, sources)
        return ParsedCommand(
            normalized,
            algorithm,
            action,
            explain_param,
            sources,
            original_argv,
            runtime_derived,
        )

    def _normalize_runtime(
        self,
        parsed: object,
        sources: dict[str, FieldSource],
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
            override_sources(
                sources,
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
            and getattr(normalized, "dataset_backend", None) == "minari"
            and getattr(normalized, "env_id", None) == "PickCube-v1"
        ):
            set_derived(
                "env_id",
                normalized.offline_dataset,
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
            elif not getattr(args, "offline_dataset", None):
                raise ConfigError(
                    "--offline_dataset is required for offline pretraining."
                )
            if getattr(args, "num_offline_steps", 1) <= 0:
                raise ConfigError("--num_offline_steps must be positive.")
        if (
            self.phase_name == "off2on"
            and getattr(args, "num_offline_steps", 0) > 0
            and not getattr(args, "offline_dataset", None)
        ):
            raise ConfigError(
                "--offline_dataset is required when --num_offline_steps > 0."
            )
        load_checkpoint = getattr(args, "load_checkpoint", None)
        if load_checkpoint is not None and not Path(load_checkpoint).is_file():
            raise ConfigError(f"Checkpoint does not exist: {load_checkpoint}")

    def _preflight_config(self, command: ParsedCommand):
        args = command.args
        return resolve_effective_config(
            args,
            training_phase=self.phase_name,
            algorithm=command.algorithm,
            sources=command.sources,
            active_environment=resolve_active_environment(args),
            algorithm_config={},
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
                    "  --explain-param PATH   Explain one parameter's value and source.\n"
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
            value = self._value_at_path(command.args, field_path)
            inactive = inactive_config_paths(command.args)
            if field_path in inactive:
                raise ConfigError(
                    f"Configuration field {field_path!r} is inactive: "
                    f"{inactive[field_path]}."
                )
            source = command.sources.get(field_path)
            payload = {
                "path": field_path,
                "value": json_value(value),
                "type": type(value).__name__,
                "source": (
                    json_value(source)
                    if source is not None
                    else {"kind": "default", "detail": None}
                ),
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
            ):
                self.dispatch(command.args)
            return
        from rl_garden.training.inspection import config_session

        with config_session(
            self._preflight_config(command),
            dry_run=False,
        ):
            self.dispatch(command.args)
