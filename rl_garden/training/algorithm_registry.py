"""Shared algorithm registry and CLI dispatch support."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
import sys
from typing import Callable, Sequence

import tyro

from rl_garden.common.cli_args import apply_log_env_defaults, logging_args_from
from rl_garden.common.resolved_config import resolved_config_json, resolved_run_config


@dataclass(frozen=True)
class AlgorithmEntry:
    args_cls: type
    run_fn: Callable


class BaseAlgorithmRegistry:
    """Discover, register, and dispatch algorithms within one training phase."""

    package_name: str
    phase_name: str

    def __init__(self) -> None:
        self._entries: dict[str, AlgorithmEntry] = {}
        self._discovered = False

    def register(self, name: str, args_cls: type, run_fn: Callable) -> None:
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

    def parse_args(self, args: Sequence[str] | None = None):
        self.discover()
        if not self._entries:
            raise RuntimeError(f"No algorithms registered in {self.package_name!r}")
        defaults = {}
        for name, entry in self._entries.items():
            default = entry.args_cls()
            logging_args = logging_args_from(default)
            if logging_args is not None:
                apply_log_env_defaults(logging_args)
            defaults[name] = default
        cli_type = tyro.extras.subcommand_type_from_defaults(defaults)
        return tyro.cli(
            cli_type,
            args=args,
        )

    def entry_for_args(self, args) -> tuple[str, AlgorithmEntry]:
        for name, entry in self._entries.items():
            if type(args) is entry.args_cls:
                return name, entry
        raise TypeError(f"No registered algorithm accepts args type {type(args).__name__!r}")

    def dispatch(self, args) -> None:
        _, entry = self.entry_for_args(args)
        entry.run_fn(args)

    def run_cli(self, args: Sequence[str] | None = None) -> None:
        cli_args = list(sys.argv[1:] if args is None else args)
        print_config = "--print-config" in cli_args
        cli_args = [arg for arg in cli_args if arg != "--print-config"]
        parsed = self.parse_args(cli_args)
        algorithm, _ = self.entry_for_args(parsed)
        if print_config:
            print(
                resolved_config_json(
                    resolved_run_config(
                        parsed,
                        training_phase=self.phase_name,
                        algorithm=algorithm,
                    )
                )
            )
            return
        # TODO: Add a --dry-run path that validates runtime configuration.
        self.dispatch(parsed)
