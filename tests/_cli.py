"""In-process driver for the ``examples/*.py`` training entrypoints.

Each entrypoint is just ``registry.run_cli()`` for one phase registry, so tests
can call that registry directly instead of paying a fresh interpreter plus
import of torch/gymnasium per case.

Process-global state touched by ``registry.run_cli`` on the print-config /
explain-param / error paths: ``registry.discover()`` imports the phase's
algorithm modules (idempotent, cached in ``sys.modules``), and
``runtime_metadata`` records ``sys.argv[0]`` (not restored; only echoed into the
printed config). ``--dry-run`` and real runs additionally enter
``config_session``; do not route those through this helper.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest import mock

Entry = Literal["train_online", "train_off2on", "pretrain_offline"]

_ENTRY_REGISTRY_MODULE: dict[str, str] = {
    "train_online": "rl_garden.training.online",
    "train_off2on": "rl_garden.training.off2on",
    "pretrain_offline": "rl_garden.training.offline",
}


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_entry(
    entry: Entry,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CliResult:
    """Run ``examples/<entry>.py`` in-process and mimic its process exit status."""
    registry = importlib.import_module(_ENTRY_REGISTRY_MODULE[entry]).registry
    stdout, stderr = io.StringIO(), io.StringIO()
    returncode = 0
    previous_cwd = Path.cwd()
    try:
        with (
            mock.patch.dict(os.environ, env or {}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            if cwd is not None:
                os.chdir(cwd)
            try:
                registry.run_cli(list(argv))
            except SystemExit as exc:
                returncode = _exit_status(exc.code, stderr)
    finally:
        os.chdir(previous_cwd)
    return CliResult(returncode, stdout.getvalue(), stderr.getvalue())


def _exit_status(code: object, stderr: io.StringIO) -> int:
    # The interpreter prints a non-int SystemExit payload to stderr and exits 1.
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=stderr)
    return 1
