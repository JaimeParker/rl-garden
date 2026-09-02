#!/usr/bin/env python3
"""Run the official WSRL repo's D4RL offline-to-online scripts.

This module is intentionally a thin, auditable subprocess wrapper. It does
not reimplement WSRL's JAX training loop or route rollouts through rl-garden's
environment bridge; it launches the release script from ``3rd_party/wsrl`` with
explicit overrides and records the resolved command for provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from baselines.core.reporting import write_json

SCRIPT_NAMES = {
    "calql": "launch_calql_finetune.sh",
    "cql": "launch_cql_finetune.sh",
    "iql": "launch_iql_finetune.sh",
    "rlpd": "launch_rlpd.sh",
    "wsrl": "launch_wsrl_finetune.sh",
}

DEFAULT_ENVS = {
    "adroit": "door-binary-v0",
    "antmaze": "antmaze-large-play-v2",
    "kitchen": "kitchen-partial-v0",
    "locomotion": "halfcheetah-medium-replay-v2",
}


@dataclass(frozen=True)
class WsrlCommand:
    command: list[str]
    cwd: str
    env_overrides: dict[str, str]
    script: str
    wsrl_commit: str
    metadata: dict[str, object]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an official WSRL offline-to-online baseline script."
    )
    parser.add_argument("--wsrl-source", default="3rd_party/wsrl")
    parser.add_argument("--domain", default="antmaze")
    parser.add_argument("--algorithm", default="calql")
    parser.add_argument(
        "--env",
        help=(
            "D4RL env id. Defaults to a domain-specific WSRL env "
            "(antmaze-large-play-v2, door-binary-v0, kitchen-partial-v0, "
            "or halfcheetah-medium-replay-v2)."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-dir", default="~/wsrl_log")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-offline-steps", type=int, default=1_000_000)
    parser.add_argument("--num-online-steps", type=int, default=500_000)
    parser.add_argument("--offline-data-ratio", type=float, default=0.5)
    parser.add_argument("--online-sampling-method", default="mixed", choices=("mixed", "append"))
    parser.add_argument(
        "--online-use-cql-loss",
        default="True",
        choices=("True", "False", "true", "false"),
        help="Forwarded to WSRL; meaningful for CQL/CalQL agents.",
    )
    parser.add_argument("--project", default="d4rl")
    parser.add_argument("--group", default="wsrl-calql-retained")
    parser.add_argument("--exp-name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-redq", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw flag passed to the WSRL script; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _candidate_script(domain: str, algorithm: str) -> Path:
    if algorithm not in SCRIPT_NAMES:
        supported = ", ".join(sorted(SCRIPT_NAMES))
        raise ValueError(f"unsupported WSRL algorithm {algorithm!r}; supported: {supported}")
    return Path("experiments/scripts") / domain / SCRIPT_NAMES[algorithm]


def _script_for(source: Path, domain: str, algorithm: str) -> Path:
    script = _candidate_script(domain, algorithm)
    if (source / script).is_file():
        return script

    scripts_root = source / "experiments" / "scripts"
    supported: list[str] = []
    if scripts_root.is_dir():
        for script_path in sorted(scripts_root.glob("*/launch*.sh")):
            script_domain = script_path.parent.name
            script_name = script_path.name
            for name, filename in SCRIPT_NAMES.items():
                if script_name == filename:
                    supported.append(f"{script_domain}/{name}")
                    break
    supported_text = ", ".join(supported) if supported else "none found"
    raise ValueError(
        f"unsupported WSRL D4RL script {domain}/{algorithm}; supported: {supported_text}"
    )


def _git_commit(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _append_flag(command: list[str], name: str, value: object) -> None:
    command.extend([name, str(value)])


def _default_env(domain: str) -> str:
    if domain not in DEFAULT_ENVS:
        supported = ", ".join(sorted(DEFAULT_ENVS))
        raise ValueError(f"unsupported WSRL domain {domain!r}; supported: {supported}")
    return DEFAULT_ENVS[domain]


def _bool_string(value: str) -> str:
    return "True" if value.lower() == "true" else "False"


def build_command(args: argparse.Namespace) -> WsrlCommand:
    source = Path(args.wsrl_source).expanduser().resolve()
    script = _script_for(source, args.domain, args.algorithm)
    script_path = source / script
    if not source.is_dir():
        raise FileNotFoundError(f"WSRL source directory not found: {source}")
    if not script_path.is_file():
        raise FileNotFoundError(f"WSRL launch script not found: {script_path}")

    env_id = args.env or _default_env(args.domain)
    exp_name = args.exp_name or f"{args.algorithm}_{env_id}_s{args.seed}"
    save_dir = str(Path(args.save_dir).expanduser().resolve())
    command = ["bash", str(script)]
    _append_flag(command, "--env", env_id)
    _append_flag(command, "--seed", args.seed)
    _append_flag(command, "--save_dir", save_dir)
    _append_flag(command, "--num_offline_steps", args.num_offline_steps)
    _append_flag(command, "--num_online_steps", args.num_online_steps)
    _append_flag(command, "--offline_data_ratio", args.offline_data_ratio)
    _append_flag(command, "--online_sampling_method", args.online_sampling_method)
    _append_flag(command, "--online_use_cql_loss", _bool_string(args.online_use_cql_loss))
    _append_flag(command, "--project", args.project)
    _append_flag(command, "--group", args.group)
    _append_flag(command, "--exp_name", exp_name)
    if args.debug:
        command.append("--debug")
    if args.use_redq:
        command.append("--use_redq")
    command.extend(args.extra_arg)

    env_overrides = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_GL": "egl",
        "PYTHONPATH": str(source),
    }
    metadata = {
        "implementation": "official-wsrl-script-forwarder",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "env": env_id,
        "seed": args.seed,
        "num_offline_steps": args.num_offline_steps,
        "num_online_steps": args.num_online_steps,
        "offline_data_ratio": args.offline_data_ratio,
        "online_sampling_method": args.online_sampling_method,
        "online_use_cql_loss": _bool_string(args.online_use_cql_loss) == "True",
        "project": args.project,
        "group": args.group,
        "exp_name": exp_name,
        "save_dir": save_dir,
        "use_redq": args.use_redq,
        "debug": args.debug,
        "extra_args": list(args.extra_arg),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "This wrapper forwards to the unmodified WSRL release script. "
            "At the inspected WSRL commit, finetune.py hard-codes WandB "
            "project/group to 'wsrl' before applying CLI values, so project "
            "and group flags are recorded here even if upstream ignores them."
        ),
    }
    return WsrlCommand(
        command=command,
        cwd=str(source),
        env_overrides=env_overrides,
        script=str(script),
        wsrl_commit=_git_commit(source),
        metadata=metadata,
    )


def _child_env(spec: WsrlCommand) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in spec.env_overrides.items():
        if key == "PYTHONPATH" and env.get("PYTHONPATH"):
            env[key] = value + os.pathsep + env[key]
        else:
            env[key] = value
    return env


def _write_command(output_dir: Path, spec: WsrlCommand) -> None:
    output = {
        **asdict(spec),
        "stdout_log": str(output_dir / "stdout.log"),
        "stderr_log": str(output_dir / "stderr.log"),
    }
    write_json(output_dir / "command.json", output)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = build_command(args)
    _write_command(output_dir, spec)

    if args.dry_run:
        print(json.dumps(asdict(spec), indent=2, sort_keys=True))
        return 0

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            env=_child_env(spec),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )

    write_json(
        output_dir / "result.json",
        {
            "returncode": process.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
    )
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
