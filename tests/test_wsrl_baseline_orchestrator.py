import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.wsrl.run_offline_to_online import (
    _build_parser,
    _script_for,
    build_command,
    main,
)


def _fake_wsrl_source(tmp_path: Path) -> Path:
    source = tmp_path / "wsrl"
    scripts = source / "experiments" / "scripts"
    for domain, names in {
        "antmaze": ("launch_calql_finetune.sh", "launch_iql_finetune.sh"),
        "kitchen": ("launch_calql_finetune.sh",),
        "locomotion": ("launch_cql_finetune.sh",),
    }.items():
        domain_dir = scripts / domain
        domain_dir.mkdir(parents=True)
        for name in names:
            (domain_dir / name).write_text("#!/usr/bin/env bash\n")
    return source


def _parse(tmp_path: Path, *extra: str):
    source = _fake_wsrl_source(tmp_path)
    return _build_parser().parse_args(
        ["--wsrl-source", str(source), "--output-dir", str(tmp_path / "out"), *extra]
    )


def test_default_command_matches_figure6_retained_calql(tmp_path):
    spec = build_command(_parse(tmp_path))

    assert spec.command[:2] == [
        "bash",
        "experiments/scripts/antmaze/launch_calql_finetune.sh",
    ]
    assert spec.cwd == str((tmp_path / "wsrl").resolve())
    assert "--env" in spec.command
    assert spec.command[spec.command.index("--env") + 1] == "antmaze-large-play-v2"
    assert spec.command[spec.command.index("--offline_data_ratio") + 1] == "0.5"
    assert spec.command[spec.command.index("--num_offline_steps") + 1] == "1000000"
    assert spec.command[spec.command.index("--online_use_cql_loss") + 1] == "True"
    assert Path(spec.command[spec.command.index("--save_dir") + 1]).is_absolute()
    assert "--use_redq" not in spec.command
    assert spec.metadata["offline_data_ratio"] == 0.5
    assert spec.metadata["online_use_cql_loss"] is True
    assert spec.metadata["online_sampling_method"] == "mixed"
    assert Path(spec.metadata["save_dir"]).is_absolute()


def test_num_online_steps_and_use_redq_are_explicit_overrides(tmp_path):
    spec = build_command(
        _parse(tmp_path, "--num-online-steps", "1000000", "--use-redq")
    )

    assert spec.command[spec.command.index("--num_online_steps") + 1] == "1000000"
    assert "--use_redq" in spec.command
    assert spec.metadata["num_online_steps"] == 1_000_000
    assert spec.metadata["use_redq"] is True


def test_extra_args_are_forwarded_after_wrapper_flags(tmp_path):
    spec = build_command(
        _parse(
            tmp_path,
            "--extra-arg=--eval_interval=2000",
            "--extra-arg=--debug",
        )
    )

    assert spec.command[-2:] == ["--eval_interval=2000", "--debug"]
    assert spec.metadata["extra_args"] == ["--eval_interval=2000", "--debug"]


def test_wsrl_d4rl_script_matrix_uses_existing_upstream_scripts(tmp_path):
    source = _fake_wsrl_source(tmp_path)

    kitchen = build_command(
        _build_parser().parse_args(
            [
                "--wsrl-source",
                str(source),
                "--domain",
                "kitchen",
                "--algorithm",
                "calql",
                "--output-dir",
                str(tmp_path / "out-kitchen"),
            ]
        )
    )
    assert kitchen.command[:2] == [
        "bash",
        "experiments/scripts/kitchen/launch_calql_finetune.sh",
    ]
    assert kitchen.command[kitchen.command.index("--env") + 1] == "kitchen-partial-v0"

    locomotion = build_command(
        _build_parser().parse_args(
            [
                "--wsrl-source",
                str(source),
                "--domain",
                "locomotion",
                "--algorithm",
                "cql",
                "--env",
                "hopper-medium-v2",
                "--online-use-cql-loss",
                "False",
                "--output-dir",
                str(tmp_path / "out-locomotion"),
            ]
        )
    )
    assert locomotion.command[:2] == [
        "bash",
        "experiments/scripts/locomotion/launch_cql_finetune.sh",
    ]
    assert locomotion.command[locomotion.command.index("--env") + 1] == "hopper-medium-v2"
    assert locomotion.command[locomotion.command.index("--online_use_cql_loss") + 1] == "False"
    assert locomotion.metadata["online_use_cql_loss"] is False


def test_unsupported_domain_algorithm_pair_fails(tmp_path):
    source = _fake_wsrl_source(tmp_path)

    with pytest.raises(ValueError, match="unsupported WSRL D4RL script"):
        _script_for(source, "locomotion", "calql")


def test_dry_run_writes_command_json(tmp_path, capsys):
    source = _fake_wsrl_source(tmp_path)
    output_dir = tmp_path / "dryrun"

    result = main(
        [
            "--wsrl-source",
            str(source),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    command_json = json.loads((output_dir / "command.json").read_text())
    assert command_json["metadata"]["env"] == "antmaze-large-play-v2"
    assert command_json["metadata"]["offline_data_ratio"] == 0.5
    assert command_json["command"][0] == "bash"
    assert "launch_calql_finetune.sh" in capsys.readouterr().out
