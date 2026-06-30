import json

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.common.resolved_config import persist_resolved_config


def test_persist_resolved_config_writes_recursive_json(tmp_path):
    args = EnvBackendArgs(env_backend="robotwin")

    config = persist_resolved_config(
        args,
        training_phase="online",
        algorithm="test",
        run_name="run",
        log_dir=str(tmp_path),
    )

    path = tmp_path / "run" / "config.json"
    assert json.loads(path.read_text()) == config
    assert config["args"]["env_backend"] == "robotwin"
    assert isinstance(config["args"]["robotwin"], dict)
