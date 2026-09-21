"""CLI-level smoke tests for the FAC offline entrypoint."""

from __future__ import annotations

import json


def test_offline_fac_print_config_matches_defaults(tmp_path, capsys):
    """FAC's offline dataset is only validated for truthiness by
    --print-config (see algorithm_registry.py's _validate_config), so a
    non-existent path is enough to exercise config materialization without
    loading real data. Asserts the printed JSON config's selection and key
    input defaults -- mirrors tests/test_off2on_fino_cli.py's offline half."""
    from rl_garden.training.offline import registry

    dataset_path = tmp_path / "fac.h5"

    registry.run_cli(
        [
            "fac",
            "--offline_dataset",
            str(dataset_path),
            "--device",
            "cpu",
            "--log_type",
            "none",
            "--log_dir",
            str(tmp_path / "runs"),
            "--print-config",
        ]
    )

    config = json.loads(capsys.readouterr().out)
    assert config["selection"] == {"training_phase": "offline", "algorithm": "fac"}
    inputs = config["inputs"]
    assert inputs["fac_alpha"] == 1.0
    assert inputs["fac_lambda"] == 1.0
    assert inputs["fac_threshold"] == "batch_adaptive"
    assert inputs["logp_method"] == "exact"
    assert inputs["logp_hutch_probes"] == 8
    assert inputs["weight_type"] == "linear"
    assert inputs["bc_lr"] == 3e-4
    assert inputs["bc_batch_size"] is None
    assert inputs["bc_pretrain_epochs"] == 250
    assert inputs["bc_pretrain_steps"] is None
    assert inputs["gamma"] == 0.995
    assert inputs["q_agg"] == "min"
    assert inputs["normalize_q_loss"] is True
