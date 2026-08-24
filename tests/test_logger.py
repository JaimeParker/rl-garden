"""Tests for Logger's TensorBoard-specific behavior: log_group directory
nesting and HParams recording at close(). No wandb network calls."""
from __future__ import annotations

from unittest.mock import MagicMock

from rl_garden.common.logger import Logger


def test_create_nests_tensorboard_writer_under_log_group(tmp_path):
    logger = Logger.create(
        log_type="tensorboard",
        log_dir=str(tmp_path),
        run_name="run1",
        log_group="env_a",
    )
    try:
        assert logger.writer.get_logdir() == str(tmp_path / "env_a" / "run1")
    finally:
        logger.writer.close()


def test_create_without_group_stays_flat(tmp_path):
    logger = Logger.create(
        log_type="tensorboard",
        log_dir=str(tmp_path),
        run_name="run1",
    )
    try:
        assert logger.writer.get_logdir() == str(tmp_path / "run1")
    finally:
        logger.writer.close()


def test_log_hparams_flattens_coerces_and_prefixes_metrics(tmp_path):
    config = {
        "args": {
            "lr": 0.001,
            "hidden_sizes": [256, 256],
            "seed": None,
            "env_id": "AntMaze",
        }
    }
    logger = Logger.create(
        log_type="tensorboard",
        log_dir=str(tmp_path),
        run_name="run1",
        config=config,
    )
    logger.add_scalar("eval/return", 12.5, step=100)
    logger.writer.add_hparams = MagicMock()

    logger.log_hparams()

    logger.writer.add_hparams.assert_called_once()
    call = logger.writer.add_hparams.call_args
    hparam_dict, metric_dict = call.args[:2]
    assert hparam_dict["lr"] == 0.001
    assert hparam_dict["hidden_sizes"] == "[256, 256]"
    assert hparam_dict["seed"] is None
    assert hparam_dict["env_id"] == "AntMaze"
    assert metric_dict == {"hparam/eval/return": 12.5}
    assert call.kwargs["run_name"] == "."

    logger.writer.close()


def test_log_hparams_noop_when_no_config_was_set():
    writer = MagicMock()
    logger = Logger(tensorboard=writer, log_type="tensorboard")

    logger.log_hparams()

    writer.add_hparams.assert_not_called()


def test_log_hparams_noop_for_none_backend():
    logger = Logger(log_type="none")
    logger.log_hparams()


def test_close_calls_log_hparams(tmp_path):
    logger = Logger.create(
        log_type="tensorboard",
        log_dir=str(tmp_path),
        run_name="run1",
        config={"a": 1},
    )
    logger.log_hparams = MagicMock()

    logger.close()

    logger.log_hparams.assert_called_once()
