from __future__ import annotations

import os
import pickle
import sys

import torch


def _write_pkl(path: str, entries: list[dict]) -> None:
    with open(path, "wb") as f:
        pickle.dump(entries, f)


def test_train_main_runs_and_saves_a_checkpoint(tmp_path, monkeypatch):
    success = [{"obs": {"wrist": torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)}} for _ in range(4)]
    failure = [{"obs": {"wrist": torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)}} for _ in range(4)]
    _write_pkl(os.path.join(tmp_path, "exp_success_1.pkl"), success)
    _write_pkl(os.path.join(tmp_path, "exp_failure_1.pkl"), failure)

    output_dir = os.path.join(tmp_path, "ckpt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--success_paths",
            os.path.join(tmp_path, "*success*.pkl"),
            "--failure_paths",
            os.path.join(tmp_path, "*failure*.pkl"),
            "--output_dir",
            output_dir,
            "--num_epochs",
            "1",
            "--batch_size",
            "4",
            "--device",
            "cpu",
            "--pretrained_weights",
            "none",
        ],
    )

    from rl_garden.models.reward.success.train import main

    main()

    checkpoint_path = os.path.join(output_dir, "success_classifier.pt")
    assert os.path.exists(checkpoint_path)

    state_dict = torch.load(checkpoint_path, weights_only=False)
    assert any("head" in k for k in state_dict)
