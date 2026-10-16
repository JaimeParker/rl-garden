"""``RSSMPretrain`` smoke tests on a converted synthetic two-embodiment NCRL
dataset: losses fall over 50 steps (both data backends), checkpoint round
trip with the RSSM under ``actor_extractor.*``, and the ``rssm_pretrain``
CLI (``--print-config``, ``--dry-run``, ``--dataset_dir`` validation)."""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from _ncrl_synthetic import write_benchmark
from convert_ncrl_dataset import convert_dataset

from rl_garden.algorithms.rssm_pretrain import RSSMPretrain, sequence_dataset_env_spec
from rl_garden.world_models.rssm import RSSMSize

EMBODIMENTS = {"cheetah": (6, (14, 16, 12)), "quadruped": (12, (15, 13, 16))}
TINY_SIZE = RSSMSize(deter=32, hidden=16, discrete=4, units=32, cnn_depth=4)
TINY_KWARGS = {
    "rssm_size": TINY_SIZE,
    "stoch": 4,
    "batch_size": 8,
    "batch_length": 8,
    "lr": 1e-3,
    "compute_dtype": "float32",
    "device": "cpu",
    "buffer_device": "cpu",
    "log_freq": 0,
    "std_log": False,
}


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ncrl")
    write_benchmark(root / "src", EMBODIMENTS, image_size=32)
    convert_dataset(root / "src", root / "dst")
    return root / "dst"


def _agent(dataset_dir, **overrides) -> RSSMPretrain:
    kwargs = {**TINY_KWARGS, **overrides}
    return RSSMPretrain(sequence_dataset_env_spec(dataset_dir), dataset_dir=dataset_dir, **kwargs)


@pytest.mark.parametrize("backend", ["mmap", "buffer"])
def test_losses_fall_over_50_steps(dataset_dir, backend):
    agent = _agent(dataset_dir, dataset_backend=backend)
    assert agent.load_dataset() == sum(sum(lengths) for _, lengths in EMBODIMENTS.values())
    assert agent.world_model.heads == ()
    assert agent.padded_action_dim == 12
    history = []
    for _ in range(50):
        info = agent.train(1)
        history.append(info)
    assert set(history[0]) >= {"dyn_loss", "rep_loss", "rgb_default_loss", "recon_loss", "total_loss", "grad_norm"}
    assert not any(key.startswith(("rew", "con")) for key in history[0])

    def mean(key, rows):
        return sum(row[key] for row in rows) / len(rows)

    assert mean("recon_loss", history[-5:]) < mean("recon_loss", history[:5])
    assert mean("total_loss", history[-5:]) < mean("total_loss", history[:5])
    assert agent._global_update == 50


def test_actor_and_critic_are_not_trained(dataset_dir):
    agent = _agent(dataset_dir)
    actor_before = {k: v.clone() for k, v in agent.policy.actor.state_dict().items()}
    model_ids = {id(p) for p in agent.model_optimizer.param_groups[0]["params"]}
    assert model_ids == {id(p) for p in agent.world_model.parameters()}
    agent.train(2)
    for key, value in agent.policy.actor.state_dict().items():
        assert torch.equal(value, actor_before[key])


def test_checkpoint_round_trip(dataset_dir, tmp_path):
    agent = _agent(dataset_dir)
    agent.train(3)
    path = agent.save(tmp_path / "rssm.pt")

    checkpoint = torch.load(path, weights_only=False)
    policy_keys = checkpoint["state"]["policy"].keys()
    assert any(key.startswith("actor_extractor._deter_net.") for key in policy_keys)
    assert any(key.startswith("actor_extractor.decoder.") for key in policy_keys)
    assert not any("reward_head" in key or "cont_head" in key for key in policy_keys)
    assert "model_optimizer" in checkpoint["state"]["optimizers"]
    hyper = checkpoint["metadata"]["hyperparameters"]
    assert hyper["padded_action_dim"] == 12
    assert hyper["image_shape"] == {"rgb_default": [32, 32, 3]}
    assert hyper["rssm_size"]["deter"] == TINY_SIZE.deter

    restored = _agent(dataset_dir, seed=7)
    restored.load(path)
    for (key, a), (_, b) in zip(agent.policy.state_dict().items(), restored.policy.state_dict().items()):
        assert torch.equal(a, b), key
    assert restored._global_update == agent._global_update


def _cli(*args: str) -> dict:
    from rl_garden.training.offline import registry

    buf = io.StringIO()
    with redirect_stdout(buf):
        registry.run_cli(["rssm_pretrain", "--log_type", "none", *args])
    return json.loads(buf.getvalue())


def test_print_config_defaults():
    config = _cli("--dataset_dir", "/nonexistent", "--print-config")
    assert config["selection"] == {"training_phase": "offline", "algorithm": "rssm_pretrain"}
    inputs = config["inputs"]
    assert inputs["batch_size"] == 32 and inputs["batch_length"] == 64
    assert inputs["num_offline_steps"] == 200_000
    assert inputs["lr"] == 1e-4 and inputs["weight_decay"] == 1e-6 and inputs["grad_clip"] == 100.0
    assert inputs["size"] == "ncrl_m" and inputs["backbone"] == "ncrl"
    assert (inputs["dyn_scale"], inputs["rep_scale"], inputs["recon_scale"]) == (1.0, 0.1, 1.0)
    assert inputs["kl_free"] == 1.0 and inputs["kl_free_mode"] == "mean"
    assert inputs["compute_dtype"] == "bfloat16"
    assert inputs["padded_action_dim"] is None


def test_dry_run_constructs_agent(dataset_dir):
    config = _cli(
        "--dataset_dir", str(dataset_dir),
        "--device", "cpu", "--buffer_device", "cpu", "--compute_dtype", "float32",
        "--size", "size12M", "--deter", "32", "--hidden", "16", "--classes", "4", "--cnn_depth", "4",
        "--stoch", "4", "--batch_length", "8",
        "--backbone", "dreamer_v3",  # the ncrl backbone needs 64x64 images
        "--dry-run",
    )
    kwargs = config["algorithm"]["constructor_kwargs"]
    assert config["algorithm"]["target"].endswith("RSSMPretrain")
    assert kwargs["dataset_dir"] == str(dataset_dir)
    assert kwargs["deter"] == 32 and kwargs["kl_free_mode"] == "mean"
    assert config["runtime"]["train_action_space"]["shape"] == [12]


def test_missing_dataset_dir_rejected():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.algorithm_registry import ParsedCommand
    from rl_garden.training.offline import registry
    from rl_garden.training.offline._args import RSSMPretrainTrainingArgs

    registry.discover()
    command = ParsedCommand(RSSMPretrainTrainingArgs(), "rssm_pretrain", "print_config", None, {}, (), {})
    with pytest.raises(ConfigError, match="--dataset_dir is required for rssm_pretrain"):
        registry._validate_config(command)
