"""Execute tutorial examples with small CPU budgets."""
import runpy
from pathlib import Path

import torch

from tests.test_tutorial_commands import WP3_EXAMPLE_COMMANDS


def test_sac_pendulum_smoke() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch02_sac_pendulum.py"
    agent = runpy.run_path(str(script))["main"](["--smoke"])
    assert agent.device.type == "cpu"
    assert agent._global_step >= 128
    assert agent._global_update > 0
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())


def test_ppo_pendulum_smoke() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch02_ppo_pendulum.py"
    agent = runpy.run_path(str(script))["main"](["--smoke"])
    assert agent.device.type == "cpu"
    assert agent._global_step >= 128
    assert agent._global_update > 0
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())


def test_encoder_config_smoke() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch03_encoder_config.py"
    agent = runpy.run_path(str(script))["main"](["--smoke"])
    assert agent.device.type == "cpu"
    assert agent._global_update == 1
    assert "rgb_camera" in agent.env.single_observation_space.spaces
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())


def test_save_resume_smoke(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch06_save_resume.py"
    result = runpy.run_path(str(script))["main"](["--smoke", "--output-dir", str(tmp_path)])
    assert result == {"saved_step": 50, "resumed_step": 66}
    assert (tmp_path / "agent.pt").is_file()


def test_offline_bc_smoke(tmp_path: Path) -> None:
    import h5py

    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch04_offline_bc.py"
    argv = [*WP3_EXAMPLE_COMMANDS["04"][2:]]
    argv[argv.index("--output-dir") + 1] = str(tmp_path)
    agent = runpy.run_path(str(script))["main"](argv)
    assert agent.device.type == "cpu"
    assert agent._global_update == 4
    assert agent.replay_buffer.sampleable_size == 128
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())
    with h5py.File(tmp_path / "pendulum.h5") as dataset:
        assert len(dataset) == 2
        trajectory = dataset["traj_0"]
        assert trajectory["obs"].shape == (65, 3)
        assert trajectory["actions"].shape == (64, 1)
        assert not trajectory["terminated"][:].any()
        assert trajectory["truncated"][:].sum() == 1
        assert trajectory["truncated"][-1]


def test_my_env_smoke(monkeypatch) -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch07_my_env.py"
    argv = [*WP3_EXAMPLE_COMMANDS["07"][2:]]
    agent = runpy.run_path(str(script))["main"](argv)
    assert agent.device.type == "cpu"
    assert agent._global_step >= 64
    assert agent._global_update > 0
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())

    import gymnasium as gym
    from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env

    # Force registration to occur through the source module, not the script run.
    monkeypatch.delitem(gym.registry, "TutorialLineReach-v0")
    monkeypatch.syspath_prepend(str(script.parent))
    env = make_gymnasium_env(GymnasiumEnvConfig(
        env_id="TutorialLineReach-v0", num_envs=1, seed=1, device="cpu",
        import_modules=("ch07_my_env",),
    ))
    try:
        observation, _ = env.reset()
        assert observation["state"].shape == (1, 1)
    finally:
        env.close()


def test_ch08_transformer_actor_smoke(monkeypatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/tutorials"))
    agent = importlib.import_module("ch08_transformer_ppo_actor").main(["--smoke"])
    assert agent._global_update > 0
    assert agent.policy.actor_capabilities.is_gaussian
    assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.actor.parameters())


def test_ch08_flow_actor_smoke(monkeypatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/tutorials"))
    online, offline = importlib.import_module("ch08_flow_sac_actor").main(["--smoke"])
    for agent in (online, offline):
        assert agent.device.type == "cpu"
        assert agent._global_update > 0
        assert not agent.policy.actor_capabilities.log_prob_is_density
        assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.actor.parameters())
    assert offline._global_update == 1


def test_ch08_backbone_smoke(monkeypatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/tutorials"))
    agent = importlib.import_module("ch08_backbone_extractor").main(["--smoke"])
    extractor = agent.policy.actor_extractor
    assert agent._global_update == 1
    assert all(not p.requires_grad and p.grad is None for p in extractor.vision.parameters())
    assert extractor.project.weight.grad is not None
    assert extractor.project.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p).all() for p in agent.policy.parameters())


def test_ch08_td3_shaped_smoke(monkeypatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/tutorials"))
    agent = importlib.import_module("ch08_td3_shaped_sac").main(["--smoke"])
    assert agent._global_update > 0
    assert not agent.policy.actor_capabilities.stochastic
    assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.actor.parameters())


def test_algorithm_family_constructions() -> None:
    script = Path(__file__).resolve().parents[1] / "examples/tutorials/ch_algorithms.py"
    agents = runpy.run_path(str(script))["main"](["--smoke"])
    assert set(agents) == {"ppo", "sac", "iql", "wsrl", "bc", "tdmpc2"}
    for agent in agents.values():
        assert agent.device.type == "cpu"
        assert agent._global_step == 0
        assert agent._global_update == 0
        parameters = list(agent.policy.parameters())
        assert parameters
        assert all(p.device.type == "cpu" and torch.isfinite(p).all() for p in parameters)
