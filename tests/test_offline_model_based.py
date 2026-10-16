"""``OfflineModelBasedAlgorithm``: the three hooks run in order once per
``train()`` step, and checkpoints round-trip through ``BaseAlgorithm``."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_model_based import OfflineModelBasedAlgorithm


class _FakeBuffer:
    def __init__(self) -> None:
        self.sample_calls = 0

    def sample(self, batch_size: int) -> torch.Tensor:
        self.sample_calls += 1
        return torch.randn(batch_size, 3)


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.world_model = torch.nn.Linear(3, 3)
        self.actor = torch.nn.Linear(3, 1)


class _RecordingAlgorithm(OfflineModelBasedAlgorithm):
    _compatible_checkpoint_algorithms = ("_RecordingAlgorithm",)

    def __init__(self, env: OfflineEnvSpec, **kwargs) -> None:
        super().__init__(env=env, buffer_device="cpu", batch_size=4, device="cpu", **kwargs)
        self.calls: list[str] = []
        self._setup_model()

    def _setup_model(self) -> None:
        self.policy = _FakePolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.replay_buffer = _FakeBuffer()

    def _update_model(self, batch):
        self.calls.append("model")
        posterior = {"z": self.policy.world_model(batch)}
        loss = posterior["z"].pow(2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return {"model_loss": float(loss.detach())}, posterior

    def _update_actor_critic(self, batch, posterior):
        self.calls.append("actor_critic")
        assert posterior["z"].shape == batch.shape
        return {"actor_loss": float(self.policy.actor(posterior["z"].detach()).mean())}

    def _update_targets(self) -> None:
        self.calls.append("targets")

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("optimizer",)


def _env_spec() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
    )


def test_train_runs_three_hooks_in_order_per_step():
    agent = _RecordingAlgorithm(_env_spec())
    info = agent.train(3)
    assert agent.calls == ["model", "actor_critic", "targets"] * 3
    assert agent.replay_buffer.sample_calls == 3
    assert agent._global_update == 3
    assert set(info) == {"model_loss", "actor_loss"}
    assert agent.world_model is agent.policy.world_model


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    agent = _RecordingAlgorithm(_env_spec())
    agent.train(2)
    path = tmp_path / "ckpt.pt"
    agent.save(path)

    restored = _RecordingAlgorithm(_env_spec())
    restored.load(path)
    for (name, a), (_, b) in zip(agent.policy.state_dict().items(), restored.policy.state_dict().items()):
        torch.testing.assert_close(a, b, msg=name)
    assert restored._global_update == agent._global_update
    assert restored.optimizer.state_dict()["state"].keys() == agent.optimizer.state_dict()["state"].keys()
