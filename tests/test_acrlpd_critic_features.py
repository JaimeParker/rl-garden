from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.observations import ObsGroups
from test_acrlpd_smoke import _FakeEnv, _make_agent


@pytest.mark.parametrize("actor_dim,critic_dim", [(2, 3), (3, 3)])
@pytest.mark.parametrize("loss_path", ["target", "actor"])
def test_acrlpd_uses_critic_observation_group(actor_dim, critic_dim, loss_path, monkeypatch):
    env = _FakeEnv()
    env.single_observation_space = spaces.Dict({
        "state_actor": spaces.Box(-np.inf, np.inf, (actor_dim,), np.float32),
        "state_critic": spaces.Box(-np.inf, np.inf, (critic_dim,), np.float32),
    })
    env.observation_space = batch_space(env.single_observation_space, env.num_envs)
    agent = _make_agent(
        env=env,
        obs_groups=ObsGroups(actor=("state_actor",), critic=("state_critic",)),
    )
    obs = {
        "state_actor": torch.full((4, actor_dim), -2.0),
        "state_critic": torch.arange(4 * critic_dim, dtype=torch.float32).reshape(4, critic_dim),
    }
    expected = agent._critic_features(obs, target=loss_path == "target")
    assert expected.shape == (4, critic_dim)
    original = agent.policy.q_values_subsampled
    seen = []

    def record_features(features, actions, *, subsample_size, target):
        seen.append(features.detach().clone())
        assert target == (loss_path == "target")
        return original(features, actions, subsample_size=subsample_size, target=target)

    monkeypatch.setattr(agent.policy, "q_values_subsampled", record_features)
    if loss_path == "target":
        result = agent._target_q(SimpleNamespace(
            next_obs=obs, rewards=torch.zeros(4), discounts=torch.full((4,), 0.97),
        ))
        assert result.shape == (4, 1)
        assert not result.requires_grad
    else:
        result, log_prob = agent._actor_loss(obs)
        assert log_prob.shape == (4, 1)
        result.backward()
        assert any(p.grad is not None for p in agent.policy.actor.parameters())
    assert torch.isfinite(result).all()
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
