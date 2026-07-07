from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import IQL, OfflineEnvSpec
from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer


def _state_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _fill_state(agent: IQL, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_dict(agent: IQL, steps: int = 4) -> None:
    env = agent.env
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_iql_state_train_step_and_checkpoint(tmp_path):
    agent = IQL(
        env=_state_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        net_arch={"pi": [16], "qf": [16], "vf": [16]},
        n_critics=3,
        critic_subsample_size=2,
        checkpoint_dir=str(tmp_path),
        std_log=False,
    )
    _fill_state(agent)

    info = agent.train(1)
    result = agent.learn_offline(2, save_filename="iql.pt")

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert torch.isfinite(torch.tensor(info["loss"]))
    assert "value_loss" in info
    assert "behavior_log_prob" in info
    assert result.final_checkpoint == tmp_path / "iql.pt"
    assert (tmp_path / "iql.pt").exists()


def test_iql_dict_uses_dict_replay_and_combined_encoder():
    agent = IQL(
        env=_dict_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        net_arch=[16],
        n_critics=2,
        critic_subsample_size=2,
        image_keys=("rgb",),
        image_fusion_mode="stack_channels",
        std_log=False,
    )
    _fill_dict(agent)

    info = agent.train(1)

    assert isinstance(agent.replay_buffer, DictReplayBuffer)
    assert torch.isfinite(torch.tensor(info["critic_loss"]))
    assert agent.policy.features_extractor.features_dim > 0


def _make_state_agent(**overrides) -> IQL:
    kwargs = dict(
        env=_state_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        net_arch={"pi": [16], "qf": [16], "vf": [16]},
        n_critics=3,
        critic_subsample_size=2,
        std_log=False,
    )
    kwargs.update(overrides)
    return IQL(**kwargs)


def test_expectile_loss_asymmetric_weighting():
    agent = _make_state_agent(expectile=0.7)
    diff = torch.tensor([2.0, -2.0])

    loss = agent._expectile_loss(diff)

    # diff>0 weighted by expectile, diff<=0 weighted by 1-expectile.
    torch.testing.assert_close(loss, torch.tensor([0.7 * 4.0, 0.3 * 4.0]))


def test_target_min_q_uses_target_critic(monkeypatch):
    agent = _make_state_agent()
    _fill_state(agent)
    recorded = {}
    original = agent.policy.min_q_value

    def _spy(*args, **kwargs):
        recorded.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent.policy, "min_q_value", _spy)
    features = agent.policy.extract_features(
        torch.randn(8, 4), stop_gradient=True
    )
    agent._target_min_q(features, torch.randn(8, 2).clamp(-1, 1))

    assert recorded["target"] is True
    assert recorded["subsample_size"] == agent.critic_subsample_size


def test_compute_losses_all_terms_finite_and_differentiable():
    agent = _make_state_agent()
    _fill_state(agent)
    data = agent._sample_train_batch(agent.batch_size)

    total_loss, metrics = agent._compute_losses(data)

    assert torch.isfinite(total_loss)
    for key in ("actor_loss", "critic_loss", "value_loss"):
        assert np.isfinite(metrics[key])
    total_loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in agent.policy.critic_value_and_encoder_parameters()
        if p.grad is not None
    ]
    assert any(g > 0 for g in grad_norms)


def test_iql_checkpoint_roundtrip_restores_weights(tmp_path):
    agent = _make_state_agent()
    _fill_state(agent)
    agent.train(2)

    path = agent.save(tmp_path / "iql.pt")

    loaded = _make_state_agent()
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_iql_box_obs_rejects_image_kwargs():
    with pytest.raises(ValueError, match="Box observation space"):
        _make_state_agent(image_keys=("rgb",))


def test_iql_unsupported_obs_space_raises_type_error():
    env = OfflineEnvSpec(
        spaces.Discrete(4),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=2,
    )
    with pytest.raises(TypeError, match="Box or Dict"):
        _make_state_agent(env=env)


def test_iql_polyak_update_moves_target_toward_online():
    agent = _make_state_agent(tau=0.5)
    with torch.no_grad():
        for p in agent.policy.critic.parameters():
            p.add_(1.0)
    target_before = [p.clone() for p in agent.policy.critic_target.parameters()]

    agent._polyak_update()

    for before, after, online in zip(
        target_before,
        agent.policy.critic_target.parameters(),
        agent.policy.critic.parameters(),
    ):
        assert not torch.equal(before, after)
        # tau=0.5 -> new target should be closer to (moved) online params.
        assert torch.allclose(after, 0.5 * before + 0.5 * online)
