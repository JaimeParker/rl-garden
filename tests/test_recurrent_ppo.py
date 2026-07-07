from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import RecurrentPPO
from rl_garden.buffers.recurrent_rollout_buffer import RecurrentRolloutBuffer


class DummyVecEnv:
    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box
    ) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(
                action_space.low, (self.num_envs,) + action_space.shape
            ),
            high=np.broadcast_to(
                action_space.high, (self.num_envs,) + action_space.shape
            ),
            dtype=action_space.dtype,
        )
        self._step = 0

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)
        self._step += 1
        obs = self._obs()
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def _obs(self):
        if isinstance(self.single_observation_space, spaces.Dict):
            return {
                "rgb": torch.randint(
                    0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8
                ),
                "state": torch.randn(self.num_envs, 4),
            }
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


def _state_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _dict_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _ppo_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "target_kl": None,
        "net_arch": [16],
    }


class RecurrentDoneVecEnv:
    """Like DummyVecEnv, but env 0 terminates at a chosen global step and reports
    ``final_observation``, so tests can exercise the recurrent bootstrap-value and
    window-boundary hidden-state-reset paths that never fire with DummyVecEnv."""

    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box, done_at_step: int
    ) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )
        self._step = 0
        self._done_at_step = done_at_step

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        self._step += 1
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: dict = {}
        if self._step == self._done_at_step:
            terminations[0] = True
            infos["final_observation"] = self._obs()  # true terminal obs for env 0
        obs = self._obs()  # gymnasium autoreset convention: next obs is already reset
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None

    def _obs(self):
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


def _capture_train_losses(agent):
    captured: list[dict] = []
    original_train = agent.train

    def _wrapped():
        result = original_train()
        captured.append(result)
        return result

    agent.train = _wrapped
    return captured


def test_recurrent_rollout_buffer_preserves_time_order_and_env_axis():
    buffer = RecurrentRolloutBuffer(
        _state_space(), _action_space(), num_steps=4, num_envs=4, device="cpu"
    )
    for t in range(4):
        obs = torch.stack(
            [torch.full((4,), t * 100.0 + n) for n in range(4)], dim=0
        )
        buffer.add(
            obs,
            torch.zeros(4, 2),
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(4, 1),
            torch.zeros(4, 1),
        )
    buffer.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4))

    initial_hidden = torch.zeros(1, 4, 3)
    seen_envs: set[int] = set()
    for mb in buffer.get_sequences(num_minibatches=2, initial_hidden=initial_hidden, shuffle=False):
        assert mb.obs.shape == (4, 2, 4)
        assert mb.initial_hidden.shape == (1, 2, 3)
        for col in range(mb.obs.shape[1]):
            marker = mb.obs[0, col, 0].item()
            env_id = int(round(marker)) % 100
            seen_envs.add(env_id)
            for t in range(4):
                assert mb.obs[t, col, 0].item() == pytest.approx(t * 100.0 + env_id)
    assert seen_envs == {0, 1, 2, 3}


def test_recurrent_rollout_buffer_episode_starts_shift():
    buffer = RecurrentRolloutBuffer(
        _state_space(), _action_space(), num_steps=5, num_envs=2, device="cpu"
    )
    for t in range(5):
        dones = torch.zeros(2)
        if t == 2:
            dones[0] = 1.0  # env 0's episode ends as a result of acting on obs[2]
        buffer.add(
            torch.zeros(2, 4),
            torch.zeros(2, 2),
            torch.zeros(2),
            dones,
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        )
    buffer.compute_returns_and_advantage(torch.zeros(2), torch.zeros(2))

    initial_hidden = torch.zeros(1, 2, 3)
    mb = next(buffer.get_sequences(num_minibatches=1, initial_hidden=initial_hidden, shuffle=False))
    assert mb.episode_starts[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert mb.episode_starts[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_recurrent_rollout_buffer_get_sequences_requires_divisible_num_envs():
    buffer = RecurrentRolloutBuffer(
        _state_space(), _action_space(), num_steps=2, num_envs=3, device="cpu"
    )
    for _ in range(2):
        buffer.add(
            torch.zeros(3, 4),
            torch.zeros(3, 2),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3, 1),
            torch.zeros(3, 1),
        )
    buffer.compute_returns_and_advantage(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError):
        next(buffer.get_sequences(num_minibatches=2, initial_hidden=torch.zeros(1, 3, 3)))


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_recurrent_ppo_learn_one_iteration_state(rnn_type):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = RecurrentPPO(
        env=env,
        **_ppo_kwargs(),
        rnn_type=rnn_type,
        rnn_hidden_size=8,
    )
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert agent._global_update == 1
    assert agent.policy.recurrent_encoder is not None
    assert len(captured) == 1
    assert torch.isfinite(torch.tensor(captured[0]["loss"]))
    assert torch.isfinite(torch.tensor(captured[0]["value_loss"]))


def test_recurrent_ppo_handles_episode_termination_across_windows():
    # num_steps=2, num_envs=2 (from _ppo_kwargs()) => batch_size=4, so
    # total_timesteps=8 spans 2 rollout windows. env 0 terminates on the LAST
    # step of the first window, exercising both the non-early-return branch of
    # _compute_final_values (bootstrap value for a just-finished episode) and
    # window_initial_hidden's reset-mask at the start of window 2.
    env = RecurrentDoneVecEnv(_state_space(), _action_space(), done_at_step=2)
    agent = RecurrentPPO(env=env, **_ppo_kwargs(), rnn_hidden_size=8)
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=8)  # 2 rollout windows of num_steps=2

    assert agent._global_step == 8
    assert agent._global_update == 2
    assert len(captured) == 2
    for losses in captured:
        assert torch.isfinite(torch.tensor(losses["loss"]))
        assert torch.isfinite(torch.tensor(losses["value_loss"]))
    assert torch.isfinite(agent.rollout_buffer.values).all()


def test_recurrent_ppo_dict_obs_smoke():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = RecurrentPPO(
        env=env,
        **_ppo_kwargs(),
        image_keys=("rgb",),
        rnn_hidden_size=8,
    )
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert torch.isfinite(torch.tensor(captured[0]["loss"]))


def test_recurrent_ppo_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = RecurrentPPO(env=env, **_ppo_kwargs(), rnn_hidden_size=8)
    agent.learn(total_timesteps=4)
    path = tmp_path / "recurrent_ppo.pt"
    agent.save(path)

    loaded = RecurrentPPO(
        env=DummyVecEnv(_state_space(), _action_space()),
        **_ppo_kwargs(),
        rnn_hidden_size=8,
    )
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_recurrent_ppo_evaluate_with_eval_env_does_not_crash():
    """Regression test: periodic eval must use the stateful predict_recurrent
    path, not the stateless policy.predict() (which would feed raw pre-RNN
    features into actor/value heads built for recurrent_encoder.features_dim
    and crash with a shape mismatch)."""
    env = DummyVecEnv(_state_space(), _action_space())
    eval_env = DummyVecEnv(_state_space(), _action_space())
    agent = RecurrentPPO(
        env=env,
        eval_env=eval_env,
        **_ppo_kwargs(),
        rnn_hidden_size=8,
    )

    metrics = agent._evaluate()

    assert isinstance(metrics, dict)


def test_recurrent_ppo_eval_resets_hidden_state_only_at_episode_boundary():
    """Mirrors test_recurrent_sac_eval_resets_hidden_state_only_at_episode_boundary:
    _eval_episode_start must reset exactly on a genuine termination/truncation,
    not on every step and not never."""
    env = DummyVecEnv(_state_space(), _action_space())
    eval_env = RecurrentDoneVecEnv(_state_space(), _action_space(), done_at_step=3)
    agent = RecurrentPPO(env=env, eval_env=eval_env, **_ppo_kwargs(), rnn_hidden_size=8)

    agent.policy.eval()
    obs, _ = agent.eval_env.reset()
    agent._eval_start_hook()
    assert torch.all(agent._eval_episode_start == 1.0)

    for step in range(1, 6):
        env_action, critic_action = agent._eval_action_and_critic_action(obs)
        obs, rewards, terminations, truncations, infos = agent.eval_env.step(env_action)
        agent._eval_step_hook(obs, critic_action, rewards, terminations, truncations, infos)
        if step == 3:
            assert bool(terminations[0].item())
            assert agent._eval_episode_start[0].item() == 1.0
        else:
            assert agent._eval_episode_start[0].item() == 0.0
