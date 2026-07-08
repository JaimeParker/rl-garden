from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import RLPD


class DummyVecEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 2)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 2)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0 + 1e-4)
        assert torch.all(actions >= -1.0 - 1e-4)
        obs = torch.randn(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(**overrides) -> RLPD:
    kwargs = dict(
        env=DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
    )
    kwargs.update(overrides)
    return RLPD(**kwargs)


def test_rlpd_defaults_match_paper_recipe():
    agent = _agent()
    assert agent.n_critics == 10
    assert agent.critic_subsample_size == 2
    assert agent.critic_use_layer_norm is True


def test_rlpd_full_knob_set_learns_without_crashing():
    agent = _agent(
        use_pnorm=True,
        backbone_type="mlp_resnet",
        kernel_init="xavier_uniform",
        weight_decay=0.1,
        use_adamw=True,
        exclude_bias_from_decay=True,
        utd=4,
    )
    agent.learn(total_timesteps=16)
    assert agent._global_step == 16


def test_rlpd_dropout_learns_without_crashing_under_legacy_critic():
    # critic_impl="vmap" (the default) uses torch.func.vmap, which errors on
    # dropout's random op unless an explicit randomness mode is set -- a
    # pre-existing EnsembleQCritic constraint unrelated to RLPD, reproducible
    # on plain SACPolicy with critic_dropout_rate>0. critic_impl="legacy"
    # sidesteps it (no vmap), so that's what dropout needs here.
    agent = _agent(
        actor_dropout_rate=0.1,
        critic_dropout_rate=0.1,
        critic_impl="legacy",
        utd=4,
    )
    agent.learn(total_timesteps=16)
    assert agent._global_step == 16


def test_rlpd_offline_mixing_batch_shape():
    agent = _agent()
    agent.offline_replay_buffer = agent._build_prior_data_buffer(32)
    agent.offline_replay_buffer.add(
        torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1)
    )
    agent.replay_buffer.add(
        torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 2), torch.zeros(2), torch.zeros(2)
    )
    agent.offline_data_ratio = 0.5

    batch = agent._sample_train_batch(8)
    assert batch.obs.shape == (8, 4)
    assert batch.actions.shape == (8, 2)
    assert batch.rewards.shape == (8,)


def test_rlpd_exclude_bias_from_decay_does_not_stale_lr_schedule():
    # Regression test: exclude_bias_from_decay must be threaded through
    # sac.py's own _setup_model() rather than rebuilt post-hoc in RLPD --
    # a post-hoc optimizer rebuild would leave self._lr_schedulers bound to
    # the discarded optimizer object, silently freezing the LR schedule.
    agent = _agent(
        weight_decay=0.1,
        use_adamw=True,
        exclude_bias_from_decay=True,
        lr_schedule="linear_warmup",
        lr_warmup_steps=4,
        learning_starts=0,
    )
    lr_before = agent.actor_optimizer.param_groups[0]["lr"]
    agent.learn(total_timesteps=8)
    lr_after = agent.actor_optimizer.param_groups[0]["lr"]
    assert lr_after != lr_before


def test_rlpd_checkpoint_roundtrip(tmp_path):
    agent = _agent()
    agent.learn(total_timesteps=16)
    path = tmp_path / "rlpd.pt"
    agent.save(path)

    loaded = _agent()
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
