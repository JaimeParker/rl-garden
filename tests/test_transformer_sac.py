from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import TransformerSAC
from rl_garden.encoders.base import BaseFeaturesExtractor


class DummyVecEnv:
    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box, num_envs: int = 2
    ) -> None:
        self.num_envs = num_envs
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )

    def reset(self, seed: int | None = None):
        del seed
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0 + 1e-4)
        assert torch.all(actions >= -1.0 - 1e-4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), rewards, terminations, truncations, {}

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


class TransformerDoneVecEnv(DummyVecEnv):
    """Like DummyVecEnv, but env 0 terminates at a chosen global step and reports
    ``final_observation``, exercising the truncation/termination-boundary path."""

    def __init__(self, observation_space, action_space, done_at_step: int, num_envs: int = 2) -> None:
        super().__init__(observation_space, action_space, num_envs=num_envs)
        self._step = 0
        self._done_at_step = done_at_step

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        del actions
        self._step += 1
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: dict = {}
        if self._step == self._done_at_step:
            terminations[0] = True
            infos["final_observation"] = self._obs()
        obs = self._obs()  # gymnasium autoreset convention: next obs is already reset
        return obs, rewards, terminations, truncations, infos


class StructuredFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal fake extractor declaring a token_and_prop layout, purely to
    exercise TransformerSAC's ViT opt-out at construction time."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=16)

    def structured_feature_config(self):
        return {"layout": "token_and_prop", "num_patches": 4, "patch_dim": 4, "prop_dim": 0}

    def extract(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        return torch.randn(obs.shape[0], 16)


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


def _transformer_sac_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 32,
        "batch_size": 4,
        "learning_starts": 10,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [16],
        "burn_in_len": 2,
        "learning_len": 2,
        "forward_len": 1,
        "embed_dim": 8,
        "head_dim": 4,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "memory_len": 2,
    }


def test_transformer_sac_learn_one_iteration_state():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerSAC(env=env, **_transformer_sac_kwargs())

    agent.learn(total_timesteps=40)

    assert agent._global_step == 40
    assert agent.policy.recurrent_encoder is not None


def test_transformer_sac_handles_episode_termination_across_windows():
    env = TransformerDoneVecEnv(_state_space(), _action_space(), done_at_step=6)
    agent = TransformerSAC(env=env, **_transformer_sac_kwargs())

    agent.learn(total_timesteps=40)

    assert agent._global_step == 40
    sample = agent.replay_buffer.sample(4, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(sample.rewards).all()
    assert torch.isfinite(sample.discounts).all()


def test_transformer_sac_actor_loss_does_not_train_encoder_or_gtrxl_when_stop_gradient_actor():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = TransformerSAC(env=env, image_keys=("rgb",), **_transformer_sac_kwargs())
    assert agent._actor_stop_gradient() is True

    obs, _ = agent.env.reset(seed=agent.seed)
    agent._on_env_reset(obs)
    for _ in range(20):
        action, env_action, action_context = agent._rollout_action(obs, False)
        next_obs, rewards, terminations, truncations, infos = agent.env.step(env_action)
        stop_bootstrap = torch.zeros(2, dtype=torch.bool)
        need_final_obs = torch.zeros(2, dtype=torch.bool)
        replay_kwargs = agent._replay_buffer_add_kwargs(
            action_context, obs, next_obs, next_obs, infos, need_final_obs
        )
        replay_kwargs.update(agent._replay_buffer_step_kwargs(terminations, truncations))
        agent.replay_buffer.add(obs, next_obs, action, rewards, stop_bootstrap, **replay_kwargs)
        agent._post_rollout_step(action_context, terminations, truncations, infos)
        obs = next_obs

    data = agent.replay_buffer.sample(4, generator=torch.Generator().manual_seed(0))
    actor_loss, _ = agent._actor_loss_from_batch(data)

    agent.policy.zero_grad()
    actor_loss.backward()

    for name, param in agent.policy.features_extractor.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), f"encoder param {name} got actor grad"
    for name, param in agent.policy.recurrent_encoder.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), f"GTrXL param {name} got actor grad"


def test_transformer_sac_eval_resets_hidden_state_only_at_episode_boundary():
    env = DummyVecEnv(_state_space(), _action_space())
    eval_env = TransformerDoneVecEnv(_state_space(), _action_space(), done_at_step=3)
    agent = TransformerSAC(env=env, eval_env=eval_env, **_transformer_sac_kwargs())

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


def test_transformer_sac_priority_replay_updates_after_train_step():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerSAC(env=env, **_transformer_sac_kwargs())

    obs, _ = agent.env.reset(seed=agent.seed)
    agent._on_env_reset(obs)
    for _ in range(20):
        action, env_action, action_context = agent._rollout_action(obs, False)
        next_obs, rewards, terminations, truncations, infos = agent.env.step(env_action)
        stop_bootstrap, need_final_obs = torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
        replay_kwargs = agent._replay_buffer_add_kwargs(action_context, obs, next_obs, next_obs, infos, need_final_obs)
        replay_kwargs.update(agent._replay_buffer_step_kwargs(terminations, truncations))
        agent.replay_buffer.add(obs, next_obs, action, rewards, stop_bootstrap, **replay_kwargs)
        agent._post_rollout_step(action_context, terminations, truncations, infos)
        obs = next_obs

    tree_before = agent.replay_buffer._priority_tree.tree.clone()
    agent.train(1, compute_info=False)
    tree_after = agent.replay_buffer._priority_tree.tree

    assert not torch.equal(tree_before, tree_after)


def test_transformer_sac_dict_obs_smoke():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = TransformerSAC(env=env, image_keys=("rgb",), **_transformer_sac_kwargs())

    agent.learn(total_timesteps=40)

    assert agent._global_step == 40


def test_transformer_sac_rejects_token_and_prop_features():
    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(NotImplementedError):
        TransformerSAC(
            env=env,
            policy_kwargs={"features_extractor_class": StructuredFeaturesExtractor},
            **_transformer_sac_kwargs(),
        )


def test_transformer_sac_rejects_nstep_kwarg():
    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(ValueError):
        TransformerSAC(env=env, nstep=3, **_transformer_sac_kwargs())


def test_transformer_sac_rejects_integer_utd_greater_than_one():
    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(ValueError):
        TransformerSAC(env=env, utd=2, **_transformer_sac_kwargs())


def test_transformer_sac_rejects_burn_in_len_shorter_than_memory_len():
    env = DummyVecEnv(_state_space(), _action_space())
    kwargs = _transformer_sac_kwargs()
    kwargs["burn_in_len"] = 2
    kwargs["memory_len"] = 4
    with pytest.raises(ValueError):
        TransformerSAC(env=env, **kwargs)


def test_transformer_sac_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerSAC(env=env, **_transformer_sac_kwargs())
    agent.learn(total_timesteps=40)
    path = tmp_path / "transformer_sac.pt"
    agent.save(path)

    loaded = TransformerSAC(env=DummyVecEnv(_state_space(), _action_space()), **_transformer_sac_kwargs())
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
