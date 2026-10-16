"""PWM smoke tests on a synthetic 3-task mmap dataset and a small, randomly
initialized ``MultitaskWorldModel`` (no simulator/hardware)."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.offline import OfflineEnvSpec, run_offline_pretraining
from rl_garden.algorithms.pwm import PWM
from rl_garden.envs.wrappers.multitask_pad import MultitaskPadWrapper
from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel

TASKS = ["task_a", "task_b", "task_c"]
OBS_DIMS = [4, 6, 5]
ACTION_DIMS = [2, 3, 2]
EPISODE_LENGTHS = [10, 8, 12]
WM_ARCH = dict(task_dim=8, latent_dim=16, enc_dim=8, num_enc_layers=2, mlp_dim=16, simnorm_dim=8, num_bins=11)


def make_world_model(seed: int = 0) -> MultitaskWorldModel:
    torch.manual_seed(seed)
    world_model = MultitaskWorldModel(
        num_tasks=len(TASKS), obs_dims=OBS_DIMS, action_dims=ACTION_DIMS, vmin=-10.0, vmax=10.0, **WM_ARCH
    )
    world_model.apply_init()
    # apply_init zeroes the reward head's last layer (a constant reward, no
    # action gradient); re-randomize it so imagination has a signal.
    torch.nn.init.normal_(world_model._reward[-1].weight, std=0.5)
    return world_model


def write_converted_world_model(directory: Path, seed: int = 0) -> Path:
    """Saves a random world model in ``convert_tdmpc2_checkpoint.py``'s format."""
    world_model = make_world_model(seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(world_model.state_dict(), directory / "world_model.pt")
    manifest = {
        "num_tasks": len(TASKS),
        "task_dim": WM_ARCH["task_dim"],
        "obs_dim": max(OBS_DIMS),
        "action_dim": max(ACTION_DIMS),
        "latent_dim": WM_ARCH["latent_dim"],
        "enc_dim": WM_ARCH["enc_dim"],
        "num_enc_layers": WM_ARCH["num_enc_layers"],
        "mlp_dim": WM_ARCH["mlp_dim"],
        "num_bins": WM_ARCH["num_bins"],
        "action_dims": ACTION_DIMS,
        "tasks": TASKS,
        "simnorm_dim": WM_ARCH["simnorm_dim"],
        "vmin": -10.0,
        "vmax": 10.0,
        "source": "synthetic",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def write_dataset(directory: Path, episodes_per_task: int = 4) -> Path:
    """Synthetic converted multitask dataset (``manifest.json`` +
    ``<task>/episodes.pt``), unpadded per task."""
    generator = torch.Generator().manual_seed(0)
    for task, obs_dim, action_dim, length in zip(TASKS, OBS_DIMS, ACTION_DIMS, EPISODE_LENGTHS):
        (directory / task).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs": torch.randn(episodes_per_task, length, obs_dim, generator=generator),
                "action": torch.rand(episodes_per_task, length, action_dim, generator=generator) * 2 - 1,
                "reward": torch.rand(episodes_per_task, length, generator=generator),
            },
            directory / task / "episodes.pt",
        )
    manifest = {
        "tasks": TASKS,
        "obs_dims": OBS_DIMS,
        "action_dims": ACTION_DIMS,
        "episode_lengths": EPISODE_LENGTHS,
        "num_episodes": [episodes_per_task] * len(TASKS),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def env_spec() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        observation_space=spaces.Box(-1e6, 1e6, shape=(max(OBS_DIMS),), dtype="float32"),
        action_space=spaces.Box(-1.0, 1.0, shape=(max(ACTION_DIMS),), dtype="float32"),
        num_envs=1,
    )


def make_agent(tmp_path: Path, *, task_idx: int = 0, mmap_name: str = "buf", world_model=None, **kwargs) -> PWM:
    world_model_path = world_model or write_converted_world_model(tmp_path / "wm")
    params = dict(
        buffer_size=200,
        batch_size=16,
        horizon=3,
        actor_units=(32, 32),
        critic_units=(32, 32),
        critic_iterations=2,
        critic_batches=2,
        lr_decay_steps=200,
        device="cpu",
        seed=0,
        std_log=False,
    )
    params.update(kwargs)
    return PWM(
        env=env_spec(),
        world_model_checkpoint=str(world_model_path),
        task_idx=task_idx,
        mmap_dir=str(tmp_path / mmap_name),
        **params,
    )


def fill(agent: PWM, task_idx: int = 0, episodes: int = 6) -> None:
    obs_dim, action_dim, length = OBS_DIMS[task_idx], ACTION_DIMS[task_idx], EPISODE_LENGTHS[task_idx]
    for _ in range(episodes):
        obs = torch.zeros(length, max(OBS_DIMS))
        obs[:, :obs_dim] = torch.randn(length, obs_dim)
        action = torch.zeros(length, max(ACTION_DIMS))
        action[:, :action_dim] = torch.rand(length, action_dim) * 2 - 1
        agent.replay_buffer.load_episode(obs, action, torch.rand(length), task_idx)


def _start_latents(agent: PWM) -> torch.Tensor:
    batch = agent._sample_train_batch()
    _, posterior = agent._update_model(batch)
    return posterior["z"]


def test_actor_gradient_reaches_actor_but_not_world_model(tmp_path):
    agent = make_agent(tmp_path)
    fill(agent)
    rollout = agent._actor_loss(_start_latents(agent))
    rollout["loss"].backward()

    actor_grads = [p.grad for p in agent.policy.actor.mu_net.parameters()]
    assert all(g is not None for g in actor_grads)
    assert sum(float(g.abs().sum()) for g in actor_grads) > 0
    assert agent.policy.actor.logstd.grad is not None and agent.policy.actor.logstd.grad.abs().sum() > 0
    for name, parameter in agent.bound_world_model.named_parameters():
        assert not parameter.requires_grad, name
        assert parameter.grad is None, name


def test_critic_targets_match_hand_computed_lambda_return(tmp_path):
    agent = make_agent(tmp_path, gamma=0.9, lam=0.8)
    rewards = torch.tensor([[1.0], [2.0], [3.0]])  # (H=3, B=1)
    values = torch.tensor([[0.5], [1.5], [2.5], [4.0]])  # V(z_0..z_3)
    g, lam = 0.9, 0.8
    g2 = 3.0 + g * 4.0
    g1 = 2.0 + g * ((1 - lam) * 2.5 + lam * g2)
    g0 = 1.0 + g * ((1 - lam) * 1.5 + lam * g1)
    expected = torch.tensor([[g0], [g1], [g2]])
    torch.testing.assert_close(agent._critic_targets(rewards, values), expected)


def test_padded_action_dims_are_zero(tmp_path):
    agent = make_agent(tmp_path, task_idx=0)  # task_a: 2 real dims of 3
    fill(agent)
    z0 = _start_latents(agent)
    action = agent._policy_action({"z": z0})
    assert action.shape == (16, 3)
    assert torch.all(action[:, 2] == 0)
    assert torch.any(action[:, :2] != 0)
    eval_action = agent._eval_action({"state": torch.randn(4, max(OBS_DIMS))})
    assert torch.all(eval_action[:, 2] == 0)


def test_train_decreases_actor_loss(tmp_path):
    agent = make_agent(tmp_path)
    fill(agent)
    metrics = [agent.train(1) for _ in range(200)]
    raw_losses = [m["actor_loss_raw"] for m in metrics]
    rewards = [m["imagined_reward"] for m in metrics]
    print(f"actor_loss_raw first5={raw_losses[:5]} last5={raw_losses[-5:]}")
    print(f"imagined_reward first5={rewards[:5]} last5={rewards[-5:]}")
    assert np.all(np.isfinite(raw_losses))
    assert np.mean(raw_losses[-20:]) < np.mean(raw_losses[5:25])
    # The raw loss also falls while the critic converges; the model reward
    # under the policy (frozen world model) only rises if the actor learns.
    assert np.mean(rewards[-20:]) > np.mean(rewards[5:25])


def test_checkpoint_round_trip(tmp_path):
    agent = make_agent(tmp_path)
    fill(agent)
    agent.train(3)
    path = agent.save(tmp_path / "ckpt" / "pwm.pt")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert not any(key.startswith("world_model") for key in payload["state"]["policy"])
    assert set(payload["state"]["optimizers"]) == {"actor_optimizer", "critic_optimizer"}
    assert payload["metadata"]["hyperparameters"]["world_model_source"]["format"] == "converted_tdmpc2_checkpoint"

    restored = make_agent(tmp_path, mmap_name="buf2")
    restored.load(path, load_replay_buffer=False)
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value)
    assert restored.policy.ret_rms.count.item() == agent.policy.ret_rms.count.item()
    assert restored._global_update == 3
    assert restored.actor_optimizer.state_dict()["state"].keys() == agent.actor_optimizer.state_dict()["state"].keys()


def test_runs_without_eval_env(tmp_path):
    agent = make_agent(tmp_path, eval_freq=2)
    fill(agent)
    assert agent.eval_env is None
    result = run_offline_pretraining(agent, num_steps=4, eval_freq=2, log_freq=0, std_log=False)
    assert result.final_update == 4
    assert np.isfinite(result.last_metrics["critic_loss"])


def test_loads_world_model_from_tdmpc2_multitask_checkpoint(tmp_path):
    from rl_garden.algorithms.tdmpc2_multitask import TDMPC2Multitask

    source = TDMPC2Multitask(
        env=env_spec(),
        tasks=TASKS,
        obs_dims=OBS_DIMS,
        action_dims=ACTION_DIMS,
        episode_lengths=EPISODE_LENGTHS,
        mmap_dir=str(tmp_path / "tdmpc2_buf"),
        buffer_size=200,
        horizon=2,
        num_q=2,
        device="cpu",
        **WM_ARCH,
    )
    checkpoint = source.save(tmp_path / "tdmpc2.pt")
    agent = make_agent(tmp_path, world_model=checkpoint, mmap_name="buf_ckpt", task_idx=1)
    loaded = agent.bound_world_model.world_model.state_dict()
    for key, value in source.policy.world_model.state_dict().items():
        torch.testing.assert_close(loaded[key], value)
    assert agent.world_model_source["tasks"] == TASKS
    assert torch.equal(agent.action_mask, torch.ones(3))


def test_world_model_horizon_and_rho_mismatch_warns(tmp_path):
    import warnings

    from rl_garden.algorithms.tdmpc2_multitask import TDMPC2Multitask

    def checkpoint(name: str, horizon: int, rho: float) -> Path:
        source = TDMPC2Multitask(
            env=env_spec(),
            tasks=TASKS,
            obs_dims=OBS_DIMS,
            action_dims=ACTION_DIMS,
            episode_lengths=EPISODE_LENGTHS,
            mmap_dir=str(tmp_path / f"{name}_buf"),
            buffer_size=200,
            horizon=horizon,
            rho=rho,
            num_q=2,
            device="cpu",
            **WM_ARCH,
        )
        return source.save(tmp_path / f"{name}.pt")

    with pytest.warns(UserWarning, match="horizon 2") as record:
        agent = make_agent(tmp_path, world_model=checkpoint("short", 2, 0.5), mmap_name="buf_short")
    assert any("rho 0.5" in str(w.message) for w in record)
    assert (agent.world_model_source["horizon"], agent.world_model_source["rho"]) == (2, 0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matched = make_agent(tmp_path, world_model=checkpoint("match", 3, 0.99), mmap_name="buf_match")
        converted = make_agent(tmp_path, mmap_name="buf_conv")
    assert matched.world_model_source["horizon"] == 3
    assert (converted.world_model_source["horizon"], converted.world_model_source["rho"]) == ("unknown", "unknown")


def test_first_update_uses_the_initial_learning_rate(tmp_path):
    agent = make_agent(tmp_path, lr_decay_steps=10)
    fill(agent)
    metrics = [agent.train(1) for _ in range(3)]
    assert metrics[0]["actor_lr"] == agent.actor_lr and metrics[0]["critic_lr"] == agent.critic_lr
    step = (agent.lr_final - agent.actor_lr) / 10
    assert metrics[2]["actor_lr"] == pytest.approx(agent.actor_lr + 2 * step)


def test_critic_minibatches_keep_the_remainder_rows(tmp_path):
    agent = make_agent(tmp_path, critic_iterations=1, critic_batches=5)
    rows = []
    agent.policy.critic.register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    zs = torch.randn(3, 16, 16)  # H * B = 48 rows, 48 // 5 = 9 per minibatch
    agent._update_critic(zs, torch.randn(3, 16))
    assert rows == [9, 9, 9, 9, 9, 3]


class FakeStateVectorEnv(gym.vector.VectorEnv):
    """Two-env torch vector env with a ``"state"`` Dict obs; every episode
    truncates after ``episode_length`` steps with reward 1 per step."""

    def __init__(self, obs_dim: int, action_dim: int, num_envs: int = 2, episode_length: int = 5) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.episode_length = episode_length
        self.last_actions = None
        self._t = 0

    def _obs(self):
        return {"state": torch.ones(self.num_envs, self.single_observation_space["state"].shape[0])}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, actions):
        self.last_actions = actions
        self._t += 1
        truncated = torch.full((self.num_envs,), self._t >= self.episode_length)
        if self._t >= self.episode_length:
            self._t = 0
        return (
            self._obs(),
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            truncated,
            {},
        )


def test_evaluates_through_multitask_pad_wrapper(tmp_path):
    fake = FakeStateVectorEnv(OBS_DIMS[0], ACTION_DIMS[0])
    eval_env = MultitaskPadWrapper(
        fake,
        obs_dim=max(OBS_DIMS),
        action_dim=max(ACTION_DIMS),
        task_idx=0,
        task_obs_dim=OBS_DIMS[0],
        task_action_dim=ACTION_DIMS[0],
    )
    agent = make_agent(tmp_path, eval_env=eval_env, num_eval_episodes=2, num_eval_steps=20)
    metrics = agent._evaluate()
    assert metrics["episodes_completed"] == 2.0
    assert metrics["return"] == pytest.approx(5.0)
    assert fake.last_actions.shape == (2, ACTION_DIMS[0])
