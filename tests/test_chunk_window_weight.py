"""ACRLPD, ACFQL, QAM and QGF give a chunk window that runs past its episode's
end zero weight in the losses their references weight with ``valid[..., -1]``
(qc ``agents/acrlpd.py:62``, ``agents/acfql.py:45``; qgf ``agents/qam.py:60,
81,96,210``, ``agents/qgf.py:92,102``), and the windows they keep match
Q-chunking's (qc ``utils/datasets.py:96-150``)."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import ACFQL, ACRLPD, OfflineEnvSpec
from rl_garden.algorithms.qam import QAM
from rl_garden.algorithms.qgf import QGF
from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer

OBS_DIM, ACTION_DIM, HORIZON, EPISODE_LEN = 4, 2, 3, 4


class _SpacesOnlyEnv:
    num_envs = 1
    single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    observation_space = batch_space(single_observation_space, 1)
    single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
    action_space = batch_space(single_action_space, 1)


def _offline_env() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32),
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
        num_envs=1,
    )


COMMON = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 200,
    "batch_size": 64,
    "net_arch": [16, 16],
}
ONLINE = {"learning_starts": 1, "training_freq": 1, "eval_freq": 0, "log_freq": 0}
AGENTS = {
    "acrlpd": lambda: ACRLPD(env=_SpacesOnlyEnv(), horizon_length=HORIZON, **ONLINE, **COMMON),
    "acfql": lambda: ACFQL(env=_SpacesOnlyEnv(), horizon_length=HORIZON, **ONLINE, **COMMON),
    "qam_ddpg": lambda: QAM(env=_offline_env(), horizon_length=HORIZON, flow_steps=3, **COMMON),
    "qam_iql": lambda: QAM(
        env=_offline_env(), horizon_length=HORIZON, flow_steps=3, critic_loss_type="iql", **COMMON
    ),
    "qgf": lambda: QGF(env=_offline_env(), horizon_length=HORIZON, denoise_steps=3, **COMMON),
}
# The losses the reference weights with valid[..., -1]; the others (ACFQL's
# per-position bc flow loss, QGF's bc loss) see the actions after the end.
WEIGHTED_LOSSES = {
    "acrlpd": ("critic_loss",),
    "acfql": ("critic_loss",),
    "qam_ddpg": ("critic_loss", "flow_loss"),
    "qam_iql": ("critic_loss", "value_loss", "flow_loss"),
    "qgf": ("critic_loss", "value_loss"),
}


def _fill_episodes(buffer, num_episodes: int = 40) -> None:
    """Episodes of ``EPISODE_LEN`` steps; odd ones terminate, even ones truncate."""
    generator = torch.Generator().manual_seed(0)
    for episode in range(num_episodes):
        for step in range(EPISODE_LEN):
            last = step == EPISODE_LEN - 1
            buffer.add(
                {"state": torch.randn(1, OBS_DIM, generator=generator)},
                {"state": torch.randn(1, OBS_DIM, generator=generator)},
                torch.rand(1, ACTION_DIM, generator=generator) * 2 - 1,
                torch.randn(1, generator=generator),
                torch.tensor([float(last and episode % 2 == 1)]),
                episode_end=torch.tensor([last]),
            )


def _losses_after_one_update(agent, batch) -> dict[str, float]:
    agent = copy.deepcopy(agent)
    agent.replay_buffer.sample = lambda batch_size: batch
    torch.manual_seed(123)
    return agent.train(1, compute_info=True)


@pytest.mark.parametrize("name", AGENTS)
def test_actions_after_the_episode_end_do_not_reach_the_weighted_losses(name):
    torch.manual_seed(0)
    agent = AGENTS[name]()
    _fill_episodes(agent.replay_buffer)
    torch.manual_seed(1)
    batch = agent.replay_buffer.sample(64)
    after_end = ~batch.valid
    assert after_end.any() and batch.valid[:, -1].any()

    noise = torch.randn_like(batch.actions) * 2 + 3
    after_end_perturbed = torch.where(after_end.unsqueeze(-1), noise, batch.actions)
    kept_perturbed = torch.where(after_end.unsqueeze(-1), batch.actions, noise)

    base = _losses_after_one_update(agent, batch)
    after_end_changed = _losses_after_one_update(
        agent, dataclasses.replace(batch, actions=after_end_perturbed)
    )
    kept_changed = _losses_after_one_update(
        agent, dataclasses.replace(batch, actions=kept_perturbed)
    )
    for key in WEIGHTED_LOSSES[name]:
        assert after_end_changed[key] == base[key], key
        assert kept_changed[key] != base[key], key


def test_the_weighted_mean_runs_over_the_whole_batch():
    torch.manual_seed(0)
    agent = AGENTS["qgf"]()
    _fill_episodes(agent.replay_buffer)
    torch.manual_seed(1)
    batch = agent.replay_buffer.sample(64)
    complete = batch.valid[:, -1]
    complete_only = type(batch)(
        **{
            field.name: (
                {k: v[complete] for k, v in value.items()}
                if isinstance(value, dict)
                else value[complete]
            )
            for field in dataclasses.fields(batch)
            for value in [getattr(batch, field.name)]
        }
    )
    full = _losses_after_one_update(agent, batch)
    kept = _losses_after_one_update(agent, complete_only)
    fraction = complete.float().mean().item()
    assert 0 < fraction < 1
    for key in ("critic_loss", "value_loss"):
        assert full[key] == pytest.approx(kept[key] * fraction, rel=1e-5)


def _qc_sequence(dataset: dict[str, np.ndarray], start: int, discount: float) -> dict:
    """``3rd_party/qc/utils/datasets.py``'s ``sample_sequence`` for one start."""
    idx = start + np.arange(HORIZON)
    rewards = np.zeros(HORIZON)
    masks = np.ones(HORIZON)
    terminals = np.zeros(HORIZON)
    valid = np.ones(HORIZON)
    rewards[0], masks[0], terminals[0] = (
        dataset["rewards"][idx[0]],
        dataset["masks"][idx[0]],
        dataset["terminals"][idx[0]],
    )
    for i in range(1, HORIZON):
        rewards[i] = rewards[i - 1] + dataset["rewards"][idx[i]] * discount**i
        masks[i] = min(masks[i - 1], dataset["masks"][idx[i]])
        terminals[i] = max(terminals[i - 1], dataset["terminals"][idx[i]])
        valid[i] = 1.0 - terminals[i - 1]
    return {
        "rewards": rewards,
        "masks": masks,
        "valid": valid,
        "actions": dataset["actions"][idx],
        "next_observations": dataset["next_observations"][idx],
    }


def test_kept_windows_match_q_chunking():
    gamma = 0.9
    buffer = ChunkedReplayBuffer(
        spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (1,), np.float32)}),
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
        num_envs=1,
        buffer_size=200,
        horizon_length=HORIZON,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )
    generator = np.random.default_rng(0)
    rows = 150
    terminated = generator.random(rows) < 0.1
    truncated = ~terminated & (generator.random(rows) < 0.1)
    dataset = {
        "observations": np.arange(rows, dtype=np.float32)[:, None],
        "next_observations": np.arange(rows, dtype=np.float32)[:, None] + 1000,
        "actions": generator.uniform(-1, 1, (rows, ACTION_DIM)).astype(np.float32),
        "rewards": generator.standard_normal(rows).astype(np.float32),
        # qc main.py:270-271: terminals=float(done), masks=1-terminated.
        "terminals": (terminated | truncated).astype(np.float32),
        "masks": 1.0 - terminated.astype(np.float32),
    }
    for row in range(rows):
        buffer.add(
            {"state": torch.tensor(dataset["observations"][row : row + 1])},
            {"state": torch.tensor(dataset["next_observations"][row : row + 1])},
            torch.tensor(dataset["actions"][row : row + 1]),
            torch.tensor(dataset["rewards"][row : row + 1]),
            torch.tensor(terminated[row : row + 1]),
            episode_end=torch.tensor(terminated[row : row + 1] | truncated[row : row + 1]),
        )

    torch.manual_seed(0)
    batch = buffer.sample(512)
    kept = 0
    for b in range(512):
        start = int(batch.obs["state"][b, 0])
        reference = _qc_sequence(dataset, start, gamma)
        np.testing.assert_array_equal(batch.valid[b].numpy(), reference["valid"].astype(bool))
        if reference["valid"][-1] == 0:
            continue
        kept += 1
        np.testing.assert_allclose(batch.actions[b].numpy(), reference["actions"])
        np.testing.assert_allclose(
            batch.rewards[b].item(), reference["rewards"][-1], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            batch.discounts[b].item(), gamma**HORIZON * reference["masks"][-1], rtol=1e-6
        )
        np.testing.assert_allclose(
            batch.next_obs["state"][b].numpy(), reference["next_observations"][-1]
        )
    assert 0 < kept < 512
