"""Off-policy replay stores the true pre-reset observation on every episode end.

``OffPolicyAlgorithm.learn()`` writes ``infos["final_observation"]`` into the
replayed ``next_obs`` of every terminated or truncated row, whatever
``bootstrap_at_done`` says; the mode only sets the stored bootstrap flag.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from _supermimic_fakes import PointMassTrackingEnv

from rl_garden.algorithms import SAC
from rl_garden.algorithms.supermimic import SuperMimic

NUM_ENVS = 4
STEPS_PER_ENV = 200


def _record_steps(env) -> dict[str, list]:
    """Wrap ``env.step`` to record terminations, truncations, the returned
    observation and ``final_observation`` (the returned one when absent)."""
    record: dict[str, list] = {"terminated": [], "truncated": [], "returned": [], "final": []}
    original_step = env.step

    def step(actions):
        obs, rewards, terminated, truncated, infos = original_step(actions)
        final = infos.get("final_observation", obs)
        record["terminated"].append(terminated.clone())
        record["truncated"].append(truncated.clone())
        record["returned"].append({key: value.clone() for key, value in obs.items()})
        record["final"].append({key: value.clone() for key, value in final.items()})
        return obs, rewards, terminated, truncated, infos

    env.step = step
    return record


def _stacked(record: dict[str, list], name: str, key: str = "state") -> torch.Tensor:
    return torch.stack([step[key] for step in record[name]])


def _sac(bootstrap_at_done: str) -> tuple[SAC, dict[str, list]]:
    env = PointMassTrackingEnv(NUM_ENVS, extra_keys=False, max_episode_steps=12)
    record = _record_steps(env)
    agent = SAC(
        env=env,
        eval_env=None,
        buffer_size=NUM_ENVS * 256,
        buffer_device="cpu",
        device="cpu",
        learning_starts=NUM_ENVS * 16,
        batch_size=16,
        training_freq=2 * NUM_ENVS,
        utd=0.25,
        net_arch=[16],
        bootstrap_at_done=bootstrap_at_done,
        eval_freq=0,
        std_log=False,
        save_final_checkpoint=False,
        seed=0,
    )
    agent.learn(NUM_ENVS * STEPS_PER_ENV)
    return agent, record


@pytest.mark.parametrize("bootstrap_at_done", ["always", "never", "truncated"])
def test_replay_next_obs_is_the_final_observation_on_every_episode_end(bootstrap_at_done):
    agent, record = _sac(bootstrap_at_done)
    buffer = agent.replay_buffer
    assert not buffer.full and buffer.pos == STEPS_PER_ENV

    terminated = torch.stack(record["terminated"])
    truncated = torch.stack(record["truncated"])
    done = terminated | truncated
    assert bool((terminated & ~truncated).any()) and bool((truncated & ~terminated).any())

    obs = buffer.obs["state"][:STEPS_PER_ENV]
    next_obs = buffer.next_obs["state"][:STEPS_PER_ENV]
    final = _stacked(record, "final")
    returned = _stacked(record, "returned")

    assert torch.equal(next_obs[done], final[done])
    assert not bool((next_obs[done] == returned[done]).all(-1).any())
    # The same env's next replay row starts the new episode with the reset obs.
    has_next_row = done.clone()
    has_next_row[-1] = False
    next_row_obs = obs[1:][has_next_row[:-1]]
    assert torch.equal(next_row_obs, returned[has_next_row])
    assert not bool((next_obs[has_next_row] == next_row_obs).all(-1).any())

    interior = ~done[:-1]
    assert torch.equal(next_obs[:-1][interior], obs[1:][interior])

    expected_stop = {
        "always": torch.zeros_like(done),
        "never": done,
        "truncated": terminated,
    }[bootstrap_at_done]
    assert torch.equal(buffer.dones[:STEPS_PER_ENV].bool(), expected_stop)


@pytest.mark.parametrize("bootstrap_at_done", ["never", "truncated"])
def test_td_target_ignores_next_obs_on_stopped_rows(bootstrap_at_done):
    agent, _ = _sac(bootstrap_at_done)
    buffer = agent.replay_buffer
    rows = torch.arange(STEPS_PER_ENV).repeat_interleave(NUM_ENVS)
    envs = torch.arange(NUM_ENVS).repeat(STEPS_PER_ENV)
    data = buffer._index_batch(rows, envs)
    stopped = data.dones.bool()
    assert bool(stopped.any()) and bool((~stopped).any())

    generator = torch.Generator().manual_seed(1)
    noise = 10.0 * torch.randn(data.next_obs["state"].shape, generator=generator)
    stopped_column = stopped.unsqueeze(-1)
    swapped = dataclasses.replace(
        data, next_obs={"state": torch.where(stopped_column, noise, data.next_obs["state"])}
    )
    all_swapped = dataclasses.replace(data, next_obs={"state": noise})

    def target_and_critic_loss(batch):
        torch.manual_seed(0)
        target = agent._target_q(batch)
        torch.manual_seed(0)
        loss, _ = agent._critic_loss(batch)
        return target, loss.detach()

    target, loss = target_and_critic_loss(data)
    swapped_target, swapped_loss = target_and_critic_loss(swapped)
    assert torch.equal(target, swapped_target)
    assert torch.equal(loss, swapped_loss)
    # Not vacuous: next_obs does enter the target on bootstrapped rows.
    all_swapped_target, _ = target_and_critic_loss(all_swapped)
    assert not torch.equal(target[~stopped], all_swapped_target[~stopped])


def test_strict_window_ending_at_a_termination_reads_the_final_observation(monkeypatch):
    env = PointMassTrackingEnv(NUM_ENVS, extra_keys=False)
    record = _record_steps(env)
    agent = SuperMimic(
        env,
        heads="learned",
        wm_hidden_dims=(32, 32),
        head_hidden_dims=(16,),
        goal_encoder_dims=(32,),
        decoder_dims=(32,),
        critic_dims=(16,),
        horizon=4,
        wm_window=2,
        wm_batches=1,
        wm_batch_size=8,
        imagination_batch=8,
        warmup_updates=1,
        learning_starts=4 * NUM_ENVS,
        sim_steps_per_iteration=2,
        updates_per_iteration=1,
        buffer_size=1024,
        buffer_device="cpu",
        device="cpu",
        std_log=False,
        save_final_checkpoint=False,
        seed=0,
    )
    agent.learn(NUM_ENVS * 64)
    buffer = agent.replay_buffer
    steps = buffer.pos
    assert not buffer.full and steps == len(record["terminated"])
    horizon = buffer.horizon

    terminated = torch.stack(record["terminated"])
    ends_t, ends_env = terminated.nonzero(as_tuple=True)
    t0 = (ends_t - horizon + 1).clamp(min=0)
    whole_window = (
        (ends_t + 1 < steps)
        & (ends_t >= horizon - 1)
        & (buffer._ep_id[t0, ends_env] == buffer._ep_id[ends_t, ends_env])
    )
    ends_t, ends_env, t0 = ends_t[whole_window], ends_env[whole_window], t0[whole_window]
    assert ends_t.numel() > 0
    assert bool(buffer._valid_window_batch(t0, ends_env).all())

    monkeypatch.setattr(buffer, "_sample_valid_window_starts", lambda batch_size: (t0, ends_env))
    sample = buffer.sample(int(t0.numel()))

    last_obs = sample.obs["state"][-1]
    assert torch.equal(last_obs, _stacked(record, "final")[ends_t, ends_env])
    reset_obs = buffer.obs["state"][ends_t + 1, ends_env]
    assert not bool((last_obs == reset_obs).all(-1).any())
    assert bool(sample.terminated[-1].all())
