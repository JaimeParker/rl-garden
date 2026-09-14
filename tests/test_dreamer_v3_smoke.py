"""DreamerV3 smoke tests: a fake vectorized env (state and rgb, no
simulator/hardware), a tiny custom ``RSSMSize``, ``compute_dtype="float32"``
-- learns a few hundred steps past ``learning_starts``, finite losses, carry
write-back exercised, checkpoint save/load round-trip (incl. slow_critic/
return_ema), and the deterministic eval path. Runtime target: well under a
minute on CPU.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.dreamer_v3 import DreamerV3
from rl_garden.world_models.rssm import RSSMSize

_TINY_SIZE = RSSMSize(deter=32, hidden=16, discrete=4, units=32, cnn_depth=4)


class DummyVecEnv:
    """Fake vectorized env: state-only or state+rgb Dict obs, an
    episode-length-5 termination on env 0 exercising the final_observation/
    is_terminal/is_last plumbing (same pattern as
    ``tests/test_recurrent_sac.py``'s ``RecurrentDoneVecEnv``)."""

    def __init__(self, action_space: spaces.Box, num_envs: int = 4, rgb: bool = False) -> None:
        self.num_envs = num_envs
        self.rgb = rgb
        spaces_dict = {"state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)}
        if rgb:
            spaces_dict["rgb_cam"] = spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
        self.single_observation_space = spaces.Dict(spaces_dict)
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )
        self._t = 0

    def _obs(self):
        obs = {"state": torch.randn(self.num_envs, 4)}
        if self.rgb:
            obs["rgb_cam"] = torch.randint(0, 256, (self.num_envs, 32, 32, 3), dtype=torch.uint8)
        return obs

    def reset(self, seed: int | None = None):
        del seed
        self._t = 0
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0 + 1e-4)
        assert torch.all(actions >= -1.0 - 1e-4)
        self._t += 1
        rewards = torch.randn(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: dict = {}
        if self._t % 5 == 0:
            terminations[0] = True
            infos["final_observation"] = self._obs()
            infos["final_info"] = {"episode": {"return": torch.ones(self.num_envs)}}
            done_mask = torch.zeros(self.num_envs, dtype=torch.bool)
            done_mask[0] = True
            infos["_final_info"] = done_mask
        obs = self._obs()  # gymnasium autoreset convention: next obs already reset
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


_TINY_KWARGS = dict(
    device="cpu",
    buffer_device="cpu",
    buffer_size=400,
    batch_size=4,
    batch_length=6,
    train_ratio=24.0,  # utd == 1 with batch_size*batch_length == 24 -- small, fast local run
    imag_horizon=3,
    learning_starts=40,
    stoch=4,
    rssm_size=_TINY_SIZE,
    compute_dtype="float32",
    discount_horizon=20.0,
    eval_freq=0,
    log_freq=0,
)


def _agent(num_envs: int = 4, rgb: bool = False, **overrides) -> DreamerV3:
    env = DummyVecEnv(_action_space(), num_envs=num_envs, rgb=rgb)
    kwargs = dict(_TINY_KWARGS)
    kwargs.update(overrides)
    return DreamerV3(env=env, **kwargs)


def test_learn_past_learning_starts_state_finite_losses():
    agent = _agent()
    agent.learn(total_timesteps=200)  # well past learning_starts=40
    assert agent._global_step >= 200
    assert agent._global_update > 0


def test_learn_past_learning_starts_rgb_finite_losses():
    agent = _agent(rgb=True)
    agent.learn(total_timesteps=120)
    assert agent._global_step >= 120
    assert agent._global_update > 0


def test_train_step_losses_are_finite():
    agent = _agent()
    agent.learn(total_timesteps=60)  # populate the buffer past learning_starts
    info = agent.train(3, compute_info=True)
    assert len(info) > 0
    for key, value in info.items():
        assert math.isfinite(value), f"{key}={value} is not finite"


def test_carry_write_back_changes_stored_carry():
    agent = _agent()
    agent.learn(total_timesteps=60)
    before = agent.replay_buffer._carry["deter"].clone()
    agent.train(5, compute_info=False)
    after = agent.replay_buffer._carry["deter"]
    assert not torch.equal(before, after), "write_back_carry did not change any stored carry"


def test_eval_deterministic_path():
    env = DummyVecEnv(_action_space(), num_envs=4)
    eval_env = DummyVecEnv(_action_space(), num_envs=4)
    agent = DreamerV3(env=env, eval_env=eval_env, **_TINY_KWARGS)
    agent.learn(total_timesteps=1)  # exercises _on_env_reset / _rollout_action once
    metrics = agent._evaluate()
    assert "return" in metrics
    assert math.isfinite(metrics["return"])


def test_checkpoint_roundtrip_includes_slow_critic_and_return_ema(tmp_path):
    agent = _agent()
    agent.learn(total_timesteps=80)
    path = tmp_path / "dreamer_v3.pt"
    agent.save(path)

    loaded = _agent()
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key

    # Explicitly confirm slow_critic and return_ema round-trip (plan section
    # E: "state_dict must include slow_critic and return_ema buffers") --
    # both are plain nn.Module children of DreamerPolicy, so the generic
    # policy.state_dict() equality check above already covers them; these
    # extra asserts pin exactly which keys prove it.
    assert any(k.startswith("slow_critic.") for k in agent.policy.state_dict())
    assert any(k.startswith("return_ema.") for k in agent.policy.state_dict())


def test_rgb_checkpoint_roundtrip(tmp_path):
    agent = _agent(rgb=True)
    agent.learn(total_timesteps=80)
    path = tmp_path / "dreamer_v3_rgb.pt"
    agent.save(path)

    loaded = _agent(rgb=True)
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


# ---------------------------------------------------------------------------
# Fixer pass (2026-09-14, plan model-based-base Part 2 / DreamerV3): gradient
# isolation, rollout-reset wiring, contdisc -> imagination disc.
# ---------------------------------------------------------------------------


def test_imagination_losses_isolated_from_repval_and_model_losses():
    """The imagination-based actor/critic losses read only a FROZEN, no-grad
    snapshot of the RSSM (``DreamerV3._imagination_losses``'s own docstring
    -- ``frozen_rssm``/``feat_all`` are computed under ``torch.no_grad()``),
    so they must carry ZERO gradient into any live RSSM/encoder/decoder
    parameter; the replay-based ``repval`` loss backprops into the LIVE
    world model (``feat_live = world_model.features(posterior)``, live/
    graph-attached) and must carry non-zero gradient into at least one of
    its parameters."""
    agent = _agent()
    agent.learn(total_timesteps=60)  # populate the buffer past learning_starts
    batch = agent.replay_buffer.sample(agent.batch_size)
    _, posterior = agent._update_model(batch)
    world_model_params = list(agent.world_model.parameters())

    imag = agent._imagination_losses(batch, posterior)

    agent.optimizer.zero_grad(set_to_none=True)
    (imag["actor_loss"] + imag["critic_loss"]).backward()
    assert all(p.grad is None for p in world_model_params)

    agent.optimizer.zero_grad(set_to_none=True)
    imag["repval_loss"].backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in world_model_params)

    del agent._pending_model_losses


def test_rollout_reset_zeros_only_the_terminated_envs_carry():
    """Drives ``_rollout_action``/``_post_rollout_step`` directly (not
    ``agent.learn()``, so the exact termination step is known) through
    ``DummyVecEnv``'s episode-length-5 termination on env 0 alone. The NEXT
    ``_rollout_action`` call must fold env 0's new embedding into a state
    built from a FULLY RESET carry (``RSSM.initial_state`` zeros + a zeroed
    previous action, matching what ``RSSM.observe``'s ``is_first`` masking
    does internally -- see ``tests/test_rssm.py``'s
    ``test_observe_resets_state_and_action_where_is_first``), while another
    env that never terminated keeps depending on its own accumulated
    history -- ``DreamerV3._post_rollout_step``'s own docstring: "the RSSM
    reset happens inside observe(), not by zeroing in the loop"."""
    agent = _agent(num_envs=4)
    obs, _ = agent.env.reset(seed=agent.seed)
    agent._on_env_reset(obs)

    for _ in range(5):  # DummyVecEnv terminates env 0 on every 5th step()
        action, env_action, action_context = agent._rollout_action(obs, learning_has_started=True)
        next_obs, rewards, terminations, truncations, infos = agent.env.step(env_action)
        agent._post_rollout_step(action_context, terminations, truncations, infos)
        obs = next_obs

    assert bool(agent._rollout_is_first[0])
    assert not bool(agent._rollout_is_first[1])
    carry_before = agent._rollout_state["deter"].clone()
    assert float(carry_before[0].abs().sum()) > 0.0  # env 0 has real history before the reset

    action, env_action, action_context = agent._rollout_action(obs, learning_has_started=True)
    new_state = action_context["new_state"]

    obs_device = agent._obs_to_policy_device(obs)
    with torch.no_grad():
        embed = agent.world_model.encode(obs_device)
    zero_state = agent.world_model.initial_state(agent.num_envs, agent.device)
    zero_action = torch.zeros(agent.num_envs, agent.policy.action_dim, device=agent.device)
    expected = agent.world_model.observe(
        zero_state, zero_action, embed, torch.ones(agent.num_envs, dtype=torch.bool, device=agent.device)
    )

    torch.testing.assert_close(new_state["deter"][0], expected["deter"][0])
    # env 1 never terminated -- its carry must depend on its OWN history,
    # not match the "as if freshly reset" expectation env 0 matches above.
    assert not torch.allclose(new_state["deter"][1], expected["deter"][1])


def test_contdisc_flag_changes_imagination_disc():
    """``contdisc`` changes the ``disc`` ``DreamerV3._imagination_losses``
    uses to build the imagination λ-return's discount weight (class
    docstring / ``RSSM``'s own docstring): ``contdisc=True`` -> ``disc=1``
    (the horizon discount is already baked into the continue head's own
    training target); ``contdisc=False`` -> ``disc=1-1/discount_horizon``
    (r2dreamer's convention, applied outside the continue head)."""
    agent_off = _agent(contdisc=False)
    agent_on = _agent(contdisc=True)
    for agent in (agent_off, agent_on):
        agent.learn(total_timesteps=60)

    batch_off = agent_off.replay_buffer.sample(agent_off.batch_size)
    _, posterior_off = agent_off._update_model(batch_off)
    imag_off = agent_off._imagination_losses(batch_off, posterior_off)

    batch_on = agent_on.replay_buffer.sample(agent_on.batch_size)
    _, posterior_on = agent_on._update_model(batch_on)
    imag_on = agent_on._imagination_losses(batch_on, posterior_on)

    assert math.isclose(imag_off["disc"], 1.0 - 1.0 / agent_off.discount_horizon)
    assert math.isclose(imag_on["disc"], 1.0)
    assert not math.isclose(imag_off["disc"], imag_on["disc"])
