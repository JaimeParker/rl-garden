from __future__ import annotations

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import CQL, WSRL, CalQL, OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms import __all__ as algorithm_exports
from rl_garden.algorithms.calql import _CalQLRolloutTrainingShell
from rl_garden.algorithms.cql import CQLAlphaLagrange
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer
from rl_garden.common import Logger
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.training.offline.cql import CQLArgs, _cql_kwargs


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def _kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 4,
        "critic_subsample_size": 2,
        "cql_n_actions": 3,
        "cql_alpha": 1.0,
    }


def _offline_kwargs() -> dict[str, object]:
    params = _kwargs()
    params.pop("learning_starts")
    params.pop("training_freq")
    params.pop("eval_freq")
    return params


def _fill(agent, steps: int = 8) -> None:
    # Marks the final step done=True so the run is one complete trajectory --
    # the MC buffer only samples/counts complete trajectories.
    # env.single_observation_space is Dict({"state": Box}) -- OfflineEnvSpec's
    # bare Box is boundary-normalized by BaseAlgorithm.__init__ (see
    # rl_garden.envs.wrappers.VectorizedDictStateWrapper) -- but
    # use_sarsa_reference=True's SarsaMCReplayBuffer stores Dict too, so
    # every branch feeds dict-shaped obs (kept generic via DictArray check
    # for parity with every other test file's _fill helper).
    from rl_garden.buffers.replay_buffer import DictArray

    env = agent.env
    state_shape = env.single_observation_space["state"].shape
    buffer_is_dict = isinstance(agent.replay_buffer.obs, DictArray)
    for step in range(steps):
        state = torch.randn(env.num_envs, *state_shape)
        next_state = torch.randn_like(state)
        obs = {"state": state} if buffer_is_dict else state
        next_obs = {"state": next_state} if buffer_is_dict else next_state
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = (
            torch.ones(env.num_envs) if step == steps - 1 else torch.zeros(env.num_envs)
        )
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _offline_env() -> OfflineEnvSpec:
    env = DummyVecEnv()
    return OfflineEnvSpec(
        env.single_observation_space,
        env.single_action_space,
        num_envs=env.num_envs,
    )


def _dict_offline_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb_cam": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _fill_dict(agent, steps: int = 4) -> None:
    # Marks the final step done=True so the run is one complete trajectory --
    # the MC buffer only samples/counts complete trajectories.
    env = agent.env
    for step in range(steps):
        obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = (
            torch.ones(env.num_envs) if step == steps - 1 else torch.zeros(env.num_envs)
        )
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _dict_arch_kwargs() -> dict[str, object]:
    return {
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 4,
        "critic_subsample_size": 2,
        "cql_n_actions": 3,
        "cql_alpha": 1.0,
        "encoder_config": EncoderConfig(image_fusion_mode="stack_channels", proprio_latent_dim=16),
        "backbone_type": "mlp",
        "std_parameterization": "exp",
        "actor_use_layer_norm": True,
        "critic_use_layer_norm": True,
        "actor_use_group_norm": False,
        "critic_use_group_norm": False,
    }


def test_cql_standalone_train_step_without_calql_bound():
    agent = CQL(env=_offline_env(), **_offline_kwargs())
    _fill(agent)

    info = agent.train(1, compute_info=True)

    assert isinstance(agent, OfflineRLAlgorithm)
    assert agent.use_cql_loss
    assert not agent.use_calql
    assert "cql_loss" in info
    assert "calql_bound_rate" not in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_cql_does_not_own_calql_or_wsrl_flow_state():
    agent = CQL(env=_offline_env(), **_offline_kwargs())

    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCReplayBuffer)
    assert not hasattr(agent, "switch_to_online_mode")
    assert not hasattr(agent, "offline_replay_buffer")
    assert not hasattr(agent, "offline_data_ratio")


def test_cql_accepts_and_wires_eval_env_constructor_args():
    eval_env = DummyVecEnv()
    kwargs = _offline_kwargs()
    agent = CQL(
        env=_offline_env(), eval_env=eval_env, eval_freq=5, num_eval_steps=3, **kwargs
    )

    # eval_env's bare Box space is boundary-normalized to Dict (see _fill's
    # docstring), so agent.eval_env is a VectorizedDictStateWrapper around
    # the original object rather than the object itself.
    assert agent.eval_env.env is eval_env
    assert agent.eval_freq == 5
    assert agent.num_eval_steps == 3


def test_offline_cql_cli_network_args_build_net_arch():
    args = CQLArgs(
        offline_dataset="demo.h5",
        hidden_dim=17,
        actor_hidden_layers=2,
        critic_hidden_layers=4,
    )

    kwargs = _cql_kwargs(args, _offline_env(), Logger(log_type="none"))

    assert kwargs["net_arch"] == {"pi": [17, 17], "qf": [17, 17, 17, 17]}


def test_offline_cql_cli_wires_cql_alpha_param():
    args = CQLArgs(offline_dataset="demo.h5", cql_alpha_param="exp_clip")

    kwargs = _cql_kwargs(args, _offline_env(), Logger(log_type="none"))

    assert kwargs["cql_alpha_param"] == "exp_clip"


def test_cql_diff_clip_mode_always_forces_clamp_when_autotuning():
    kwargs = _offline_kwargs()
    kwargs["cql_autotune_alpha"] = True
    kwargs["cql_clip_diff_min"] = -1e-6
    kwargs["cql_clip_diff_max"] = 1e-6

    agent_unclamped = CQL(
        env=_offline_env(), cql_diff_clip_mode="skip_when_autotune", **kwargs
    )
    _fill(agent_unclamped)
    data = agent_unclamped.replay_buffer.sample(agent_unclamped.batch_size)
    q_pred = agent_unclamped._critic_forward(data.obs, data.actions, target=False)
    _, info_unclamped = agent_unclamped._cql_regularizer(data, q_pred)

    agent_clamped = CQL(env=_offline_env(), cql_diff_clip_mode="always", **kwargs)
    q_pred_clamped = agent_clamped._critic_forward(data.obs, data.actions, target=False)
    _, info_clamped = agent_clamped._cql_regularizer(data, q_pred_clamped)

    # cql_autotune_alpha=True skips the clamp unless cql_diff_clip_mode="always"
    # forces it; with a vanishingly small clip window, the unclamped diff
    # should fall outside it while the clamped one is forced back inside.
    assert abs(info_unclamped["cql_q_diff"].item()) > 1e-6
    assert abs(info_clamped["cql_q_diff"].item()) <= 1e-6 + 1e-9


def test_cql_penalty_scale_lagrange_times_alpha_multiplies_by_cql_alpha_exactly_once():
    kwargs = _offline_kwargs()
    kwargs["cql_autotune_alpha"] = True
    kwargs["use_td_loss"] = False
    kwargs["cql_alpha"] = 3.0

    scratch_agent = CQL(env=_offline_env(), **kwargs)
    _fill(scratch_agent)
    data = scratch_agent.replay_buffer.sample(scratch_agent.batch_size)

    agent_off = CQL(env=_offline_env(), cql_penalty_scale="lagrange_only", **kwargs)
    loss_off, _ = agent_off._critic_loss(data)

    agent_on = CQL(
        env=_offline_env(), cql_penalty_scale="lagrange_times_alpha", **kwargs
    )
    loss_on, _ = agent_on._critic_loss(data)

    assert torch.allclose(loss_on, loss_off * agent_off.cql_alpha, rtol=1e-4, atol=1e-6)


def test_cql_alpha_param_exp_clip_uses_exp_clip_parameterization():
    kwargs = _offline_kwargs()
    kwargs["cql_autotune_alpha"] = True
    kwargs["cql_alpha_lagrange_init"] = 2.0

    agent_exp_clip = CQL(env=_offline_env(), cql_alpha_param="exp_clip", **kwargs)
    agent_default = CQL(env=_offline_env(), **kwargs)

    assert agent_exp_clip.cql_alpha_lagrange.param_type == "exp_clip"
    assert agent_default.cql_alpha_lagrange.param_type == "softplus"
    assert torch.isclose(
        agent_exp_clip.cql_alpha_lagrange(), torch.tensor(2.0), atol=1e-4
    )
    assert torch.isclose(
        agent_default.cql_alpha_lagrange(), torch.tensor(2.0), atol=1e-4
    )


def test_off2on_calql_and_wsrl_thread_all_three_cql_parity_axes():
    from unittest.mock import MagicMock

    from rl_garden.algorithms.off2on_calql import Off2OnCalQL

    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    flags = {
        "cql_diff_clip_mode": "always",
        "cql_penalty_scale": "lagrange_times_alpha",
        "cql_alpha_param": "exp_clip",
    }
    common = {
        "buffer_size": 100,
        "buffer_device": "cpu",
        "learning_starts": 10,
        "batch_size": 8,
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 4,
        "critic_subsample_size": 2,
        "cql_autotune_alpha": True,
        "device": "cpu",
        "seed": 42,
    }

    wsrl_agent = WSRL(env=env, **common, **flags)
    off2on_calql_agent = Off2OnCalQL(env=env, **common, **flags)

    for agent in (wsrl_agent, off2on_calql_agent):
        assert agent.cql_diff_clip_mode == "always"
        assert agent.cql_penalty_scale == "lagrange_times_alpha"
        assert agent.cql_alpha_param == "exp_clip"
        assert agent.cql_alpha_lagrange.param_type == "exp_clip"


class TestCQLAlphaLagrange:
    """Unit tests for the CQL alpha Lagrange multiplier (rl_garden.algorithms.cql)."""

    def test_lagrange_forward(self):
        lagrange = CQLAlphaLagrange(init_value=5.0)
        alpha = lagrange()
        assert alpha.shape == ()
        assert alpha.item() > 0

    def test_lagrange_gradient(self):
        lagrange = CQLAlphaLagrange(init_value=5.0)
        alpha = lagrange()
        loss = alpha * 2.0
        loss.backward()

        assert lagrange.log_alpha.grad is not None

    @pytest.mark.parametrize("param_type", ["softplus", "exp_clip"])
    @pytest.mark.parametrize("init_value", [0.5, 1.0, 5.0])
    def test_lagrange_init_parity_across_parameterizations(
        self, param_type, init_value
    ):
        lagrange = CQLAlphaLagrange(init_value=init_value, param_type=param_type)
        assert abs(lagrange().item() - init_value) < 1e-4

    def test_lagrange_exp_clip_upper_bound(self):
        lagrange = CQLAlphaLagrange(
            init_value=1.0, param_type="exp_clip", exp_clip_max=1e6
        )
        with torch.no_grad():
            lagrange.log_alpha.fill_(20.0)
        assert lagrange().item() == pytest.approx(1e6)

    def test_lagrange_invalid_param_type_raises(self):
        with pytest.raises(ValueError, match="Unknown param_type"):
            CQLAlphaLagrange(init_value=1.0, param_type="bogus")


def test_cql_train_step_and_checkpoint(tmp_path):
    agent = CQL(env=_offline_env(), checkpoint_dir=str(tmp_path), **_offline_kwargs())
    _fill(agent)

    info = agent.train(1, compute_info=True)
    result = agent.learn_offline(2, save_filename="offline_cql.pt")

    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCReplayBuffer)
    assert "cql_loss" in info
    assert "calql_bound_rate" not in info
    assert result.final_checkpoint == tmp_path / "offline_cql.pt"
    assert (tmp_path / "offline_cql.pt").exists()


def test_calql_standalone_train_step_logs_bound_rate():
    agent = CalQL(env=_offline_env(), **_offline_kwargs())
    _fill(agent)

    info = agent.train(1, compute_info=True)

    assert isinstance(agent, CQL)
    assert agent.use_calql
    assert "cql_loss" in info
    assert "calql_bound_rate" in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_calql_owns_mc_replay_without_wsrl_flow_state():
    agent = CalQL(env=_offline_env(), **_offline_kwargs())

    assert isinstance(agent.replay_buffer, MCReplayBuffer)
    assert not hasattr(agent, "switch_to_online_mode")
    assert not hasattr(agent, "offline_replay_buffer")
    assert not hasattr(agent, "offline_data_ratio")


def test_calql_train_step_logs_bound_rate():
    agent = CalQL(
        env=_offline_env(),
        sparse_reward_mc=True,
        sparse_negative_reward=-1.0,
        success_threshold=0.5,
        **_offline_kwargs(),
    )
    _fill(agent)

    info = agent.train(1, compute_info=True)

    assert isinstance(agent.replay_buffer, MCReplayBuffer)
    assert agent.replay_buffer.sparse_reward_mc
    assert agent.replay_buffer.sparse_negative_reward == -1.0
    assert "cql_loss" in info
    assert "calql_bound_rate" in info


def test_calql_default_does_not_build_sarsa_reference_network():
    agent = CalQL(env=_offline_env(), **_offline_kwargs())

    assert agent.use_sarsa_reference is False
    assert agent.sarsa_q_net is None
    assert agent.sarsa_q_target is None
    assert agent.sarsa_q_optimizer is None
    assert agent._extra_batch_slice_keys == ()
    assert isinstance(agent.replay_buffer, MCReplayBuffer)


def test_calql_sarsa_reference_builds_network_and_uses_sarsa_buffer():
    from rl_garden.buffers.sarsa_buffer import SarsaMCReplayBuffer

    agent = CalQL(
        env=_offline_env(),
        use_sarsa_reference=True,
        sarsa_hidden_dims=(16,),
        **_offline_kwargs(),
    )

    assert agent.sarsa_q_net is not None
    assert agent.sarsa_q_target is not None
    assert agent.sarsa_q_optimizer is not None
    assert agent._extra_batch_slice_keys == ("next_actions", "next_action_valid")
    assert isinstance(agent.replay_buffer, SarsaMCReplayBuffer)


def test_calql_sarsa_reference_dict_obs_raises():
    with pytest.raises(ValueError, match="does not support image observations"):
        CalQL(
            env=_dict_offline_env(),
            use_sarsa_reference=True,
            **_offline_kwargs(),
        )


def test_calql_sarsa_reference_train_step_updates_sarsa_net_and_logs_loss():
    agent = CalQL(
        env=_offline_env(),
        use_sarsa_reference=True,
        sarsa_hidden_dims=(16,),
        **_offline_kwargs(),
    )
    _fill(agent)
    before = [p.clone() for p in agent.sarsa_q_net.parameters()]

    info = agent.train(1, compute_info=True)

    after = list(agent.sarsa_q_net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert "sarsa_loss" in info
    assert torch.isfinite(torch.tensor(info["sarsa_loss"]))
    assert "calql_bound_rate" in info


def test_calql_sarsa_reference_regularizer_uses_sarsa_net_not_mc_returns():
    agent = CalQL(
        env=_offline_env(),
        use_sarsa_reference=True,
        sarsa_hidden_dims=(16,),
        **_offline_kwargs(),
    )
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)
    q_pred = agent._critic_forward(data.obs, data.actions, target=False)

    # Corrupt mc_returns; the SARSA path must ignore it entirely.
    import dataclasses

    corrupted = dataclasses.replace(
        data, mc_returns=torch.full_like(data.mc_returns, 1e6)
    )
    torch.manual_seed(0)
    loss_normal, _ = agent._cql_regularizer(data, q_pred)
    torch.manual_seed(0)
    loss_corrupted, _ = agent._cql_regularizer(corrupted, q_pred)

    assert torch.allclose(loss_normal, loss_corrupted)


def test_calql_sarsa_reference_checkpoint_roundtrip(tmp_path):
    """Regression for the ``_SarsaReferenceQ`` -> ``ScalarQNetwork``
    promotion (rl_garden/networks/value.py): the extraction only moved the
    class, so ``sarsa_q_net``'s state-dict keys (all under its ``net.``
    submodule) must be unchanged and a checkpoint saved by one agent must
    still load cleanly into a fresh one with matching parameters."""
    agent = CalQL(
        env=_offline_env(),
        use_sarsa_reference=True,
        sarsa_hidden_dims=(16,),
        checkpoint_dir=str(tmp_path),
        **_offline_kwargs(),
    )
    _fill(agent)
    agent.train(1, compute_info=False)

    sd = agent.sarsa_q_net.state_dict()
    assert sd and all(key.startswith("net.") for key in sd)

    ckpt = agent.save(tmp_path / "calql_sarsa.pt")
    fresh = CalQL(
        env=_offline_env(),
        use_sarsa_reference=True,
        sarsa_hidden_dims=(16,),
        **_offline_kwargs(),
    )
    fresh.load(ckpt, load_replay_buffer=False)

    for key, value in sd.items():
        assert torch.equal(fresh.sarsa_q_net.state_dict()[key], value)


def test_cql_dict_obs_train_step_and_checkpoint(tmp_path):
    agent = CQL(
        env=_dict_offline_env(),
        checkpoint_dir=str(tmp_path),
        **_offline_kwargs(),
    )
    _fill_dict(agent)

    info = agent.train(1, compute_info=True)
    result = agent.learn_offline(2, save_filename="offline_cql_dict.pt")

    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert isinstance(agent.policy.actor_extractor, CombinedExtractor)
    assert "cql_loss" in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))
    assert result.final_checkpoint == tmp_path / "offline_cql_dict.pt"
    assert (tmp_path / "offline_cql_dict.pt").exists()


def _asymmetric_offline_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb_cam": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                "state_object_pose": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _fill_asymmetric(agent, steps: int = 4) -> None:
    env = agent.env
    for _ in range(steps):
        obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
            "state_object_pose": torch.randn(env.num_envs, 3),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
            "state_object_pose": torch.randn(env.num_envs, 3),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_cql_asymmetric_obs_groups_critic_sees_extra_state_actor_does_not():
    """End-to-end asymmetric actor/critic encoders via state_<name>: critic
    sees state_object_pose, actor does not, encoder_sharing="separate".
    Asserts the actor extractor's schema lacks the key and, over one real
    train() step, gradients reach only the right extractor."""
    from rl_garden.observations import ObsGroups

    kwargs = _offline_kwargs()
    agent = CQL(
        env=_asymmetric_offline_env(),
        encoder_config=EncoderConfig(proprio_latent_dim=4),
        obs_groups=ObsGroups(
            actor=("rgb_cam", "state"), critic=("rgb_cam", "state", "state_object_pose")
        ),
        encoder_sharing="separate",
        **kwargs,
    )

    actor_extractor = agent.policy.actor_extractor
    critic_extractor = agent.policy.critic_extractor
    assert critic_extractor is not None and critic_extractor is not actor_extractor
    assert "state_object_pose" not in actor_extractor.state_keys
    assert "state_object_pose" in critic_extractor.state_keys

    _fill_asymmetric(agent)
    data = agent.replay_buffer.sample(4)

    actor_loss, _ = agent._actor_loss(data.obs)
    actor_grad_on_actor = torch.autograd.grad(
        actor_loss, list(actor_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in actor_grad_on_actor)
    # The actor loss's Q(s, pi(s)) term re-extracts critic-role features with
    # stop_gradient=True (SACPolicy.critic_features_for); CombinedExtractor's
    # own stop_gradient convention detaches only the image branch, so this
    # checks image-branch isolation specifically, not the whole
    # critic_extractor (its proprio branch legitimately still requires_grad
    # here -- unrelated to obs_groups).
    actor_grad_on_critic_image = torch.autograd.grad(
        actor_loss, list(critic_extractor.image_encoder.parameters()), allow_unused=True
    )
    assert all(g is None for g in actor_grad_on_critic_image)

    critic_loss, _ = agent._critic_loss(data)
    critic_grad_on_critic = torch.autograd.grad(
        critic_loss, list(critic_extractor.parameters()), allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in critic_grad_on_critic)
    # Unlike SAC, CQL's critic loss is not actor-independent: the CQL
    # regularizer samples OOD actions from the current policy
    # (_sample_n_actions_with_log_probs -> policy.extract_features, an
    # undetached actor_extractor forward), so critic_loss legitimately has a
    # live gradient path into actor_extractor too -- this is not a leak to
    # guard against, just CQL's own regularizer design.

    # A real end-to-end update step also runs cleanly.
    info = agent.train(gradient_steps=1, compute_info=True)
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_calql_dict_obs_uses_mc_dict_replay_buffer():
    agent = CalQL(
        env=_dict_offline_env(),
        **_offline_kwargs(),
    )

    assert isinstance(agent.replay_buffer, MCReplayBuffer)


def test_calql_dict_obs_checkpoint_loads_into_wsrl(tmp_path):
    arch_kwargs = _dict_arch_kwargs()

    calql_agent = CalQL(
        env=_dict_offline_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        checkpoint_dir=str(tmp_path),
        **arch_kwargs,
    )
    _fill_dict(calql_agent)
    calql_agent.train(1)
    result = calql_agent.learn_offline(1, save_filename="calql_dict.pt")
    checkpoint_path = result.final_checkpoint
    assert checkpoint_path is not None and checkpoint_path.exists()

    wsrl_agent = WSRL(
        env=_dict_offline_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        **arch_kwargs,
    )
    wsrl_agent.load(checkpoint_path)
    _fill_dict(wsrl_agent)

    info = wsrl_agent.train(1, compute_info=True)

    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_wsrl_uses_private_rollout_shell_not_public_offline_calql():
    assert issubclass(WSRL, _CalQLRolloutTrainingShell)
    assert not issubclass(WSRL, CalQL)


def test_offline_cql_names_are_not_public_exports():
    assert "OfflineCQL" not in algorithm_exports
    assert "OfflineCalQL" not in algorithm_exports


def _write_demo_h5(path):
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.zeros((7, 4), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((6, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones((6,), dtype=np.float32))
        dones = np.zeros((6,), dtype=np.float32)
        dones[-1] = 1.0
        group.create_dataset("dones", data=dones)


def test_pretrain_offline_cli_algorithm_selection(tmp_path):
    dataset = tmp_path / "demo.h5"
    _write_demo_h5(dataset)

    for algorithm in ("bc", "cql", "calql", "wsrl", "iql"):
        checkpoint_dir = tmp_path / algorithm
        cmd = [
            sys.executable,
            "examples/pretrain_offline.py",
            algorithm,
            "--offline_dataset",
            str(dataset),
            "--num_offline_steps",
            "2",
            "--buffer_device",
            "cpu",
            "--log_type",
            "none",
            "--no-std-log",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--log_dir",
            str(tmp_path / "logs"),
            "--exp_name",
            algorithm,
            "--batch_size",
            "4",
            "--buffer_size",
            "32",
        ]
        if algorithm in {"cql", "calql", "wsrl"}:
            cmd.extend(
                [
                    "--n_critics",
                    "4",
                    "--critic_subsample_size",
                    "2",
                    "--cql_n_actions",
                    "2",
                    "--no-use-compile",
                ]
            )
        elif algorithm == "iql":
            cmd.extend(
                [
                    "--device",
                    "cpu",
                    "--n_critics",
                    "4",
                    "--critic_subsample_size",
                    "2",
                ]
            )
        else:
            cmd.extend(["--device", "cpu"])
        subprocess.run(cmd, check=True)
        expected = f"{algorithm}_offline_pretrained.pt"
        assert (checkpoint_dir / expected).exists()
        config = json.loads((tmp_path / "logs" / algorithm / "config.json").read_text())
        assert config["schema_version"] == 3
        assert config["status"] == "materialized"
        assert config["runtime"]["dry_run"] is False
        assert config["selection"] == {
            "training_phase": "offline",
            "algorithm": algorithm,
        }


def test_pretrain_offline_cli_rejects_legacy_algorithm_flag():
    cmd = [
        sys.executable,
        "examples/pretrain_offline.py",
        "--algorithm",
        "cql",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode != 0


def test_pretrain_cql_offline_cli_requires_dataset():
    cmd = [
        sys.executable,
        "examples/pretrain_offline.py",
        "cql",
        "--num_offline_steps",
        "1",
        "--log_type",
        "none",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--offline_dataset is required" in result.stderr
