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
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.mc_buffer import MCDictReplayBuffer, MCTensorReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common import Logger
from rl_garden.encoders.combined import CombinedExtractor
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
    env = agent.env
    for step in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
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
                "rgb": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
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
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
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
        "image_keys": ("rgb",),
        "image_fusion_mode": "stack_channels",
        "proprio_latent_dim": 16,
        "use_proprio": True,
        "enable_stacking": False,
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

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert not hasattr(agent, "switch_to_online_mode")
    assert not hasattr(agent, "offline_replay_buffer")
    assert not hasattr(agent, "offline_data_ratio")


def test_cql_accepts_and_wires_eval_env_constructor_args():
    eval_env = DummyVecEnv()
    kwargs = _offline_kwargs()
    agent = CQL(
        env=_offline_env(), eval_env=eval_env, eval_freq=5, num_eval_steps=3, **kwargs
    )

    assert agent.eval_env is eval_env
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
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
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

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCTensorReplayBuffer)
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

    assert isinstance(agent.replay_buffer, MCTensorReplayBuffer)
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

    assert isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert agent.replay_buffer.sparse_reward_mc
    assert agent.replay_buffer.sparse_negative_reward == -1.0
    assert "cql_loss" in info
    assert "calql_bound_rate" in info


def test_cql_dict_obs_train_step_and_checkpoint(tmp_path):
    agent = CQL(
        env=_dict_offline_env(),
        image_keys=("rgb",),
        checkpoint_dir=str(tmp_path),
        **_offline_kwargs(),
    )
    _fill_dict(agent)

    info = agent.train(1, compute_info=True)
    result = agent.learn_offline(2, save_filename="offline_cql_dict.pt")

    assert isinstance(agent.replay_buffer, DictReplayBuffer)
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)
    assert "cql_loss" in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))
    assert result.final_checkpoint == tmp_path / "offline_cql_dict.pt"
    assert (tmp_path / "offline_cql_dict.pt").exists()


def test_calql_dict_obs_uses_mc_dict_replay_buffer():
    agent = CalQL(
        env=_dict_offline_env(),
        image_keys=("rgb",),
        **_offline_kwargs(),
    )

    assert isinstance(agent.replay_buffer, MCDictReplayBuffer)


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
