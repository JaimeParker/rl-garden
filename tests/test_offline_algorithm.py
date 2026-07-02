"""Tests for offline-only algorithm scaffolding."""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms import (
    OfflineEnvSpec,
    OfflineRLAlgorithm,
    infer_box_specs_from_h5,
    infer_specs_from_h5,
    run_offline_pretraining,
)
from rl_garden.policies.base import BasePolicy


class DummyPolicy(BasePolicy):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(obs_dim, action_dim)

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic
        return self.net(obs)


class DummyOfflineAlgorithm(OfflineRLAlgorithm):
    def __init__(self, env: OfflineEnvSpec, **kwargs) -> None:
        super().__init__(env, **kwargs)
        obs_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.policy = DummyPolicy(obs_dim, action_dim).to(self.device)

    def _setup_model(self) -> None:
        pass

    def train(
        self, gradient_steps: int, compute_info: bool = False
    ) -> dict[str, float]:
        self._global_update += gradient_steps
        if not compute_info:
            return {}
        return {"dummy_loss": float(self._global_update)}


def test_infer_box_specs_from_h5(tmp_path):
    path = tmp_path / "demo.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.zeros((4, 7), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))

    obs_space, action_space = infer_box_specs_from_h5(
        path, action_low=-0.5, action_high=0.5
    )

    assert obs_space.shape == (7,)
    assert action_space.shape == (2,)
    assert np.all(action_space.low == -0.5)
    assert np.all(action_space.high == 0.5)


def test_infer_box_specs_rejects_dict_obs(tmp_path):
    path = tmp_path / "demo_dict.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.zeros((4, 7), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))

    try:
        infer_box_specs_from_h5(path)
    except NotImplementedError as exc:
        assert "Dict observations" in str(exc)
    else:  # pragma: no cover - makes assertion message clearer.
        raise AssertionError("Expected dict observations to be rejected")


def test_infer_specs_from_h5_supports_dict_obs(tmp_path):
    path = tmp_path / "demo_dict.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = group.create_group("obs")
        obs.create_dataset("rgb", data=np.zeros((4, 64, 64, 3), dtype=np.uint8))
        obs.create_dataset("state", data=np.zeros((4, 7), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))

    obs_space, action_space = infer_specs_from_h5(
        path, action_low=-0.25, action_high=0.25
    )

    assert isinstance(obs_space, spaces.Dict)
    assert obs_space["rgb"].shape == (64, 64, 3)
    assert obs_space["rgb"].dtype == np.uint8
    assert obs_space["state"].shape == (7,)
    assert action_space.shape == (2,)
    assert np.all(action_space.low == -0.25)


def test_run_offline_pretraining_tracks_steps_and_saves(tmp_path):
    env = OfflineEnvSpec(
        observation_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    agent = DummyOfflineAlgorithm(env, device="cpu", logger=None, std_log=False)

    result = run_offline_pretraining(
        agent,
        num_steps=3,
        gradient_steps=2,
        checkpoint_dir=tmp_path,
        checkpoint_freq=2,
        save_filename="final_offline.pt",
        std_log=False,
    )

    assert result.final_step == 3
    assert agent._global_step == 3
    assert agent._global_update == 6
    assert result.last_metrics == {"dummy_loss": 6.0}
    assert result.final_checkpoint == tmp_path / "final_offline.pt"
    assert (tmp_path / "checkpoint_2.pt").exists()
    assert (tmp_path / "final_offline.pt").exists()


def test_offline_algorithm_learn_uses_offline_steps(tmp_path):
    env = OfflineEnvSpec(
        observation_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    agent = DummyOfflineAlgorithm(
        env,
        device="cpu",
        logger=None,
        std_log=False,
        checkpoint_dir=str(tmp_path),
        checkpoint_freq=0,
    )

    returned = agent.learn(2)

    assert returned is agent
    assert agent._global_step == 2
    assert agent._global_update == 2
    assert (tmp_path / "offline_pretrained.pt").exists()


def test_run_offline_pretraining_can_skip_final_checkpoint(tmp_path):
    env = OfflineEnvSpec(
        observation_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    agent = DummyOfflineAlgorithm(env, device="cpu", logger=None, std_log=False)

    result = run_offline_pretraining(
        agent,
        num_steps=2,
        checkpoint_dir=tmp_path,
        checkpoint_freq=1,
        save_final_checkpoint=False,
        std_log=False,
    )

    assert result.final_checkpoint is None
    assert (tmp_path / "checkpoint_1.pt").exists()
    assert (tmp_path / "checkpoint_2.pt").exists()
    assert not (tmp_path / "offline_pretrained.pt").exists()


def test_offline_registry_builders_create_wsrl_and_iql():
    from rl_garden.algorithms import IQL, OfflineEnvSpec, WSRL
    from rl_garden.training.offline.iql import IQLArgs, build_iql
    from rl_garden.training.offline.wsrl import WSRLOfflineArgs, build_wsrl

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    env_spec = OfflineEnvSpec(obs_space, action_space, num_envs=1)

    base_kwargs = dict(
        offline_dataset_path="/tmp/unused.h5",
        num_offline_steps=1,
        buffer_size=64,
        buffer_device="cpu",
        device="cpu",
        batch_size=4,
        net_arch=[16],
        n_critics=4,
        critic_subsample_size=2,
        cql_n_actions=4,
        cql_alpha=1.0,
        log_type="none",
        use_compile=False,
    )

    class _NoopLogger:
        def add_scalar(self, *a, **k):
            pass

        def add_summary(self, *a, **k):
            pass

        def add_text(self, *a, **k):
            pass

        def close(self):
            pass

    args_wsrl = WSRLOfflineArgs(training_freq=1, **base_kwargs)
    agent_wsrl = build_wsrl(args_wsrl, env_spec, _NoopLogger())
    assert isinstance(agent_wsrl, WSRL)
    assert agent_wsrl.device.type == "cpu"
    assert agent_wsrl.net_arch == [16]

    iql_kwargs = {
        key: value
        for key, value in base_kwargs.items()
        if key not in {"cql_n_actions", "cql_alpha", "use_compile"}
    }
    args_iql = IQLArgs(**iql_kwargs)
    agent_iql = build_iql(args_iql, env_spec, _NoopLogger())
    assert isinstance(agent_iql, IQL)
    assert agent_iql.device.type == "cpu"
    assert agent_iql.net_arch == [16]


def test_eval_env_config_carries_dict_obs_vision_fields():
    """The optional --env_id eval env must mirror dict-obs/vision CLI args.

    Without this, pretrain_offline.py's periodic zero-shot eval env defaults
    to a flat ``obs_mode="state"`` ManiSkill env, which crashes BC/IQL
    policies built for per-camera RGB+state Dict observation spaces.
    """
    from rl_garden.training.offline._runner import _eval_env_request
    from rl_garden.training.offline.bc import BCArgs

    args = BCArgs(
        offline_dataset_path="/tmp/unused.h5",
        env_id="StackCube-v1",
        num_eval_envs=16,
        obs_mode="rgb",
        include_state=True,
        per_camera_rgbd=True,
        camera_width=64,
        camera_height=64,
        reward_scale=2.0,
        reward_bias=-1.0,
    )

    req = _eval_env_request(args)

    assert req.env_id == "StackCube-v1"
    assert req.num_eval_envs == 16
    assert req.obs_mode == "rgb"
    assert req.include_state is True
    assert req.per_camera_rgbd is True
    assert req.camera_width == 64
    assert req.camera_height == 64
    assert req.reward_scale == 2.0
    assert req.reward_bias == -1.0
