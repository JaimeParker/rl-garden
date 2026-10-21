"""Tests for the generic ``gymnasium`` env backend.

User-authored envs come from ``tests/_gymnasium_user_envs.py`` (imported via
``import_modules``, like a user's package); the state-only fixture is
gymnasium's own ``Pendulum-v1``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from _cli import run_entry

from rl_garden.common.env_args import GymnasiumConfig
from rl_garden.envs.backend_registry import EnvRequest, make_training_envs
from rl_garden.envs.backends.gymnasium import GymnasiumBackend
from rl_garden.envs.gymnasium import GymnasiumEnvConfig, make_gymnasium_env
from rl_garden.envs.gymnasium.env import _make_env_fn
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
USER_ENVS = ("_gymnasium_user_envs",)


def _make_req(**overrides):
    defaults = {
        "env_id": "Pendulum-v1",
        "num_envs": 4,
        "control_mode": "n/a",
        "render_mode": "rgb_array",
        "seed": 1,
        "observation": ObservationConfig(),
        "num_eval_envs": 2,
        "backend_config": GymnasiumConfig(),
    }
    defaults.update(overrides)
    return EnvRequest(**defaults)


def _pixels_cfg(**overrides) -> GymnasiumEnvConfig:
    defaults = {
        "env_id": "RlgTestPixels-v0",
        "num_envs": 2,
        "seed": 0,
        "import_modules": USER_ENVS,
    }
    defaults.update(overrides)
    return GymnasiumEnvConfig(**defaults)


def _run_fresh_interpreter(code: str) -> None:
    pythonpath = os.pathsep.join(
        p for p in (str(TESTS_DIR), str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        env=os.environ | {"PYTHONPATH": pythonpath, "MPLCONFIGDIR": "/tmp"},
    )


def test_backend_registers_with_api_version_2():
    from rl_garden.envs.backend_registry import _REGISTRY, discover_env_backends

    discover_env_backends()
    assert _REGISTRY["gymnasium"] is GymnasiumBackend
    assert GymnasiumBackend.api_version == 2
    assert GymnasiumBackend.config_field == "gymnasium"


def test_resolve_config_splits_train_eval_num_envs_and_parses_kwargs():
    backend_config = GymnasiumConfig(
        import_modules=("my_pkg.envs",),
        env_kwargs_json='{"g": 9.81}',
        vectorization="async",
    )
    req = _make_req(backend_config=backend_config, reward_scale=2.0)

    train_cfg = GymnasiumBackend.resolve_config(req, is_eval=False)
    eval_cfg = GymnasiumBackend.resolve_config(req, is_eval=True)

    assert train_cfg.num_envs == 4
    assert eval_cfg.num_envs == 2
    assert train_cfg.env_kwargs == {"g": 9.81}
    assert train_cfg.import_modules == ("my_pkg.envs",)
    assert train_cfg.vectorization == "async"
    assert train_cfg.device == "cpu"
    assert train_cfg.reward_scale == 2.0


def test_resolve_config_does_not_import_user_modules():
    _run_fresh_interpreter(
        """
import sys
import gymnasium
from rl_garden.common.env_args import GymnasiumConfig
from rl_garden.envs.backend_registry import _REGISTRY, EnvRequest, discover_env_backends
from rl_garden.observations.config import ObservationConfig

discover_env_backends()
req = EnvRequest(
    env_id="RlgTestPixels-v0", num_envs=2, control_mode="n/a", render_mode="rgb_array",
    seed=0, observation=ObservationConfig(rgb=("front",)),
    backend_config=GymnasiumConfig(import_modules=("_gymnasium_user_envs",)),
)
cfg = _REGISTRY["gymnasium"].resolve_config(req, is_eval=False)
assert cfg.import_modules == ("_gymnasium_user_envs",)
assert "_gymnasium_user_envs" not in sys.modules
assert "RlgTestPixels-v0" not in gymnasium.registry
"""
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"backend_config": GymnasiumConfig(vectorization="fork")},
            ValueError,
            "vectorization must be 'sync' or 'async', got 'fork'",
        ),
        (
            {"backend_config": GymnasiumConfig(env_kwargs_json="[1]")},
            ValueError,
            "env_kwargs_json must encode a JSON object, got list",
        ),
        (
            {"backend_config": GymnasiumConfig(import_modules=("my-pkg.envs",))},
            ValueError,
            "import_modules entry 'my-pkg.envs' is not a dotted Python module path",
        ),
        (
            {"observation": ObservationConfig(rgb=("front",), image_size=(64, 64))},
            ObservationContractError,
            "cannot resize images",
        ),
        (
            {"observation": ObservationConfig(frame_stack=2)},
            ObservationContractError,
            "frame_stack=2 requires an rgb/depth observation",
        ),
    ],
)
def test_resolve_config_rejects_invalid_settings(overrides, error, message):
    with pytest.raises(error, match=message):
        GymnasiumBackend.resolve_config(_make_req(**overrides), is_eval=False)


def test_bare_box_observation_is_wrapped_as_state():
    env = make_gymnasium_env(GymnasiumEnvConfig(env_id="Pendulum-v1", num_envs=3, seed=0))

    assert set(env.single_observation_space.spaces) == {"state"}
    obs, _ = env.reset()
    assert obs["state"].shape == (3, 3)
    assert obs["state"].dtype == torch.float32
    assert obs["state"].device.type == "cpu"
    env.close()


def test_contract_dict_observation_passes_registry_validation():
    req = _make_req(
        env_id="RlgTestPixels-v0",
        num_envs=2,
        observation=ObservationConfig(rgb=("front",)),
        backend_config=GymnasiumConfig(import_modules=USER_ENVS),
    )

    train_env, eval_env = make_training_envs("gymnasium", req)

    obs, _ = train_env.reset()
    assert obs["rgb_front"].shape == (2, 16, 16, 3)
    assert obs["rgb_front"].dtype == torch.uint8
    assert obs["state"].shape == (2, 2)
    assert eval_env.num_envs == 2
    train_env.close()
    eval_env.close()


def test_observation_key_mismatch_raises_contract_error():
    req = _make_req(
        env_id="RlgTestPixels-v0",
        num_envs=1,
        observation=ObservationConfig(rgb=("side",)),
        backend_config=GymnasiumConfig(import_modules=USER_ENVS),
        create_eval_env=False,
    )

    with pytest.raises(ObservationContractError):
        make_training_envs("gymnasium", req)


def test_frame_stack_stacks_image_keys():
    env = make_gymnasium_env(_pixels_cfg(image_keys=("rgb_front",), frame_stack=3))

    obs, _ = env.reset()
    assert obs["rgb_front"].shape == (2, 3, 16, 16, 3)
    assert obs["state"].shape == (2, 2)
    env.close()


def test_episode_end_exposes_final_info_and_final_observation():
    env = make_gymnasium_env(_pixels_cfg())
    env.reset()
    actions = torch.zeros((2, 1))

    for _ in range(3):
        _, _, terminated, truncated, infos = env.step(actions)

    assert bool(truncated.all()) and not bool(terminated.any())
    assert bool(infos["_final_observation"].all())
    assert bool(infos["_final_info"].all())
    episode = infos["final_info"]["episode"]
    assert torch.equal(episode["r"].float(), torch.full((2,), 3.0))
    assert torch.equal(episode["l"].long(), torch.full((2,), 3))
    env.close()


def test_pre_wrapped_record_episode_statistics_is_not_doubled():
    from gymnasium.wrappers import RecordEpisodeStatistics

    single = _make_env_fn("RlgTestPreWrapped-v0", {}, USER_ENVS)()
    count = 0
    while isinstance(single, gym.Wrapper):
        count += isinstance(single, RecordEpisodeStatistics)
        single = single.env
    assert count == 1

    env = make_gymnasium_env(_pixels_cfg(env_id="RlgTestPreWrapped-v0"))
    env.reset()
    for _ in range(3):
        _, _, _, truncated, infos = env.step(torch.zeros((2, 1)))
    assert bool(truncated.all())
    assert torch.equal(infos["final_info"]["episode"]["r"].float(), torch.full((2,), 3.0))
    env.close()


def test_reset_without_seed_replays_configured_seed():
    env = make_gymnasium_env(GymnasiumEnvConfig(env_id="Pendulum-v1", num_envs=2, seed=7))

    first, _ = env.reset()
    env.step(torch.ones((2, 1)))
    second, _ = env.reset()

    assert torch.equal(first["state"], second["state"])
    env.close()


def test_reward_scale_and_bias_are_applied():
    env = make_gymnasium_env(_pixels_cfg(reward_scale=2.0, reward_bias=0.5))
    env.reset()

    _, rewards, _, _, _ = env.step(torch.zeros((2, 1)))

    assert torch.allclose(rewards.float(), torch.full((2,), 2.5))
    env.close()


def test_unregistered_env_id_error_explains_registration():
    with pytest.raises(gym.error.Error, match="is not registered in this process"):
        make_gymnasium_env(GymnasiumEnvConfig(env_id="RlgTestMissing-v0", num_envs=1, seed=0))


def test_missing_import_module_error_names_cli_field():
    cfg = GymnasiumEnvConfig(
        env_id="Pendulum-v1", num_envs=1, seed=0, import_modules=("rlg_test_no_such_module",)
    )

    with pytest.raises(ModuleNotFoundError, match="--gymnasium.import_modules"):
        make_gymnasium_env(cfg)


def test_env_thunk_imports_user_modules_in_fresh_interpreter():
    _run_fresh_interpreter(
        """
import sys
from rl_garden.envs.gymnasium.env import _make_env_fn

thunk = _make_env_fn("RlgTestPixels-v0", {}, ("_gymnasium_user_envs",))
assert "_gymnasium_user_envs" not in sys.modules
env = thunk()
assert "_gymnasium_user_envs" in sys.modules
assert set(env.observation_space.spaces) == {"rgb_front", "state"}
"""
    )


@pytest.mark.slow
def test_async_spawn_workers_import_user_modules():
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode

    from rl_garden.envs.vector_env import TorchVectorEnvAdapter

    env_fns = [_make_env_fn("RlgTestPixels-v0", {}, USER_ENVS) for _ in range(2)]
    vec_env = AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP, context="spawn")
    env = TorchVectorEnvAdapter(vec_env, device="cpu")
    try:
        env.reset(seed=0)
        for _ in range(3):
            _, _, _, truncated, infos = env.step(torch.zeros((2, 1)))
        assert bool(truncated.all())
        assert torch.equal(infos["final_info"]["episode"]["r"].float(), torch.full((2,), 3.0))
    finally:
        env.close()


def test_print_config_accepts_cli_and_yaml_and_rejects_bad_overrides(tmp_path):
    common = ["--print-config", "--log-type", "none", "--log-dir", str(tmp_path)]

    cli = run_entry(
        "train_online",
        [
            "sac",
            "--env-backend",
            "gymnasium",
            "--env-id",
            "MyReach-v0",
            "--gymnasium.import-modules",
            "rlg_not_installed.envs",
            "--gymnasium.vectorization",
            "async",
            *common,
        ],
    )
    assert cli.returncode == 0, cli.stderr
    environment = json.loads(cli.stdout)["active_environment"]
    assert environment["backend"] == "gymnasium"
    assert environment["config"]["import_modules"] == ["rlg_not_installed.envs"]
    assert environment["config"]["vectorization"] == "async"
    assert "rlg_not_installed" not in sys.modules

    preset = tmp_path / "gymnasium.yaml"
    preset.write_text(
        "env_backend: gymnasium\n"
        "env_id: MyReach-v0\n"
        "gymnasium:\n"
        "  import_modules: [rlg_not_installed.envs]\n"
        "  vectorization: async\n",
        encoding="utf-8",
    )
    yaml = run_entry("train_online", ["sac", "--config", str(preset), *common])
    assert yaml.returncode == 0, yaml.stderr
    environment = json.loads(yaml.stdout)["active_environment"]
    assert environment["config"]["import_modules"] == ["rlg_not_installed.envs"]
    assert environment["config"]["vectorization"] == "async"

    bad_json = run_entry(
        "train_online",
        ["sac", "--env-backend", "gymnasium", "--gymnasium.env-kwargs-json", "[1]", *common],
    )
    assert bad_json.returncode != 0
    assert "Invalid gymnasium.env_kwargs_json: expected a JSON object." in bad_json.stderr

    inactive = run_entry(
        "train_online",
        ["sac", "--env-backend", "maniskill", "--gymnasium.vectorization", "async", *common],
    )
    assert inactive.returncode != 0
    assert "'gymnasium.vectorization' is inactive" in inactive.stderr
    assert list(tmp_path.iterdir()) == [preset]


ODD_KEY_MAP = {"pixels": "rgb_front", "proprio": "state"}


def test_obs_key_map_renames_spaces_observations_and_final_observation():
    env = make_gymnasium_env(_pixels_cfg(env_id="RlgTestOddKeys-v0", obs_key_map=ODD_KEY_MAP))

    assert list(env.single_observation_space.spaces) == ["rgb_front", "state"]
    assert list(env.observation_space.spaces) == ["rgb_front", "state"]
    assert env.observation_space["rgb_front"].shape == (2, 16, 16, 3)
    obs, _ = env.reset()
    assert set(obs) == {"rgb_front", "state"}
    for _ in range(3):
        obs, _, _, truncated, infos = env.step(torch.zeros((2, 1)))

    assert set(obs) == {"rgb_front", "state"}
    assert bool(truncated.all())
    assert bool(infos["_final_observation"].all())
    final_obs = infos["final_observation"]
    assert set(final_obs) == {"rgb_front", "state"}
    assert final_obs["rgb_front"].shape == (2, 16, 16, 3)
    assert final_obs["rgb_front"].dtype == torch.uint8
    assert final_obs["state"].shape == (2, 2)
    env.close()


def test_obs_key_map_from_cli_config_passes_registry_validation():
    req = _make_req(
        env_id="RlgTestOddKeys-v0",
        num_envs=2,
        observation=ObservationConfig(rgb=("front",)),
        backend_config=GymnasiumConfig(
            import_modules=USER_ENVS, obs_key_map=("pixels=rgb_front", "proprio=state")
        ),
        create_eval_env=False,
    )

    train_env, _ = make_training_envs("gymnasium", req)

    obs, _ = train_env.reset()
    assert obs["rgb_front"].shape == (2, 16, 16, 3)
    assert obs["state"].shape == (2, 2)
    train_env.close()


def test_obs_key_map_then_frame_stack_stacks_renamed_image_key():
    env = make_gymnasium_env(
        _pixels_cfg(
            env_id="RlgTestOddKeys-v0",
            obs_key_map=ODD_KEY_MAP,
            image_keys=("rgb_front",),
            frame_stack=3,
        )
    )

    obs, _ = env.reset()
    assert obs["rgb_front"].shape == (2, 3, 16, 16, 3)
    assert obs["state"].shape == (2, 2)
    obs, _, _, _, _ = env.step(torch.zeros((2, 1)))
    assert obs["rgb_front"].shape == (2, 3, 16, 16, 3)
    env.close()


@pytest.mark.parametrize(
    ("backend_config", "error", "message"),
    [
        (GymnasiumConfig(obs_key_map=("pixels",)), ValueError, "entry 'pixels' must be 'src=dst'"),
        (GymnasiumConfig(obs_key_map=("=state",)), ValueError, "entry '=state' must be 'src=dst'"),
        (GymnasiumConfig(obs_key_map=("a=b=c",)), ValueError, "entry 'a=b=c' must be 'src=dst'"),
        (
            GymnasiumConfig(obs_key_map=("pixels=rgb_front", "pixels=state")),
            ValueError,
            "duplicate source/destination key 'pixels'",
        ),
        (
            GymnasiumConfig(obs_key_map=("pixels=state", "proprio=state")),
            ValueError,
            "duplicate source/destination key 'state'",
        ),
        (
            GymnasiumConfig(obs_key_map=("pixels=rgb_side",)),
            ObservationContractError,
            r"destination 'rgb_side' is not an expected observation key \['rgb_front', 'state'\]",
        ),
        (
            GymnasiumConfig(vector_env_factory="my_pkg.vec"),
            ValueError,
            "vector_env_factory 'my_pkg.vec' must be 'pkg.module:function'",
        ),
        (
            GymnasiumConfig(vector_env_factory="my-pkg.vec:make"),
            ValueError,
            "must be 'pkg.module:function'",
        ),
        (
            GymnasiumConfig(vector_env_factory="my_pkg.vec:make:extra"),
            ValueError,
            "must be 'pkg.module:function'",
        ),
        (
            GymnasiumConfig(vector_env_factory="my_pkg.vec:make", vectorization="async"),
            ValueError,
            "vectorization is not used with vector_env_factory",
        ),
        (
            GymnasiumConfig(vector_env_returns_torch=True),
            ValueError,
            "vector_env_returns_torch requires vector_env_factory",
        ),
    ],
)
def test_resolve_config_rejects_invalid_key_map_and_factory_settings(
    backend_config, error, message
):
    req = _make_req(observation=ObservationConfig(rgb=("front",)), backend_config=backend_config)

    with pytest.raises(error, match=message):
        GymnasiumBackend.resolve_config(req, is_eval=False)


def test_resolve_config_parses_key_map_and_factory_fields():
    backend_config = GymnasiumConfig(
        obs_key_map=("pixels=rgb_front", "proprio=state"),
        vector_env_factory="my_pkg.vec:make",
        vector_env_returns_torch=True,
    )
    req = _make_req(observation=ObservationConfig(rgb=("front",)), backend_config=backend_config)

    cfg = GymnasiumBackend.resolve_config(req, is_eval=False)

    assert cfg.obs_key_map == ODD_KEY_MAP
    assert cfg.vector_env_factory == "my_pkg.vec:make"
    assert cfg.vector_env_returns_torch is True


def test_obs_key_map_source_missing_from_env_raises():
    with pytest.raises(
        ObservationContractError,
        match=r"source 'pixels' not in env observation keys \['rgb_front', 'state'\]",
    ):
        make_gymnasium_env(_pixels_cfg(obs_key_map={"pixels": "rgb_front"}))


def test_obs_key_map_onto_existing_env_key_raises():
    with pytest.raises(ObservationContractError, match="renames a key onto another"):
        make_gymnasium_env(_pixels_cfg(obs_key_map={"rgb_front": "state"}))


def test_obs_key_map_requires_dict_observation_space():
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    from rl_garden.envs.gymnasium.env import _RenameObservationKeys
    from rl_garden.envs.vector_env import TorchVectorEnvAdapter

    box_env = TorchVectorEnvAdapter(
        SyncVectorEnv([lambda: gym.make("Pendulum-v1")], autoreset_mode=AutoresetMode.SAME_STEP),
        device="cpu",
    )
    with pytest.raises(
        ObservationContractError, match="obs_key_map requires a Dict observation space"
    ):
        _RenameObservationKeys(box_env, {"obs": "state"})
    box_env.close()


def _factory_cfg(function: str, **overrides) -> GymnasiumEnvConfig:
    return _pixels_cfg(vector_env_factory=f"_gymnasium_user_envs:{function}", **overrides)


def test_numpy_factory_goes_through_torch_adapter():
    env = make_gymnasium_env(_factory_cfg("make_numpy_vec"))

    obs, _ = env.reset()
    assert obs["rgb_front"].dtype == torch.uint8
    assert obs["rgb_front"].device.type == "cpu"
    assert obs["state"].dtype == torch.float32
    for _ in range(3):
        _, rewards, terminated, truncated, infos = env.step(torch.zeros((2, 1)))
    assert rewards.dtype == torch.float32
    assert terminated.dtype == torch.bool and bool(truncated.all())
    assert torch.equal(infos["final_info"]["episode"]["r"].float(), torch.full((2,), 3.0))
    env.close()


def test_numpy_factory_forwards_env_kwargs_and_applies_key_map_via_registry():
    req = _make_req(
        env_id="RlgTestOddKeys-v0",
        num_envs=2,
        observation=ObservationConfig(rgb=("front",)),
        backend_config=GymnasiumConfig(
            import_modules=USER_ENVS,
            vector_env_factory="_gymnasium_user_envs:make_numpy_vec",
            env_kwargs_json='{"env_id": "RlgTestOddKeys-v0"}',
            obs_key_map=("pixels=rgb_front", "proprio=state"),
        ),
    )

    train_env, eval_env = make_training_envs("gymnasium", req)

    obs, _ = train_env.reset()
    assert obs["rgb_front"].shape == (2, 16, 16, 3)
    assert eval_env.num_envs == 2
    train_env.close()
    eval_env.close()


def test_torch_factory_is_used_without_adapter_and_replays_seed():
    from _gymnasium_user_envs import TorchCountingVecEnv

    from rl_garden.envs.vector_env import TorchVectorEnvAdapter

    env = make_gymnasium_env(
        _factory_cfg("make_torch_vec", seed=5, vector_env_returns_torch=True, reward_scale=2.0)
    )
    inner = env
    while not isinstance(inner, TorchCountingVecEnv):
        assert not isinstance(inner, TorchVectorEnvAdapter)
        inner = inner.env

    first, _ = env.reset()
    for _ in range(3):
        _, rewards, _, truncated, infos = env.step(torch.zeros((2, 1)))
    second, _ = env.reset()

    assert inner.reset_seeds == [5, 5]
    assert torch.equal(first["state"], second["state"])
    assert bool(truncated.all())
    assert torch.allclose(rewards, torch.full((2,), 2.0))
    assert torch.equal(infos["final_info"]["episode"]["r"], torch.full((2,), 3.0))
    env.close()


@pytest.mark.parametrize(
    ("function", "error", "message"),
    [
        (
            "make_next_step_vec",
            ValueError,
            "vector env must use AutoresetMode.SAME_STEP, got metadata",
        ),
        (
            "make_box_vec",
            ObservationContractError,
            "vector_env_factory must return a Dict observation space .*got Box",
        ),
        ("make_missing_vec", AttributeError, "has no attribute 'make_missing_vec'"),
    ],
)
def test_factory_result_is_rejected(function, error, message):
    with pytest.raises(error, match=message):
        make_gymnasium_env(_factory_cfg(function))


def test_resolve_config_does_not_import_vector_env_factory_module():
    _run_fresh_interpreter(
        """
import sys
from rl_garden.common.env_args import GymnasiumConfig
from rl_garden.envs.backend_registry import _REGISTRY, EnvRequest, discover_env_backends
from rl_garden.observations.config import ObservationConfig

discover_env_backends()
req = EnvRequest(
    env_id="RlgTestPixels-v0", num_envs=2, control_mode="n/a", render_mode="rgb_array",
    seed=0, observation=ObservationConfig(),
    backend_config=GymnasiumConfig(
        vector_env_factory="_gymnasium_user_envs:make_torch_vec",
        vector_env_returns_torch=True,
    ),
)
cfg = _REGISTRY["gymnasium"].resolve_config(req, is_eval=False)
assert cfg.vector_env_factory == "_gymnasium_user_envs:make_torch_vec"
assert "_gymnasium_user_envs" not in sys.modules
"""
    )


def test_print_config_accepts_key_map_and_factory_fields(tmp_path):
    common = ["--print-config", "--log-type", "none", "--log-dir", str(tmp_path)]

    cli = run_entry(
        "train_online",
        [
            "sac",
            "--env-backend",
            "gymnasium",
            "--env-id",
            "MyReach-v0",
            "--gymnasium.obs-key-map",
            "proprio=state",
            "pixels=rgb_front",
            "--gymnasium.vector-env-factory",
            "rlg_not_installed.vec:make",
            *common,
        ],
    )
    assert cli.returncode == 0, cli.stderr
    config = json.loads(cli.stdout)["active_environment"]["config"]
    assert config["obs_key_map"] == ["proprio=state", "pixels=rgb_front"]
    assert config["vector_env_factory"] == "rlg_not_installed.vec:make"
    assert config["vector_env_returns_torch"] is False

    preset = tmp_path / "gymnasium_factory.yaml"
    preset.write_text(
        "env_backend: gymnasium\n"
        "env_id: MyReach-v0\n"
        "gymnasium:\n"
        "  obs_key_map: [proprio=state]\n"
        "  vector_env_factory: rlg_not_installed.vec:make\n"
        "  vector_env_returns_torch: true\n",
        encoding="utf-8",
    )
    yaml = run_entry("train_online", ["sac", "--config", str(preset), *common])
    assert yaml.returncode == 0, yaml.stderr
    config = json.loads(yaml.stdout)["active_environment"]["config"]
    assert config["obs_key_map"] == ["proprio=state"]
    assert config["vector_env_returns_torch"] is True
    assert "rlg_not_installed" not in sys.modules
