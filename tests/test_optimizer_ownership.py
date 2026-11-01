"""Optimizer ownership of every online SAC-family algorithm and of every
registered DDPG / TD3 / TD3+BC-family algorithm (online, offline, off2on).

For each such algorithm, under each ``encoder_sharing`` value it accepts: every
trainable policy parameter is held by some optimizer (nothing is silently left
untrained), no optimizer holds a parameter twice, and optimizers overlap
exactly where ``encoder_sharing`` says the trunk is trained by more than one
loss.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import DDPG, IQL, OPAL, FlashSAC, OfflineEnvSpec
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.algorithms.supe import load_opal_vae
from rl_garden.algorithms.td3_bc import TD3BCCore
from rl_garden.encoders.config import EncoderConfig
from rl_garden.envs.wrappers import SkillActionWrapper
from rl_garden.training.off2on.registry import registry as off2on_registry
from rl_garden.training.offline.registry import registry as offline_registry
from rl_garden.training.online.registry import registry

STATE_DIM = 4
ACTION_DIM = 2
SKILL_DIM = 3
SKILL_HORIZON = 3
IMAGE_ENCODER_CONFIG = EncoderConfig(features_dim=16, plain_conv_pooling="gap", proprio_latent_dim=4)

#: Registered without an ``algorithm_cls`` factory, so its class cannot be read
#: off the registry entry.
UNFACTORIED_SAC_FAMILY = {"flash_sac": FlashSAC}


class TinyVecEnv(gym.Env):
    """Two-env vector env with a ``state`` (and optionally ``rgb_cam``) Dict obs."""

    num_envs = 2

    def __init__(self, *, action_dim: int = ACTION_DIM, with_image: bool = False) -> None:
        single = {"state": spaces.Box(-1.0, 1.0, (STATE_DIM,), np.float32)}
        if with_image:
            single["rgb_cam"] = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.single_observation_space = spaces.Dict(single)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self, seed=None, options=None):
        del seed, options
        return self._observation(), {}

    def step(self, actions):
        del actions
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._observation(), torch.zeros(self.num_envs), done, done.clone(), {}

    def _observation(self) -> dict[str, torch.Tensor]:
        obs = {"state": torch.zeros(self.num_envs, STATE_DIM)}
        if "rgb_cam" in self.single_observation_space.spaces:
            obs["rgb_cam"] = torch.zeros(self.num_envs, 64, 64, 3, dtype=torch.uint8)
        return obs


COMMON_KWARGS = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 32,
    "batch_size": 4,
    "learning_starts": 0,
    "eval_freq": 0,
    "log_freq": 0,
    "std_log": False,
    "net_arch": [8],
}
SEQUENCE_KWARGS = {"burn_in_len": 2, "learning_len": 2, "forward_len": 1}
TRANSFORMER_KWARGS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_transformer_layers": 1,
    "memory_len": 2,
}


def _write_opal_dataset(path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for i in range(2):
            group = f.create_group(f"traj_{i}")
            group.create_dataset("obs", data=rng.standard_normal((9, STATE_DIM)).astype(np.float32))
            group.create_dataset(
                "actions", data=rng.uniform(-1, 1, (8, ACTION_DIM)).astype(np.float32)
            )
            group.create_dataset("rewards", data=np.zeros(8, dtype=np.float32))
            group.create_dataset("terminated", data=np.eye(1, 8, 7, dtype=bool)[0])
            group.create_dataset("truncated", data=np.zeros(8, dtype=bool))


@pytest.fixture(scope="module")
def pretrained_checkpoints(tmp_path_factory) -> dict[str, str]:
    """A JSRL guide checkpoint and a SUPE OPAL checkpoint, built once."""
    root = tmp_path_factory.mktemp("ownership_checkpoints")
    offline_env = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (STATE_DIM,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32),
        num_envs=1,
    )
    guide = root / "iql.pt"
    IQL(env=offline_env, device="cpu", net_arch=[8]).save(guide)

    dataset = root / "opal.h5"
    _write_opal_dataset(dataset)
    opal = root / "opal.pt"
    OPAL(
        env=offline_env,
        dataset_path=str(dataset),
        device="cpu",
        batch_size=4,
        skill_dim=SKILL_DIM,
        chunk_size=SKILL_HORIZON,
        hidden_size=8,
        vae_hidden_dims=(8, 8),
    ).save(opal)
    return {"jsrl_guide": str(guide), "supe_opal": str(opal)}


def _skill_env(opal_checkpoint: str) -> SkillActionWrapper:
    raw_env = TinyVecEnv()
    vae = load_opal_vae(opal_checkpoint, STATE_DIM, raw_env.single_action_space, device="cpu")
    return SkillActionWrapper(
        raw_env, vae.decoder, horizon=SKILL_HORIZON, skill_dim=SKILL_DIM, deterministic=False
    )


def _build_flash_sac(with_image, encoder_sharing, checkpoints):
    del with_image, encoder_sharing, checkpoints
    return FlashSAC(
        env=TinyVecEnv(),
        actor_hidden_dim=8,
        actor_num_blocks=1,
        critic_hidden_dim=8,
        critic_num_blocks=1,
        num_bins=11,
        **{key: value for key, value in COMMON_KWARGS.items() if key != "net_arch"},
    )


def _sac_like(**extra) -> Callable:
    """Builder for a SAC subclass that takes the standard SAC kwargs."""

    def build(algorithm_cls, with_image, encoder_sharing, checkpoints):
        del checkpoints
        kwargs = dict(COMMON_KWARGS, encoder_sharing=encoder_sharing, **extra)
        if with_image:
            kwargs["encoder_config"] = IMAGE_ENCODER_CONFIG
        return algorithm_cls(env=TinyVecEnv(with_image=with_image), **kwargs)

    return build


def _build_rlpd_hybrid(algorithm_cls, with_image, encoder_sharing, checkpoints):
    del checkpoints
    kwargs = dict(COMMON_KWARGS, encoder_sharing=encoder_sharing, discrete_hidden_dim=8)
    if with_image:
        kwargs["encoder_config"] = IMAGE_ENCODER_CONFIG
    env = TinyVecEnv(action_dim=ACTION_DIM + 1, with_image=with_image)
    return algorithm_cls(env=env, **kwargs)


def _build_jsrl(algorithm_cls, with_image, encoder_sharing, checkpoints):
    del with_image
    return algorithm_cls(
        env=TinyVecEnv(),
        guide_checkpoint=checkpoints["jsrl_guide"],
        guide_algorithm="iql",
        max_horizon=2,
        encoder_sharing=encoder_sharing,
        **COMMON_KWARGS,
    )


def _build_supe(algorithm_cls, with_image, encoder_sharing, checkpoints):
    del with_image
    return algorithm_cls(
        env=_skill_env(checkpoints["supe_opal"]),
        opal_checkpoint=checkpoints["supe_opal"],
        encoder_sharing=encoder_sharing,
        **COMMON_KWARGS,
    )


#: registry name -> (builder, variants, accepts image observations). A variant
#: is a label plus extra constructor kwargs.
BUILDERS: dict[str, tuple[Callable, dict[str, dict], bool]] = {
    "sac": (_sac_like(), {"": {}}, True),
    "sac_flow": (_sac_like(), {"": {}}, True),
    "rlpd": (_sac_like(), {"": {}}, True),
    "explore": (_sac_like(), {"": {}}, False),
    "acrlpd": (_sac_like(horizon_length=3), {"": {}}, False),
    "rlpd_hybrid": (_build_rlpd_hybrid, {"": {}}, True),
    "jsrl": (_build_jsrl, {"": {}}, False),
    "supe": (_build_supe, {"": {}}, False),
    "recurrent_sac": (
        _sac_like(**SEQUENCE_KWARGS, rnn_hidden_size=8),
        {"gru": {"rnn_type": "gru"}, "lstm": {"rnn_type": "lstm"}},
        True,
    ),
    "transformer_sac": (_sac_like(**SEQUENCE_KWARGS, **TRANSFORMER_KWARGS), {"": {}}, True),
}


def sac_family_classes() -> dict[str, type]:
    registry.discover()
    family: dict[str, type] = {}
    for name, entry in registry.entries().items():
        if entry.algorithm_cls is not None:
            algorithm_cls = entry.algorithm_cls()
        else:
            algorithm_cls = UNFACTORIED_SAC_FAMILY.get(name)
        if algorithm_cls is not None and issubclass(algorithm_cls, SACCore):
            family[name] = algorithm_cls
    return family


SAC_FAMILY = sac_family_classes()


def ownership_cases() -> list:
    cases = []
    for name, algorithm_cls in sorted(SAC_FAMILY.items()):
        if name == "flash_sac":
            cases.append(pytest.param(name, "", False, None, id="flash_sac"))
            continue
        _, variants, accepts_images = BUILDERS[name]
        observation_kinds = (False, True) if accepts_images else (False,)
        for variant in variants:
            for with_image in observation_kinds:
                for sharing in algorithm_cls.encoder_sharing_choices:
                    label = "-".join(
                        part
                        for part in (name, variant, "image" if with_image else "state", sharing)
                        if part
                    )
                    cases.append(pytest.param(name, variant, with_image, sharing, id=label))
    return cases


def build_agent(name: str, variant: str, with_image: bool, encoder_sharing, checkpoints):
    if name == "flash_sac":
        return _build_flash_sac(with_image, encoder_sharing, checkpoints)
    builder, variants, _ = BUILDERS[name]

    def algorithm_with_variant(**kwargs):
        return SAC_FAMILY[name](**kwargs, **variants[variant])

    return builder(algorithm_with_variant, with_image, encoder_sharing, checkpoints)


def optimizer_parameters(agent) -> dict[str, list[torch.nn.Parameter]]:
    optimizers = {}
    for name in agent._optimizer_names():
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            optimizers[name] = [p for group in optimizer.param_groups for p in group["params"]]
    return optimizers


def expected_cross_optimizer_parameters(agent) -> set[int]:
    """Ids of the parameters more than one optimizer is expected to hold: the
    trunk (extractor plus any sequence trunk) under ``"shared"``, else none."""
    policy = agent.policy
    if getattr(policy, "encoder_sharing", None) != "shared":
        return set()
    trunk = [policy.actor_extractor]
    if hasattr(policy, "recurrent_encoder"):
        trunk.append(policy.recurrent_encoder)
    return {id(p) for module in trunk for p in module.parameters()}


def test_every_registered_sac_family_algorithm_has_a_builder():
    assert set(SAC_FAMILY) - {"flash_sac"} == set(BUILDERS)


@pytest.mark.parametrize(("name", "variant", "with_image", "encoder_sharing"), ownership_cases())
def test_optimizers_own_every_trainable_policy_parameter(
    name, variant, with_image, encoder_sharing, pretrained_checkpoints
):
    agent = build_agent(name, variant, with_image, encoder_sharing, pretrained_checkpoints)
    assert_optimizers_own_every_trainable_policy_parameter(agent)


def assert_optimizers_own_every_trainable_policy_parameter(agent) -> None:
    optimizers = optimizer_parameters(agent)

    for optimizer_name, params in optimizers.items():
        assert len({id(p) for p in params}) == len(params), f"{optimizer_name} holds a duplicate"

    owned = {id(p) for params in optimizers.values() for p in params}
    orphaned = [
        param_name
        for param_name, param in agent.policy.named_parameters()
        if param.requires_grad and id(param) not in owned
    ]
    assert not orphaned, f"trainable parameters in no optimizer: {orphaned}"

    shared_ids: set[int] = set()
    for (_, left), (_, right) in combinations(optimizers.items(), 2):
        shared_ids |= {id(p) for p in left} & {id(p) for p in right}
    assert shared_ids == expected_cross_optimizer_parameters(agent)


# --- DDPG / TD3 / TD3+BC family ----------------------------------------------------------------

DDPG_FAMILY_BASES = (DDPG, TD3BCCore)
FAMILY_REGISTRIES = {"online": registry, "offline": offline_registry, "off2on": off2on_registry}
OFFLINE_KWARGS = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 32,
    "batch_size": 4,
    "std_log": False,
    "net_arch": [8],
}


def _build_ddpg_like(algorithm_cls, with_image, encoder_sharing):
    # DDPG and TD3 require an image observation key.
    assert with_image
    online_kwargs = {key: value for key, value in COMMON_KWARGS.items() if key != "net_arch"}
    return algorithm_cls(
        env=TinyVecEnv(with_image=True),
        feature_dim=8,
        hidden_dim=8,
        encoder_sharing=encoder_sharing,
        **online_kwargs,
    )


def _offline_env(with_image: bool) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        TinyVecEnv(with_image=with_image).single_observation_space,
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32),
        num_envs=2,
    )


def _build_offline_td3bc_like(**extra) -> Callable:
    def build(algorithm_cls, with_image, encoder_sharing):
        kwargs = dict(OFFLINE_KWARGS, encoder_sharing=encoder_sharing, **extra)
        if with_image:
            kwargs["encoder_config"] = IMAGE_ENCODER_CONFIG
        return algorithm_cls(env=_offline_env(with_image), **kwargs)

    return build


def _build_off2on_spot(algorithm_cls, with_image, encoder_sharing):
    kwargs = dict(COMMON_KWARGS, encoder_sharing=encoder_sharing, vae_hidden_dim=8)
    if with_image:
        kwargs["encoder_config"] = IMAGE_ENCODER_CONFIG
    return algorithm_cls(env=TinyVecEnv(with_image=with_image), vae_iterations=0, **kwargs)


#: "<phase>/<registry name>" -> (builder, observation kinds it accepts: with_image flags).
DDPG_FAMILY_BUILDERS: dict[str, tuple[Callable, tuple[bool, ...]]] = {
    "online/drqv2": (_build_ddpg_like, (True,)),
    "online/td3": (_build_ddpg_like, (True,)),
    "offline/td3_bc": (_build_offline_td3bc_like(), (False, True)),
    "offline/rebrac": (_build_offline_td3bc_like(), (False, True)),
    "offline/spot": (_build_offline_td3bc_like(vae_hidden_dim=8), (False, True)),
    "off2on/spot": (_build_off2on_spot, (False, True)),
}


def ddpg_family_classes() -> dict[str, type]:
    family: dict[str, type] = {}
    for phase, phase_registry in FAMILY_REGISTRIES.items():
        phase_registry.discover()
        for name, entry in phase_registry.entries().items():
            if entry.algorithm_cls is None:
                continue
            algorithm_cls = entry.algorithm_cls()
            if issubclass(algorithm_cls, DDPG_FAMILY_BASES):
                family[f"{phase}/{name}"] = algorithm_cls
    return family


DDPG_FAMILY = ddpg_family_classes()


def ddpg_family_ownership_cases() -> list:
    cases = []
    for key, algorithm_cls in sorted(DDPG_FAMILY.items()):
        _, observation_kinds = DDPG_FAMILY_BUILDERS[key]
        for with_image in observation_kinds:
            for sharing in algorithm_cls.encoder_sharing_choices:
                label = "-".join((key, "image" if with_image else "state", sharing))
                cases.append(pytest.param(key, with_image, sharing, id=label))
    return cases


def test_every_registered_ddpg_family_algorithm_has_a_builder():
    assert set(DDPG_FAMILY) == set(DDPG_FAMILY_BUILDERS)


@pytest.mark.parametrize(("key", "with_image", "encoder_sharing"), ddpg_family_ownership_cases())
def test_ddpg_family_optimizers_own_every_trainable_policy_parameter(
    key, with_image, encoder_sharing
):
    builder, _ = DDPG_FAMILY_BUILDERS[key]
    agent = builder(DDPG_FAMILY[key], with_image, encoder_sharing)
    assert agent.policy.encoder_sharing == encoder_sharing
    assert_optimizers_own_every_trainable_policy_parameter(agent)
