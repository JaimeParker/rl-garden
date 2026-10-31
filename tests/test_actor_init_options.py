"""Per-dimension initialisation options of ``DiagGaussianActor``.

``log_std_init_per_dim``, ``action_mean_bias_init`` and ``action_mean_head_gain``
default to ``None``, which must leave the constructed parameters (and RNG
consumption) identical to a construction that never mentions them. Setting
``action_mean_head_gain`` consumes extra RNG (``orthogonal_`` draws), so
modules built after the actor, such as the value network, legitimately differ
from a default construction; those tests only compare the actor head.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.encoders import FlattenExtractor
from rl_garden.networks import RecurrentLatentEncoder
from rl_garden.networks.actor_critic import DiagGaussianActor, _build_trunk
from rl_garden.policies import PPOPolicy, RecurrentPPOPolicy

OBS_DIM = 4
ACT_DIM = 3
HIDDEN = [8]
SEED = 7


def _state_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)


def _actor(**kwargs) -> DiagGaussianActor:
    return DiagGaussianActor(OBS_DIM, _action_space(), HIDDEN, **kwargs)


def _ppo_policy(**kwargs) -> PPOPolicy:
    obs_space = _state_space()
    return PPOPolicy(
        obs_space,
        _action_space(),
        FlattenExtractor(obs_space),
        net_arch=HIDDEN,
        **kwargs,
    )


def _recurrent_ppo_policy(**kwargs) -> RecurrentPPOPolicy:
    obs_space = _state_space()
    extractor = FlattenExtractor(obs_space)
    return RecurrentPPOPolicy(
        obs_space,
        _action_space(),
        extractor,
        RecurrentLatentEncoder(input_dim=extractor.features_dim, hidden_size=8),
        net_arch=HIDDEN,
        **kwargs,
    )


def _assert_state_dicts_equal(left: nn.Module, right: nn.Module) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key, value in left_state.items():
        assert torch.equal(value, right_state[key]), key


# --- direct DiagGaussianActor options ---


def test_log_std_init_per_dim_sets_each_dimension():
    values = [-0.5, -1.0, 0.25]
    actor = _actor(log_std_init_per_dim=values)

    assert actor.log_std.shape == (1, ACT_DIM)
    assert torch.equal(
        actor.log_std.detach(), torch.tensor(values, dtype=torch.float32).reshape(1, -1)
    )


def test_log_std_init_per_dim_accepts_tuple_and_overrides_scalar():
    actor = _actor(log_std_init=-3.0, log_std_init_per_dim=(0.0, 1.0, 2.0))

    assert torch.equal(actor.log_std.detach(), torch.tensor([[0.0, 1.0, 2.0]]))


def test_log_std_init_per_dim_wrong_length_raises():
    with pytest.raises(ValueError, match=rf"{ACT_DIM}.*2|2.*{ACT_DIM}"):
        _actor(log_std_init_per_dim=[0.0, 1.0])


def test_action_mean_bias_init_sets_bias():
    values = [0.5, -1.0, 2.0]
    actor = _actor(action_mean_bias_init=values)

    assert torch.equal(
        actor.mean.bias.detach(), torch.tensor(values, dtype=torch.float32)
    )


def test_action_mean_bias_init_wrong_length_raises():
    with pytest.raises(ValueError, match=rf"{ACT_DIM}.*4|4.*{ACT_DIM}"):
        _actor(action_mean_bias_init=[0.0, 1.0, 2.0, 3.0])


def test_action_mean_head_gain_makes_weight_orthogonal_with_zero_bias():
    gain = 0.01
    actor = _actor(action_mean_head_gain=gain)

    weight = actor.mean.weight.detach()
    smaller_gram = (
        weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
    )
    assert torch.allclose(
        smaller_gram, gain**2 * torch.eye(smaller_gram.shape[0]), atol=1e-7
    )
    assert torch.count_nonzero(actor.mean.bias) == 0


def test_gain_and_bias_together_keep_the_bias():
    bias = [0.5, -1.0, 2.0]
    actor = _actor(action_mean_head_gain=0.01, action_mean_bias_init=bias)

    assert torch.equal(
        actor.mean.bias.detach(), torch.tensor(bias, dtype=torch.float32)
    )


def test_defaults_match_plain_linear_mean_and_scalar_log_std():
    log_std_init = -0.75
    torch.manual_seed(SEED)
    actor = _actor(log_std_init=log_std_init)

    torch.manual_seed(SEED)
    _, trunk_dim = _build_trunk(
        OBS_DIM,
        HIDDEN,
        backbone_type="mlp",
        use_layer_norm=False,
        use_group_norm=False,
        num_groups=32,
        dropout_rate=None,
        kernel_init=None,
    )
    reference_mean = nn.Linear(trunk_dim, ACT_DIM)

    assert torch.equal(actor.mean.weight, reference_mean.weight)
    assert torch.equal(actor.mean.bias, reference_mean.bias)
    assert torch.equal(actor.log_std.detach(), torch.full((1, ACT_DIM), log_std_init))


# --- policies ---


@pytest.mark.parametrize("build_policy", [_ppo_policy, _recurrent_ppo_policy])
def test_policy_defaults_are_identical_to_omitting_the_options(build_policy):
    torch.manual_seed(SEED)
    omitted = build_policy(log_std_init=-0.25)
    torch.manual_seed(SEED)
    explicit_none = build_policy(
        log_std_init=-0.25,
        log_std_init_per_dim=None,
        action_mean_bias_init=None,
        action_mean_head_gain=None,
    )

    _assert_state_dicts_equal(omitted, explicit_none)
    assert torch.equal(
        omitted.actor.log_std.detach(), torch.full((1, ACT_DIM), -0.25)
    )


@pytest.mark.parametrize("build_policy", [_ppo_policy, _recurrent_ppo_policy])
def test_policy_forwards_options_to_actor(build_policy):
    policy = build_policy(
        log_std_init_per_dim=(-0.5, -1.0, -1.5),
        action_mean_bias_init=(0.1, 0.2, 0.3),
        action_mean_head_gain=0.01,
    )

    assert torch.equal(
        policy.actor.log_std.detach(), torch.tensor([[-0.5, -1.0, -1.5]])
    )
    assert torch.equal(policy.actor.mean.bias.detach(), torch.tensor([0.1, 0.2, 0.3]))
    weight = policy.actor.mean.weight.detach()
    assert torch.allclose(weight @ weight.T, 1e-4 * torch.eye(ACT_DIM), atol=1e-7)


# --- algorithms: checkpoint metadata ---


class _StubVecEnv:
    num_envs = 2
    single_observation_space = _state_space()
    single_action_space = _action_space()

    def __init__(self) -> None:
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, ACT_DIM), dtype=np.float32
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, OBS_DIM), {}

    def close(self) -> None:
        return None


def _agent_kwargs() -> dict:
    return {
        "device": "cpu",
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "std_log": False,
        "target_kl": None,
        "net_arch": HIDDEN,
        "seed": SEED,
    }


_INIT_OPTION_KEYS = (
    "log_std_init_per_dim",
    "action_mean_bias_init",
    "action_mean_head_gain",
)


@pytest.mark.parametrize(
    "agent_cls,extra_kwargs",
    [(PPO, {}), (RecurrentPPO, {"rnn_hidden_size": 8})],
)
def test_checkpoint_metadata_roundtrips_init_options(agent_cls, extra_kwargs):
    agent = agent_cls(
        _StubVecEnv(),
        log_std_init_per_dim=(-0.5, -1.0, -1.5),
        action_mean_bias_init=(0.1, 0.2, 0.3),
        action_mean_head_gain=0.01,
        **_agent_kwargs(),
        **extra_kwargs,
    )

    metadata = agent._checkpoint_metadata()

    assert metadata["log_std_init_per_dim"] == [-0.5, -1.0, -1.5]
    assert metadata["action_mean_bias_init"] == pytest.approx([0.1, 0.2, 0.3])
    assert metadata["action_mean_head_gain"] == 0.01
    assert isinstance(metadata["log_std_init_per_dim"], list)
    assert isinstance(metadata["action_mean_bias_init"], list)

    rebuilt = agent_cls(
        _StubVecEnv(),
        **{key: metadata[key] for key in _INIT_OPTION_KEYS},
        **_agent_kwargs(),
        **extra_kwargs,
    )
    assert torch.equal(
        rebuilt.policy.actor.log_std.detach(), agent.policy.actor.log_std.detach()
    )
    assert torch.equal(
        rebuilt.policy.actor.mean.bias.detach(), agent.policy.actor.mean.bias.detach()
    )


@pytest.mark.parametrize(
    "agent_cls,extra_kwargs",
    [(PPO, {}), (RecurrentPPO, {"rnn_hidden_size": 8})],
)
def test_checkpoint_metadata_defaults_are_none(agent_cls, extra_kwargs):
    agent = agent_cls(_StubVecEnv(), **_agent_kwargs(), **extra_kwargs)

    metadata = agent._checkpoint_metadata()

    for key in _INIT_OPTION_KEYS:
        assert metadata[key] is None


# --- CLI pass-through ---


@pytest.mark.parametrize(
    "algorithm,module,builder",
    [
        ("ppo", "rl_garden.training.online.ppo", "build_ppo"),
        (
            "recurrent_ppo",
            "rl_garden.training.online.recurrent_ppo",
            "build_recurrent_ppo",
        ),
        (
            "transformer_ppo",
            "rl_garden.training.online.transformer_ppo",
            "build_transformer_ppo",
        ),
    ],
)
def test_actor_init_cli_reaches_constructor(monkeypatch, algorithm, module, builder):
    import importlib

    from rl_garden.training import inspection
    from rl_garden.training.online import registry

    env = _StubVecEnv()
    action_dim = env.single_action_space.shape[0]
    log_std_per_dim = [-0.5 - 0.5 * i for i in range(action_dim)]
    bias = [(-1.0) ** i * float(i) for i in range(action_dim)]

    args = registry.parse_args(
        [
            algorithm,
            "--log-type",
            "none",
            "--log-std-init-per-dim",
            *[str(v) for v in log_std_per_dim],
            "--action-mean-bias-init",
            *[str(v) for v in bias],
            "--action-mean-head-gain",
            "0.01",
        ]
    )
    defaults = registry.parse_args([algorithm, "--log-type", "none"])
    assert defaults.log_std_init_per_dim is None
    assert defaults.action_mean_bias_init is None
    assert defaults.action_mean_head_gain is None

    captured: dict = {}

    def _capture_constructor_kwargs(target, /, **kwargs):
        captured.update(kwargs)
        return target

    monkeypatch.setattr(inspection, "construct_agent", _capture_constructor_kwargs)
    getattr(importlib.import_module(module), builder)(args, env, None, None, None)

    assert list(captured["log_std_init_per_dim"]) == log_std_per_dim
    assert list(captured["action_mean_bias_init"]) == bias
    assert captured["action_mean_head_gain"] == 0.01
