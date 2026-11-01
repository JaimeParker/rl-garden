"""Extension points of the PPO family (``policy_class``,
``_policy_head_kwargs``, ``PPOPolicy._build_actor``,
``rollout_extra_specs``, ``evaluation_outputs``/``return_outputs``,
``_aux_loss``), and a seeded ``state_dict`` snapshot that pins them
bit-identical for every existing user.

The snapshot digests were recorded on CPU (torch 2.6, one thread) BEFORE the
extension points existed; the refactor must reproduce them bit for bit. A
different torch build or BLAS may round differently: re-record there
instead of loosening the comparison.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from _decision_fakes import NUM_ENVS, NUM_STEPS, _ComponentMaskVecEnv, _FixedHoldVecEnv
from gymnasium import spaces
from test_graph_losses import (
    _collect_window,
    _first_minibatch,
    _InfoTargetEnv,
    _make_agent,
)

from rl_garden.algorithms import PPO, RecurrentPPO, TransformerPPO
from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.networks import MLPResNet
from rl_garden.networks.actor_critic import (
    DiagGaussianActor,
    DiagGaussianActorBase,
    SplitDiagGaussianActor,
)
from rl_garden.networks.graph.builtin import mlp_actor_critic
from rl_garden.policies import GraphPPOPolicy, PPOPolicy, RecurrentPPOPolicy

STATE_SPACE = _FixedHoldVecEnv.single_observation_space
ACTION_3 = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
MINIBATCH_ENVS = NUM_ENVS // 2  # every agent below uses num_minibatches=2


class _RecordingLogger:
    def __init__(self) -> None:
        self.scalars: dict[str, list[float]] = defaultdict(list)

    def add_scalar(self, key, value, step):
        self.scalars[key].append(value)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def actor_constructions(monkeypatch) -> list[type]:
    """The class of every ``DiagGaussianActorBase`` constructed while the
    test runs, in order."""
    built: list[type] = []
    module_init = nn.Module.__init__

    def counting_init(self, *args, **kwargs):
        built.append(type(self))
        module_init(self, *args, **kwargs)

    monkeypatch.setattr(DiagGaussianActorBase, "__init__", counting_init)
    return built


# --- policy_class and _policy_head_kwargs ----------------------------------------------------


class _TaggedPolicy(PPOPolicy):
    def __init__(self, *args, tag: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tag = tag


class _TaggedPPO(PPO):
    policy_class = _TaggedPolicy

    def _policy_head_kwargs(self, observation_space):
        return {**super()._policy_head_kwargs(observation_space), "tag": "custom"}


class _TaggedRecurrentPolicy(RecurrentPPOPolicy):
    def __init__(self, *args, tag: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tag = tag


class _TaggedRecurrentPPO(RecurrentPPO):
    policy_class = _TaggedRecurrentPolicy

    def _policy_head_kwargs(self, observation_space):
        return {**super()._policy_head_kwargs(observation_space), "tag": "custom"}


def test_policy_class_and_head_kwargs_select_and_configure_the_policy():
    policy = _make_agent(_TaggedPPO).policy
    assert type(policy) is _TaggedPolicy
    assert policy.tag == "custom"
    assert policy.actor.mean.in_features == 8  # the algorithm's net_arch still applies
    assert type(_make_agent(PPO).policy) is PPOPolicy


def test_sequence_policy_class_receives_the_recurrent_encoder():
    policy = _make_agent(_TaggedRecurrentPPO, rnn_hidden_size=8).policy
    assert type(policy) is _TaggedRecurrentPolicy
    assert policy.tag == "custom"
    assert policy.recurrent_encoder.features_dim == 8
    assert type(_make_agent(RecurrentPPO, rnn_hidden_size=8).policy) is RecurrentPPOPolicy


class _KwargRecordingRecurrentPolicy(RecurrentPPOPolicy):
    received: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        type(self).received = kwargs
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize(
    "agent_cls, extra_kwargs",
    [
        pytest.param(RecurrentPPO, {"rnn_hidden_size": 8}, id="gru"),
        pytest.param(TransformerPPO, {}, id="gtrxl"),
    ],
)
def test_sequence_ppo_applies_critic_backbone_type_to_the_value_head_only(
    agent_cls, extra_kwargs
):
    class Recording(agent_cls):
        policy_class = _KwargRecordingRecurrentPolicy

    kwargs = {**(TRANSFORMER_KWARGS if agent_cls is TransformerPPO else {}), **extra_kwargs}
    policy = _make_agent(Recording, critic_backbone_type="mlp_resnet", **kwargs).policy
    assert _KwargRecordingRecurrentPolicy.received["critic_backbone_type"] == "mlp_resnet"
    assert isinstance(policy.value_net.trunk, MLPResNet)
    assert not isinstance(policy.actor.trunk, MLPResNet)

    default = _make_agent(Recording, **kwargs).policy
    assert _KwargRecordingRecurrentPolicy.received["critic_backbone_type"] is None
    assert not isinstance(default.value_net.trunk, MLPResNet)


# --- _build_actor ----------------------------------------------------------------------------


class _MarkedActor(DiagGaussianActor):
    pass


class _MarkedActorPolicy(PPOPolicy):
    def _build_actor(self, features_dim, action_space, hidden_dims, **actor_options):
        return _MarkedActor(features_dim, action_space, hidden_dims, **actor_options)


class _MarkedActorPPO(PPO):
    policy_class = _MarkedActorPolicy


def test_build_actor_override_is_the_ppo_policy_actor(actor_constructions):
    agent = _make_agent(_MarkedActorPPO, log_std_init=-1.0)
    assert actor_constructions == [_MarkedActor]
    assert type(agent.policy.actor) is _MarkedActor
    assert torch.equal(agent.policy.actor.log_std, torch.full((1, 2), -1.0))


def test_a_policy_graph_still_builds_a_graph_policy_under_a_custom_policy_class():
    agent = _make_agent(_MarkedActorPPO, policy_graph="builtin:mlp")
    assert type(agent.policy) is GraphPPOPolicy
    assert type(agent.policy.actor) is DiagGaussianActor


class _RecordingGraphPolicy(GraphPPOPolicy):
    def _build_actor(self, features_dim, action_space, hidden_dims, **actor_options):
        actor = super()._build_actor(features_dim, action_space, hidden_dims, **actor_options)
        self.build_actor_calls = getattr(self, "build_actor_calls", 0) + 1
        return actor


TWO_SINK_GRAPH = {
    "nodes": {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "wide": {"type": "mlp", "net_arch": [6]},
        "narrow": {"type": "mlp", "net_arch": [5]},
        "arm": {"type": "actor", "action_dims": [0, 2]},
        "gripper": {"type": "actor", "action_dims": [2, 3], "net_arch": []},
        "value": {"type": "value"},
    },
    "edges": [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "wide"},
        {"from": "encoder", "to": "narrow"},
        {"from": "wide", "to": "arm"},
        {"from": "narrow", "to": "gripper"},
        {"from": "encoder", "to": "value"},
    ],
}


def test_several_actor_sinks_build_one_actor_through_build_actor(actor_constructions):
    policy = _RecordingGraphPolicy(STATE_SPACE, ACTION_3, graph=TWO_SINK_GRAPH, net_arch=[8])
    assert actor_constructions == [SplitDiagGaussianActor]
    assert policy.build_actor_calls == 1
    assert type(policy.actor) is SplitDiagGaussianActor
    arm, gripper = policy.actor.heads
    assert (arm.mean.in_features, arm.mean.out_features) == (8, 2)  # the pi widths, [8]
    assert (gripper.mean.in_features, gripper.mean.out_features) == (5, 1)  # net_arch: []
    assert list(dict(policy.named_children())) == ["actor", "value_net", "graph_nodes"]


def test_one_actor_sink_builds_the_ppo_policy_actor_through_build_actor(actor_constructions):
    policy = _RecordingGraphPolicy(STATE_SPACE, ACTION_3, graph=mlp_actor_critic(), net_arch=[8])
    assert actor_constructions == [DiagGaussianActor]
    assert policy.build_actor_calls == 1


# --- evaluation_outputs / return_outputs ------------------------------------------------------


class _LatentRecurrentPolicy(RecurrentPPOPolicy):
    """Reads the update-time latent: an auxiliary head on it."""

    evaluation_outputs = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_head = nn.Linear(self.recurrent_encoder.features_dim, 1)


class _LatentRecurrentPPO(RecurrentPPO):
    policy_class = _LatentRecurrentPolicy

    def _aux_loss(self, data, evaluation, decision_mask):
        latent = evaluation.outputs["latent"]
        self.seen_latents.append(latent)
        prediction = self.policy.aux_head(latent).squeeze(-1)
        loss = (prediction - evaluation.returns).pow(2).mean()
        return loss, {"latent_probe": loss.detach()}


class _LatentPolicy(PPOPolicy):
    evaluation_outputs = True


class _LatentPPO(PPO):
    policy_class = _LatentPolicy


def _sequence_actor_latent(policy: RecurrentPPOPolicy, data) -> torch.Tensor:
    """The actor's input for a ``(T, B)`` minibatch, computed by hand."""
    num_steps, num_envs = data.actions.shape[:2]
    raw = policy.extract_features(flatten_leading_dims(data.obs))
    latent, _ = policy.recurrent_encoder.forward_sequence(
        raw.reshape(num_steps, num_envs, -1), data.initial_hidden, data.episode_starts
    )
    return latent.reshape(num_steps * num_envs, -1)


def test_recurrent_latent_is_the_actor_features_of_the_minibatch():
    agent = _make_agent(_LatentRecurrentPPO, rnn_hidden_size=8)
    agent.seen_latents = []
    _collect_window(agent)
    data = _first_minibatch(agent)
    latent = agent._evaluate_minibatch(data).outputs["latent"]
    assert latent.shape == (NUM_STEPS * MINIBATCH_ENVS, 8)
    with torch.no_grad():
        assert torch.equal(latent, _sequence_actor_latent(agent.policy, data))


def test_aux_loss_on_the_latent_trains_a_head_and_is_logged():
    logger = _RecordingLogger()
    agent = _make_agent(_LatentRecurrentPPO, rnn_hidden_size=8, logger=logger, log_freq=1)
    agent.seen_latents = []
    initial_head = agent.policy.aux_head.weight.detach().clone()
    metrics = []
    train = agent.train
    agent.train = lambda: metrics.append(train()) or metrics[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.learn(total_timesteps=agent.num_envs * agent.num_steps)

    # 2 epochs x 2 minibatches, each (T * B, D).
    assert [tuple(latent.shape) for latent in agent.seen_latents] == [
        (NUM_STEPS * MINIBATCH_ENVS, 8)
    ] * 4
    grad = agent.policy.aux_head.weight.grad
    assert grad is not None and torch.count_nonzero(grad) > 0
    assert not torch.equal(agent.policy.aux_head.weight, initial_head)
    assert metrics[0]["aux/latent_probe"] > 0
    assert logger.scalars["losses/aux/latent_probe"] == [metrics[0]["aux/latent_probe"]]


def test_flat_latent_is_the_actor_features_and_the_default_aux_loss_adds_nothing():
    agent = _make_agent(_LatentPPO)
    _collect_window(agent)
    data = _first_minibatch(agent)
    evaluation = agent._evaluate_minibatch(data)
    assert evaluation.outputs.keys() == {"latent"}
    assert torch.equal(evaluation.outputs["latent"], agent.policy.extract_actor_features(data.obs))
    assert agent._aux_loss(data, evaluation, None) == (None, {})
    assert _make_agent(PPO)._evaluate_policy(lambda: (1, 2)) == ((1, 2), None)


def test_return_outputs_appends_the_latent_and_changes_nothing_else():
    torch.manual_seed(0)
    flat = _make_agent(PPO).policy
    obs = {"state": torch.randn(6, 4)}
    actions = torch.rand(6, 2) * 1.8 - 0.9
    for evaluate in (flat.evaluate_actions, flat.evaluate_actions_with_dist_params):
        *evaluated, outputs = evaluate(obs, actions, return_outputs=True)
        for actual, expected in zip(evaluated, evaluate(obs, actions), strict=True):
            assert torch.equal(actual, expected)
        assert torch.equal(outputs["latent"], flat.extract_actor_features(obs))

    recurrent = _make_agent(RecurrentPPO, rnn_hidden_size=8).policy
    sequence = (
        {"state": torch.randn(5, 3, 4)},
        torch.rand(5, 3, 2) * 1.8 - 0.9,
        recurrent.initial_hidden_state(3, torch.device("cpu")),
        torch.zeros(5, 3),
    )
    for evaluate in (
        recurrent.evaluate_actions_sequence,
        recurrent.evaluate_actions_sequence_with_dist_params,
    ):
        *evaluated, outputs = evaluate(*sequence, return_outputs=True)
        for actual, expected in zip(evaluated, evaluate(*sequence), strict=True):
            assert torch.equal(actual, expected)
        assert outputs["latent"].shape == (15, 8)


# --- rollout_extra_specs ---------------------------------------------------------------------


TARGET_SPECS = {"target": ((4,), torch.float32)}


class _TargetPolicy(PPOPolicy):
    def rollout_extra_specs(self):
        return TARGET_SPECS


class _TargetRecurrentPolicy(RecurrentPPOPolicy):
    def rollout_extra_specs(self):
        return TARGET_SPECS


class _TargetPPO(PPO):
    policy_class = _TargetPolicy


class _TargetRecurrentPPO(RecurrentPPO):
    policy_class = _TargetRecurrentPolicy


@pytest.mark.parametrize(
    "agent_cls,extra_kwargs",
    [
        pytest.param(_TargetPPO, {}, id="ppo"),
        pytest.param(_TargetRecurrentPPO, {"rnn_hidden_size": 8}, id="gru"),
    ],
)
def test_rollout_extra_specs_of_a_plain_policy_are_stored_and_sliced(agent_cls, extra_kwargs):
    agent = _make_agent(agent_cls, _InfoTargetEnv(), **extra_kwargs)
    _collect_window(agent)
    buffer = agent.rollout_buffer
    assert buffer.extra_specs == TARGET_SPECS
    # The env reports the state each step acted on: the stored observation.
    assert torch.equal(buffer.extras["target"], buffer.obs["state"])
    if agent.stateful_policy:
        samples = list(buffer.get_sequences(agent.num_minibatches, agent._rollout_initial_hidden))
    else:
        samples = list(buffer.get(agent.minibatch_size))
    for data in samples:
        assert torch.equal(data.extras["target"], data.obs["state"])
    assert _make_agent(PPO).rollout_buffer.extra_specs == {}

TRANSFORMER_KWARGS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_transformer_layers": 1,
    "mlp_num": 1,
    "memory_len": 2,
}


def _loss_graph(*, stateful: bool, mask: str = "none") -> str:
    """obs -> encoder -> trunk [-> gru] -> {actor, value, watcher}; the
    watcher carries an mse loss on ``batch.rewards``."""
    nodes = {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "trunk": {"type": "mlp", "net_arch": [8]},
        "watcher": {"type": "mlp", "net_arch": [8], "output_dim": 1},
        "actor": {"type": "actor"},
        "value": {"type": "value"},
    }
    edges = [{"from": "obs", "to": "encoder"}, {"from": "encoder", "to": "trunk"}]
    head_source = "trunk"
    if stateful:
        nodes["gru"] = {"type": "gru", "hidden_size": 8}
        edges.append({"from": "trunk", "to": "gru"})
        head_source = "gru"
    edges += [{"from": head_source, "to": sink} for sink in ("actor", "value", "watcher")]
    losses = [{"node": "watcher", "type": "mse", "target": "batch.rewards", "mask": mask}]
    return yaml.safe_dump({"nodes": nodes, "edges": edges, "losses": losses})


GRU = (RecurrentPPO, {"rnn_type": "gru", "rnn_hidden_size": 8})
#: case -> (algorithm, kwargs, decision-level env with a component mask).
STATE_CASES = {
    "ppo": (PPO, {}, False),
    "ppo-adaptive_kl": (PPO, {"lr_schedule": "adaptive_kl"}, False),
    "ppo-components": (PPO, {}, True),
    "gru": (*GRU, False),
    "gru-adaptive_kl": (GRU[0], {**GRU[1], "lr_schedule": "adaptive_kl"}, False),
    "gru-components": (*GRU, True),
    "gtrxl": (TransformerPPO, TRANSFORMER_KWARGS, False),
    "graph-builtin-mlp": (PPO, {"policy_graph": "builtin:mlp"}, False),
    "graph-builtin-recurrent": (GRU[0], {**GRU[1], "policy_graph": "builtin:recurrent"}, False),
    "graph-mlp-loss": (PPO, {"policy_graph": _loss_graph(stateful=False)}, False),
    "graph-mlp-loss-adaptive_kl": (
        PPO,
        {"policy_graph": _loss_graph(stateful=False), "lr_schedule": "adaptive_kl"},
        False,
    ),
    "graph-gru-loss": (GRU[0], {**GRU[1], "policy_graph": _loss_graph(stateful=True)}, False),
    "graph-gru-loss-components": (
        GRU[0],
        {**GRU[1], "policy_graph": _loss_graph(stateful=True, mask="decision")},
        True,
    ),
}

#: The build the digests below were recorded on (CPU only).
SNAPSHOT_TORCH_VERSION = "2.6.0+cu124"

#: sha256 of the policy ``state_dict`` after one seeded ``learn()`` window.
STATE_SNAPSHOT: dict[str, str] = {
    "graph-builtin-mlp": "161d3bf3b8bbc3aa33d94328cf334abe01c77c7db3b4de74f208f06d0dc7c8a6",
    "graph-builtin-recurrent": "2457752f826eac30dfeef9613b4bdc34ea2c553174050c5aafc5d7f161e6d32b",
    "graph-gru-loss": "65ac2532eac17afbf07518c7ffd02f13e7eb12d172aac52b09d4b70065ff6682",
    "graph-gru-loss-components": "cf0e6117dedbbfe39b1f4659dcf002440ac5122ae5162d85afea9bd7d99df5b5",
    "graph-mlp-loss": "2671f59dc8d5e0fe4d89575fe0a9b45f370a41497e1958dd7ed3ac60af8f8ee5",
    "graph-mlp-loss-adaptive_kl": "fe86e210a9787e37b1ee2c9fafb55cc6ca284231846c25ab2469db7e770a7d5d",
    "gru": "4b25ed3867503bbc9a466372e4546a877a2963e361fdad98628369d92f86f345",
    "gru-adaptive_kl": "a214a85e1478158b4a0df279416d84d1daa9e3d3e98d234b518f35310c6fe628",
    "gru-components": "39c78122417d2e9de5873b6b03e006eaa97d82a9814f8d721979188c2a5186a7",
    "gtrxl": "9d9a7be987215116d9a434dfac2efbf1647e00e1836ce502b374681bb20e8017",
    "ppo": "161d3bf3b8bbc3aa33d94328cf334abe01c77c7db3b4de74f208f06d0dc7c8a6",
    "ppo-adaptive_kl": "72f18d493b146181d81d4d00464542da9e5f4dc94c24713d470caa2499b2d8b9",
    "ppo-components": "a5b3ead7888606be127d26dd5ff581cf72b00c0dc59e838345d5567973ac8953",
}


def state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """sha256 over every entry's name, dtype, shape and bytes, in key order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trained_state_digest(case: str) -> str:
    """Digest of the policy after one seeded window of ``case``."""
    agent_cls, extra_kwargs, components = STATE_CASES[case]
    env = _ComponentMaskVecEnv() if components else _FixedHoldVecEnv(emit_decisions=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = agent_cls(
            env,
            device="cpu",
            num_steps=NUM_STEPS,
            num_minibatches=2,
            update_epochs=2,
            eval_freq=0,
            log_freq=0,
            std_log=False,
            target_kl=None,
            net_arch=[8],
            seed=0,
            decision_level=components,
            **extra_kwargs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
    finally:
        torch.set_num_threads(threads)
    return state_dict_digest(agent.policy.state_dict())


@pytest.mark.skipif(
    torch.__version__ != SNAPSHOT_TORCH_VERSION,
    reason=f"state_dict snapshot recorded on torch {SNAPSHOT_TORCH_VERSION} CPU; re-record",
)
@pytest.mark.parametrize("case", sorted(STATE_CASES))
def test_seeded_learn_window_state_dict_matches_the_recorded_snapshot(case):
    assert trained_state_digest(case) == STATE_SNAPSHOT[case]
