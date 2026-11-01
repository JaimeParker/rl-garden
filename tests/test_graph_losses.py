"""Policy graph ``losses:``: auxiliary losses on extra nodes trained with the
PPO loss, their targets (``info.<key>`` buffer extras, ``batch.rewards``,
``batch.dones``), masking under decision-level PPO, weight 0, metrics and
validation."""

from __future__ import annotations

import dataclasses
import math
import warnings

import pytest
import torch
import yaml
from _decision_fakes import NUM_ENVS, NUM_STEPS, _FixedHoldVecEnv

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.effective_config import ConfigError

AGENTS = [
    pytest.param(PPO, {}, False, id="ppo"),
    pytest.param(RecurrentPPO, {"rnn_hidden_size": 8}, True, id="gru"),
]


def _graph(
    *,
    stateful: bool = False,
    watcher_detach: bool = False,
    output_dim: int = 1,
    loss: dict | None = None,
    extra_losses: list[dict] | None = None,
) -> str:
    """obs -> encoder -> trunk [-> gru] -> {actor, value, watcher}; the
    watcher carries ``loss`` (an mse on ``batch.rewards`` by default)."""
    nodes = {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "trunk": {"type": "mlp", "net_arch": [8]},
        "watcher": {"type": "mlp", "net_arch": [8], "output_dim": output_dim},
        "actor": {"type": "actor"},
        "value": {"type": "value"},
    }
    edges = [{"from": "obs", "to": "encoder"}, {"from": "encoder", "to": "trunk"}]
    head_source = "trunk"
    if stateful:
        nodes["gru"] = {"type": "gru", "hidden_size": 8}
        edges.append({"from": "trunk", "to": "gru"})
        head_source = "gru"
    edges += [
        {"from": head_source, "to": "actor"},
        {"from": head_source, "to": "value"},
        {"from": head_source, "to": "watcher", "detach": watcher_detach},
    ]
    losses = [{"node": "watcher", **(loss or {"type": "mse", "target": "batch.rewards"})}]
    for extra in extra_losses or []:
        nodes[extra["node"]] = {"type": "mlp", "net_arch": [4], "output_dim": 1}
        edges.append({"from": head_source, "to": extra["node"]})
        losses.append(extra)
    return yaml.safe_dump({"nodes": nodes, "edges": edges, "losses": losses})


def _make_agent(agent_cls, env=None, **kwargs):
    defaults = {
        "device": "cpu",
        "num_steps": NUM_STEPS,
        "num_minibatches": 2,
        "update_epochs": 2,
        "eval_freq": 0,
        "log_freq": 0,
        "std_log": False,
        "target_kl": None,
        "net_arch": [8],
        "seed": 0,
    }
    defaults.update(kwargs)
    return agent_cls(env if env is not None else _FixedHoldVecEnv(emit_decisions=False), **defaults)


def _learn(agent, windows: int = 1) -> list[dict[str, float]]:
    """Learn ``windows`` windows; returns every ``train()`` metrics dict."""
    train = agent.train
    recorded: list[dict[str, float]] = []

    def recording_train() -> dict[str, float]:
        recorded.append(train())
        return recorded[-1]

    agent.train = recording_train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.learn(total_timesteps=windows * agent.num_envs * agent.num_steps)
    return recorded


def _skip_update() -> dict[str, float]:
    return {}


def _collect_window(agent) -> None:
    """One real rollout, no update: the buffer keeps the window."""
    agent.train = _skip_update
    _learn(agent)


def _first_minibatch(agent):
    if agent.stateful_policy:
        return next(agent.rollout_buffer.get_sequences(
            agent.num_minibatches, initial_hidden=agent._rollout_initial_hidden
        ))
    return next(agent.rollout_buffer.get(agent.minibatch_size))


class _ConstantRewardEnv(_FixedHoldVecEnv):
    """Every reward is 2: a target the watcher starts far from."""

    def step(self, action):
        obs, _, terminated, truncated, infos = super().step(action)
        return obs, torch.full_like(self.env_index, 2.0), terminated, truncated, infos


class _InfoTargetEnv(_FixedHoldVecEnv):
    """Reports ``infos["target"]``: the state the step acted on, so a stored
    target must equal the stored observation row for row; and
    ``infos["label"]``: an integer class per env. ``drop_target_at_step``
    omits ``target`` on that step."""

    def __init__(self, drop_target_at_step: int | None = None, **kwargs) -> None:
        super().__init__(emit_decisions=False, **kwargs)
        self.drop_target_at_step = drop_target_at_step

    def step(self, action):
        acted_on, step = self.last_state, self._t
        obs, rewards, terminated, truncated, infos = super().step(action)
        if step != self.drop_target_at_step:
            infos["target"] = acted_on.clone()
        infos["label"] = self.env_index.long() % 3
        return obs, rewards, terminated, truncated, infos


def _aux_grads(agent) -> dict[str, bool]:
    """Backpropagate only the graph losses of one minibatch; which node
    groups got a nonzero gradient."""
    data = _first_minibatch(agent)
    evaluation = agent._evaluate_minibatch(data)
    aux_loss, _ = agent._graph_aux_loss(
        data, evaluation, agent._minibatch_masks(data).get("decision_mask")
    )
    agent.policy.zero_grad(set_to_none=True)
    aux_loss.backward()

    def touched(prefix: str) -> bool:
        return any(
            param.grad is not None and bool(param.grad.abs().sum() > 0)
            for name, param in agent.policy.named_parameters()
            if name.startswith(prefix)
        )

    return {
        "watcher": touched("graph_nodes.watcher."),
        "trunk": touched("graph_nodes.trunk."),
        "actor": touched("actor."),
        "value": touched("value_net."),
    }


# --- (a) an mse watcher on batch.rewards -------------------------------------------------


@pytest.mark.parametrize("agent_cls,extra_kwargs,stateful", AGENTS)
def test_watcher_loss_decreases_over_windows(agent_cls, extra_kwargs, stateful):
    agent = _make_agent(
        agent_cls,
        _ConstantRewardEnv(emit_decisions=False),
        policy_graph=_graph(stateful=stateful),
        learning_rate=1e-2,
        **extra_kwargs,
    )
    losses = [metrics["graph_loss/watcher"] for metrics in _learn(agent, windows=3)]
    assert losses[0] > losses[1] > losses[2]


@pytest.mark.parametrize("agent_cls,extra_kwargs,stateful", AGENTS)
@pytest.mark.parametrize("watcher_detach", [False, True], ids=["live", "detached"])
def test_watcher_edge_detach_decides_whether_the_trunk_gets_its_gradient(
    agent_cls, extra_kwargs, stateful, watcher_detach
):
    agent = _make_agent(
        agent_cls,
        policy_graph=_graph(stateful=stateful, watcher_detach=watcher_detach),
        **extra_kwargs,
    )
    _collect_window(agent)
    assert _aux_grads(agent) == {
        "watcher": True, "trunk": not watcher_detach, "actor": False, "value": False,
    }


# --- (b) info.<key> targets: stored, sliced, aligned -------------------------------------


@pytest.mark.parametrize("agent_cls,extra_kwargs,stateful", AGENTS)
def test_info_targets_are_stored_and_sliced_with_their_observations(
    agent_cls, extra_kwargs, stateful
):
    agent = _make_agent(
        agent_cls,
        _InfoTargetEnv(),
        policy_graph=_graph(
            stateful=stateful, output_dim=4, loss={"type": "mse", "target": "info.target"}
        ),
        **extra_kwargs,
    )
    _collect_window(agent)
    buffer = agent.rollout_buffer
    assert buffer.extra_specs == {"target": ((4,), torch.float32)}
    assert torch.equal(buffer.extras["target"], buffer.obs["state"])

    if stateful:
        samples = list(buffer.get_sequences(agent.num_minibatches, agent._rollout_initial_hidden))
    else:
        samples = list(buffer.get(agent.minibatch_size))
    for data in samples:
        state = data.obs["state"]
        assert torch.equal(data.extras["target"], state)
        # Coordinates from the observation: column 1 is env / 10, column 2
        # the step of this (first) window / 10.
        env_ids = torch.round(state[..., 1] * 10).long()
        steps = torch.round(state[..., 2] * 10).long()
        assert torch.equal(data.rewards, buffer.rewards[steps, env_ids])
        assert torch.equal(data.dones, buffer.dones[steps, env_ids])
    metrics = type(agent).train(agent)
    assert math.isfinite(metrics["graph_loss/watcher"])


def test_a_missing_info_target_raises_naming_the_key():
    agent = _make_agent(
        PPO,
        _InfoTargetEnv(drop_target_at_step=3),
        policy_graph=_graph(output_dim=4, loss={"type": "mse", "target": "info.target"}),
    )
    with pytest.raises(ValueError, match=r"'info\.target' is missing from infos at rollout step 3"):
        _learn(agent)


def test_info_target_with_a_wrong_shape_raises():
    agent = _make_agent(
        PPO,
        _InfoTargetEnv(),
        policy_graph=_graph(output_dim=3, loss={"type": "mse", "target": "info.target"}),
    )
    with pytest.raises(ValueError, match=r"infos\['target'\].*must have shape \(4, 3\)"):
        _learn(agent)


def test_cross_entropy_trains_on_integer_info_labels():
    agent = _make_agent(
        PPO,
        _InfoTargetEnv(),
        policy_graph=_graph(output_dim=3, loss={"type": "cross_entropy", "target": "info.label"}),
    )
    assert agent.rollout_buffer.extra_specs == {"label": ((), torch.int64)}
    metrics = _learn(agent)[0]
    assert metrics["graph_loss/watcher"] > 0
    assert agent.rollout_buffer.extras["label"].dtype == torch.int64


def test_buffer_extras_are_stored_on_every_step_or_none():
    buffer = RolloutBuffer(
        _FixedHoldVecEnv.single_observation_space,
        _FixedHoldVecEnv.single_action_space,
        num_steps=3,
        num_envs=2,
        device="cpu",
        extra_specs={"target": ((2,), torch.float32)},
    )
    step = {
        "obs": {"state": torch.zeros(2, 4)},
        "actions": torch.zeros(2, 2),
        "rewards": torch.zeros(2),
        "dones": torch.zeros(2),
        "values": torch.zeros(2),
        "log_probs": torch.zeros(2),
    }
    with pytest.raises(ValueError, match=r"got extras \['other'\].*declares \['target'\]"):
        buffer.add(**step, extras={"other": torch.zeros(2, 2)})
    buffer.add(**step, extras={"target": torch.ones(2, 2)})
    buffer.add(**step)
    buffer.add(**step)
    with pytest.raises(ValueError, match=r"1 of 3 rollout steps passed the extras \['target'\]"):
        buffer.compute_returns_and_advantage(torch.zeros(2), torch.zeros(2))

    buffer.reset()
    assert not buffer.window_has_extras
    for _ in range(3):
        buffer.add(**step)
    buffer.compute_returns_and_advantage(torch.zeros(2), torch.zeros(2))
    assert all(sample.extras == {} for sample in buffer.get(2))


# --- (c) mask: decision with a minibatch without decisions -------------------------------


def test_masked_loss_on_a_minibatch_without_decisions_is_a_connected_zero():
    num_steps, update_epochs = 5, 3
    # Env 1 decides at steps 0 and 12: the second window (steps 5-9) has none.
    agent = _make_agent(
        RecurrentPPO,
        _FixedHoldVecEnv(holds=(1, 12), episode_ends={}),
        num_steps=num_steps,
        update_epochs=update_epochs,
        decision_level=True,
        rnn_hidden_size=8,
        policy_graph=_graph(
            stateful=True,
            loss={"type": "bce_with_logits", "target": "batch.dones", "mask": "decision"},
        ),
    )
    graph_aux_loss = agent._graph_aux_loss
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_graph_aux_loss(data, evaluation, decision_mask):
        aux_loss, node_losses = graph_aux_loss(data, evaluation, decision_mask)
        recorded.append((aux_loss, node_losses["watcher"]))
        return aux_loss, node_losses

    agent._graph_aux_loss = recording_graph_aux_loss
    metrics = _learn(agent, windows=2)

    second_window = recorded[-2 * update_epochs:]
    zeros = [aux for aux, node_loss in second_window if node_loss == 0]
    assert len(zeros) == update_epochs
    assert all(aux.requires_grad and aux.grad_fn is not None for aux in zeros)
    assert all(node_loss > 0 for _, node_loss in recorded[: 2 * update_epochs])
    assert math.isfinite(metrics[-1]["graph_loss/watcher"])
    assert all(torch.isfinite(param).all() for param in agent.policy.parameters())


# --- (d) weight 0 trains nothing --------------------------------------------------------


@pytest.mark.parametrize("agent_cls,extra_kwargs,stateful", AGENTS)
def test_zero_weight_loss_leaves_training_bit_identical(agent_cls, extra_kwargs, stateful):
    def trained(weight: float, *, drop_losses: bool = False) -> dict:
        loss = {"type": "mse", "target": "batch.rewards", "weight": weight}
        agent = _make_agent(
            agent_cls, policy_graph=_graph(stateful=stateful, loss=loss), **extra_kwargs
        )
        if drop_losses:
            agent.policy.graph_spec = dataclasses.replace(agent.policy.graph_spec, losses=())
        metrics = _learn(agent)[0]
        assert ("graph_loss/watcher" in metrics) is not drop_losses
        return agent.policy.state_dict()

    reference = trained(0.0, drop_losses=True)
    zero_weight = trained(0.0)
    live = trained(1.0)
    assert zero_weight.keys() == reference.keys() == live.keys()
    for name, tensor in reference.items():
        assert torch.equal(zero_weight[name], tensor), name
    assert any(not torch.equal(live[name], tensor) for name, tensor in reference.items())


# --- (e) metrics ------------------------------------------------------------------------


def test_every_loss_node_is_logged_and_graphs_without_losses_log_none():
    graph = _graph(
        extra_losses=[{"node": "probe", "type": "bce_with_logits", "target": "batch.dones"}]
    )
    metrics = _learn(_make_agent(PPO, policy_graph=graph))[0]
    assert {"graph_loss/watcher", "graph_loss/probe"} <= metrics.keys()
    assert all(math.isfinite(metrics[key]) for key in ("graph_loss/watcher", "graph_loss/probe"))

    plain = _learn(_make_agent(PPO, policy_graph="builtin:mlp"))[0]
    assert not any(key.startswith("graph_loss/") for key in plain)


def test_loss_nodes_run_only_in_the_update():
    agent = _make_agent(PPO, policy_graph=_graph())
    calls = []
    agent.policy.graph_nodes["watcher"].register_forward_hook(
        lambda *args: calls.append(1)
    )
    _collect_window(agent)
    assert calls == []
    agent._evaluate_minibatch(_first_minibatch(agent))
    assert calls == [1]


# --- (f) validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "graph,match",
    [
        pytest.param(
            _graph(output_dim=3),
            r"loss on node 'watcher': target 'batch.rewards' is one value per step, but the "
            r"node outputs 3",
            id="batch-target-on-a-wide-node",
        ),
        pytest.param(
            _graph(
                output_dim=1,
                loss={"type": "mse", "target": "info.target"},
                extra_losses=[{"node": "probe", "type": "cross_entropy", "target": "info.target"}],
            ),
            r"target 'info.target' is used by losses that need different layouts",
            id="one-key-two-layouts",
        ),
        pytest.param(
            _graph(loss={"type": "huber", "target": "batch.rewards"}),
            r"loss on node 'watcher' has type 'huber'",
            id="unknown-type",
        ),
        pytest.param(
            _graph(loss={"type": "mse", "target": "infos.target"}),
            r"loss on node 'watcher' has target 'infos.target'",
            id="bad-target-form",
        ),
    ],
)
def test_invalid_losses_are_rejected_at_construction(graph, match):
    with pytest.raises(ConfigError, match=match):
        _make_agent(PPO, policy_graph=graph)


def test_a_loss_on_a_sink_is_rejected():
    graph = yaml.safe_load(_graph())
    graph["losses"] = [{"node": "value", "type": "mse", "target": "batch.rewards"}]
    graph["edges"] = [edge for edge in graph["edges"] if edge["to"] != "watcher"]
    del graph["nodes"]["watcher"]
    with pytest.raises(ConfigError, match=r"loss on node 'value'"):
        _make_agent(PPO, policy_graph=yaml.safe_dump(graph))


def test_rollout_buffer_samples_carry_rewards_and_dones_for_every_agent():
    agent = _make_agent(PPO)
    _collect_window(agent)
    data = _first_minibatch(agent)
    assert data.rewards.shape == data.dones.shape == (NUM_ENVS * NUM_STEPS // 2,)
    assert data.extras == {}
