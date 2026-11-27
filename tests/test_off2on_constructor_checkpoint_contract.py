"""Constructor inheritance and shared checkpoint scheduling contracts."""
from collections import Counter
import dataclasses
import importlib
import inspect

import pytest

from rl_garden.algorithms.awac import _AWACRolloutTrainingShell
from rl_garden.algorithms.iql import _IQLRolloutTrainingShell
from rl_garden.algorithms.off2on_awac import Off2OnAWAC
from rl_garden.algorithms.off2on_iql import Off2OnIQL
from rl_garden.algorithms.off2on_spot import Off2OnSPOT
from rl_garden.algorithms.spot import _SPOTRolloutTrainingShell

from rl_garden.algorithms.floq import Off2OnFloQ, _FloQRolloutTrainingShell
from rl_garden.algorithms.fino import Off2OnFINO, _FINORolloutTrainingShell
from rl_garden.algorithms.fac import Off2OnFAC, _FACRolloutTrainingShell
from rl_garden.algorithms.value_flows import Off2OnValueFlows, _ValueFlowsRolloutTrainingShell


@pytest.mark.parametrize('cls,parent,extra', [
    (Off2OnIQL, _IQLRolloutTrainingShell, {'num_eval_episodes'}),
    (Off2OnAWAC, _AWACRolloutTrainingShell, set()),
    (Off2OnSPOT, _SPOTRolloutTrainingShell, set()),
    (Off2OnFloQ, _FloQRolloutTrainingShell, set()),
    (Off2OnFINO, _FINORolloutTrainingShell, set()),
    (Off2OnFAC, _FACRolloutTrainingShell, set()),
    (Off2OnValueFlows, _ValueFlowsRolloutTrainingShell, set()),
])
def test_off2on_constructor_contains_every_shell_parameter(cls, parent, extra):
    if getattr(parent, "params_class", None) is not None:
        assert cls.params_class is parent.params_class
        # Learning keywords are resolved by Params; operational keywords remain explicit.
        fields = {field.name for field in dataclasses.fields(cls.params_class)}
        assert not (fields & inspect.signature(cls).parameters.keys())
        assert "params" in inspect.signature(cls).parameters
        assert inspect.signature(cls).parameters["overrides"].kind is inspect.Parameter.VAR_KEYWORD
    actual = inspect.signature(cls).parameters
    expected = inspect.signature(parent).parameters
    assert set(actual) == set(expected) | extra
    for name, parameter in expected.items():
        assert actual[name].kind == parameter.kind, name
        assert actual[name].default == parameter.default, name
    if extra:
        parameter = actual['num_eval_episodes']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    else:
        assert inspect.signature(cls) == inspect.signature(parent)


@pytest.mark.parametrize('phase', ['online', 'offline', 'off2on'])
def test_registered_algorithms_start_with_no_periodic_checkpoint(phase):
    registry = importlib.import_module(f'rl_garden.training.{phase}.registry').registry
    registry.discover()
    for name, entry in registry.entries().items():
        if entry.algorithm_cls is not None:
            cls = entry.algorithm_cls()
            assert cls._last_checkpoint_step == -1, (phase, name)


@pytest.mark.parametrize('save_final', [True, False])
@pytest.mark.parametrize('family', ['sac', 'ppo', 'dagger'])
def test_final_checkpoint_write_count_at_periodic_boundary(tmp_path, monkeypatch, family, save_final):
    if family == 'sac':
        from rl_garden.algorithms.sac import SAC
        from test_checkpoint import _sac_kwargs, _state_step_env
        agent = SAC(env=_state_step_env(), **_sac_kwargs())
    elif family == 'ppo':
        from rl_garden.algorithms.ppo import PPO
        from test_ppo import DummyVecEnv, _action_space, _ppo_kwargs, _state_space
        agent = PPO(env=DummyVecEnv(_state_space(), _action_space()), **_ppo_kwargs())
    else:
        from test_dagger import _make_agent
        agent = _make_agent(rollout_steps_per_round=1, gradient_steps_per_round=1, std_log=False)
    agent.checkpoint_dir = str(tmp_path)
    agent.checkpoint_freq = 1
    agent.save_final_checkpoint = save_final
    writes = []
    original = agent._save_checkpoint

    def save(name):
        writes.append(name)
        original(name)

    monkeypatch.setattr(agent, '_save_checkpoint', save)
    agent.learn(4)
    counts = Counter(writes)
    assert counts['final.pt'] == int(save_final)
    assert (tmp_path / 'final.pt').exists() is save_final
    assert counts[f'checkpoint_{agent._global_step}.pt'] == 1
    assert all(count == 1 for count in counts.values())
