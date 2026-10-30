"""Physical scaling: semantics, RNG, immutable inputs and checkpoint isolation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.scaled_task_state import KEYS, ScaledTaskStateExtractor, scaling_spec
from rl_garden.algorithms.scaled_task_state_residual import ScaledTaskStateResidualSACV5
from rl_garden.algorithms.task_state_residual import TaskStateResidualSACV5
from rl_garden.policies.base_policies import BasePolicyOutput, BasePolicyProvider


def observation_space():
    return spaces.Dict({k: spaces.Box(-np.inf, np.inf, (n,), np.float32)
                        for k, n in zip(KEYS, (14, 132))})


def test_scaling_field_units_and_coverage():
    fields, divisors, units = scaling_spec()
    spec = dict(zip(fields, zip(divisors, units)))
    assert len(spec) == 146 and all(d > 0 for d in divisors)
    assert spec['command_state.left.arm_0'] == (np.pi, 'rad')
    assert spec['command_state.right.gripper'] == (1., 'dimensionless')
    assert spec['left.position.finger_0'] == (.04765, 'm')
    assert spec['right.velocity.finger_1'] == (.1, 'm/s')
    assert spec['hinge.velocity'] == (np.pi, 'rad/s')
    assert spec['hinge.origin.z'] == (1., 'm')
    assert spec['hinge.axis.x'] == (1., 'dimensionless')
    assert spec['left.to_contact_0.x_parent_frame'] == (.25, 'm')
    assert spec['hinge_child.extent_z'] == (.25, 'm')
    assert sum(u == 'binary' for u in units) == 10


def test_forward_no_mutation_rng_clipping_or_running_statistics():
    rng = torch.get_rng_state().clone()
    extractor = ScaledTaskStateExtractor(observation_space())
    assert torch.equal(rng, torch.get_rng_state())
    assert list(extractor.parameters()) == []
    raw = (extractor.state_divisor * 3).repeat(2, 1).requires_grad_()
    obs = {KEYS[0]: raw[:, :14], KEYS[1]: raw[:, 14:]}
    before = raw.detach().clone()
    out = extractor(obs)
    torch.testing.assert_close(out, torch.full((2, 146), 3.))
    assert torch.equal(before, raw.detach())
    out.sum().backward()
    torch.testing.assert_close(raw.grad, (1/extractor.state_divisor).repeat(2, 1))
    extractor.eval()
    torch.testing.assert_close(extractor(obs), out)
    assert torch.equal(rng, torch.get_rng_state())


def test_shape_guard():
    space = observation_space()
    space[KEYS[1]] = spaces.Box(-1, 1, (128,), np.float32)
    with pytest.raises(ValueError, match='14\\+132'):
        ScaledTaskStateExtractor(space)


class ZeroBase(BasePolicyProvider):
    def select_action(self, obs):
        return BasePolicyOutput(actions=obs['state'].new_zeros((len(obs['state']), 14)))

    def reset(self, env_ids=None):
        pass


def make_agent(cls):
    act = spaces.Box(-1, 1, (14,), np.float32)
    env = SimpleNamespace(num_envs=1, single_observation_space=observation_space(),
                          single_action_space=act,
                          action_space=spaces.Box(-1, 1, (1, 14), np.float32))
    return cls(env=env, base_action_provider=ZeroBase(env.single_observation_space, act),
               device='cpu', buffer_device='cpu', buffer_size=8, batch_size=2,
               net_arch=[16], critic_impl='legacy', bootstrap_at_done='truncated')


def test_same_initial_parameters_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(13)
    native = make_agent(TaskStateResidualSACV5)
    torch.manual_seed(13)
    scaled = make_agent(ScaledTaskStateResidualSACV5)
    for (n, p), (m, q) in zip(native.policy.named_parameters(), scaled.policy.named_parameters()):
        assert n == m
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    path = tmp_path / 'scaled.pt'
    scaled.save(path)
    restored = make_agent(ScaledTaskStateResidualSACV5)
    restored.load(path, load_replay_buffer=False)
    for key, value in scaled.policy.state_dict().items():
        torch.testing.assert_close(value, restored.policy.state_dict()[key], rtol=0, atol=0)
    assert restored._checkpoint_metadata()['selected_state_schema']['normalization']['clipping'] is False
    with pytest.raises(ValueError):
        native.load(path, load_replay_buffer=False)
    path = tmp_path / 'native.pt'
    native.save(path)
    with pytest.raises(ValueError):
        restored.load(path, load_replay_buffer=False)


def test_registry_keeps_native_config_unchanged():
    from rl_garden.training.online._registry import registry
    from rl_garden.training.online.task_state_residual_sac_v5 import TaskStateResidualSACV5Args
    from rl_garden.training.online.task_state_residual_sac_v5_scaled import ScaledTaskStateResidualSACV5Args
    registry.discover()
    args = ScaledTaskStateResidualSACV5Args()
    assert registry.entry_for_args(args)[0] == 'task_state_residual_sac_v5_scaled'
    assert args.bootstrap_at_done == 'truncated'
    assert not hasattr(TaskStateResidualSACV5Args(), 'state_scaling')


@pytest.mark.parametrize('embodiment', ['aloha-agilex', ('aloha-agilex',), ['aloha-agilex']])
def test_builder_accepts_serialized_embodiment(monkeypatch, embodiment):
    from rl_garden.training.online.task_state_residual_sac_v5_scaled import ScaledTaskStateResidualSACV5Args, build_scaled_task_state
    from rl_garden.training.online import residual_sac
    args = ScaledTaskStateResidualSACV5Args()
    args.robotwin.embodiment = embodiment
    monkeypatch.setattr(residual_sac, 'build_residual_sac', lambda *a, **kw: kw['algorithm_class'])
    assert build_scaled_task_state(args, None, None, None, None) is ScaledTaskStateResidualSACV5
