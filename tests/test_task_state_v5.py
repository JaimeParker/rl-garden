"""V5 contact windows: no extra ticks, reads, RNG draws or stale reset flags."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

package = types.ModuleType('_task_state_v5_test')
package.__path__ = [str(Path(__file__).parents[1] / 'rl_garden/envs/robotwin')]
sys.modules[package.__name__] = package
v4 = importlib.import_module(package.__name__ + '.task_state_v4')
v5 = importlib.import_module(package.__name__ + '.task_state_v5')
sources = importlib.import_module(package.__name__ + '.state_sources')


class Scene:
    def __init__(self, sequence):
        self.sequence, self.ticks, self.reads = sequence, 0, 0

    def step(self):
        self.ticks += 1
        return self.ticks

    def get_contacts(self):
        self.reads += 1
        return np.asarray(self.sequence[self.ticks], dtype=np.float32)


def task_for(monkeypatch, sequence):
    monkeypatch.setattr(v4, 'contact_flags', lambda contacts, *args: contacts.copy())
    return NS(scene=Scene(sequence), _rlg_contact_groups_v4=({}, {}))


@pytest.mark.parametrize('sequence,any_flags,last_flags', [
    ([[0]*4, [0, 1, 0, 0], [0]*4], [0, 1, 0, 0], [0]*4),
    ([[0]*4, [0]*4, [1, 0, 0, 1]], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([[0]*4, [0]*4, [0]*4], [0]*4, [0]*4),
])
def test_last_tick_and_any_tick(monkeypatch, sequence, any_flags, last_flags):
    task = task_for(monkeypatch, sequence)
    rng = np.random.get_state()
    with v4.observe_control_contacts(task, capture_last=True):
        assert task.scene.step() == 1
        assert task.scene.step() == 2
    assert task.scene.ticks == task.scene.reads == task._rlg_contact_ticks_v4 == 2
    assert 'step' not in vars(task.scene)
    np.testing.assert_equal(task._rlg_contact_flags_v4, any_flags)
    np.testing.assert_equal(task._rlg_contact_last_flags_v5, last_flags)
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])


def test_zero_tick_reset_and_no_aliasing(monkeypatch):
    task = task_for(monkeypatch, [[0, 0, 1, 0], [0]*4])
    monkeypatch.setattr(v5, 'prepare_task_state_v4', lambda t: setattr(
        t, '_rlg_contact_flags_v4', t.scene.get_contacts()))
    task._rlg_contact_last_flags_v5 = np.ones(4)
    v5.prepare_task_state_v5(task)
    np.testing.assert_equal(task._rlg_contact_last_flags_v5, [0, 0, 1, 0])
    assert not np.shares_memory(task._rlg_contact_flags_v4, task._rlg_contact_last_flags_v5)
    with v4.observe_control_contacts(task, capture_last=True):
        pass
    np.testing.assert_equal(task._rlg_contact_last_flags_v5, [0, 0, 1, 0])
    assert task._rlg_contact_ticks_v4 == 0
    with v4.observe_control_contacts(task, capture_last=True):
        task.scene.step()
    assert not task._rlg_contact_last_flags_v5.any()
    assert not task._rlg_contact_flags_v4.any()


def test_exception_restores_method_and_does_not_publish_partial_state(monkeypatch):
    task = task_for(monkeypatch, [[0]*4, [1]*4])
    task._rlg_contact_last_flags_v5 = np.zeros(4)
    with pytest.raises(RuntimeError):
        with v4.observe_control_contacts(task, capture_last=True):
            task.scene.step()
            raise RuntimeError('failure')
    assert 'step' not in vars(task.scene)
    assert not task._rlg_contact_last_flags_v5.any()


def test_schema_and_getter(monkeypatch):
    monkeypatch.setattr(v5, 'read_task_state_v4', lambda task: np.arange(128))
    task = NS(_rlg_contact_last_flags_v5=np.array([0, 1, 1, 0]))
    a = sources.read_sources(task, ['open_laptop_robot_v5'])['state_open_laptop_robot_v5']
    assert a.dtype == np.float32 and a.shape == (132,)
    np.testing.assert_equal(a[:128], np.arange(128))
    np.testing.assert_equal(a[-4:], [0, 1, 1, 0])
    assert sources.SOURCE_FIELDS['open_laptop_robot_v5'][:128] == sources.SOURCE_FIELDS['open_laptop_robot_v4']
    for names, task_name, mode in [(['open_laptop_robot_v5'], 'other', 'delta_ee'),
        (['open_laptop_robot_v5'], 'open_laptop', 'joint'),
        (['open_laptop_robot_v4', 'open_laptop_robot_v5'], 'open_laptop', 'delta_ee')]:
        with pytest.raises(ValueError):
            sources.validate_sources(names, task_name, mode)
    task._rlg_contact_last_flags_v5[0] = 2
    with pytest.raises(ValueError, match='binary'):
        v5.read_task_state_v5(task)
