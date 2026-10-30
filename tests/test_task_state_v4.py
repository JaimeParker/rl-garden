"""Pure CPU tests for semantic geometry/contact observations."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

package = types.ModuleType('_task_state_v4_test')
package.__path__ = [str(Path(__file__).parents[1] / 'rl_garden/envs/robotwin')]
sys.modules[package.__name__] = package
v4 = importlib.import_module(package.__name__ + '.task_state_v4')
sources = importlib.import_module(package.__name__ + '.state_sources')


def body(i):
    return NS(entity=NS(get_per_scene_id=lambda: i))


def contact(a, b, separation=0., impulse=(0, 0, .001)):
    return NS(bodies=(body(a), body(b)), points=[NS(separation=separation, impulse=impulse)])


def test_contact_pairs_order_filtering_and_thresholds():
    fingers, parts = {10: 0, 11: 1}, {20: 0, 21: 1}
    assert v4.contact_flags([contact(10, 20), contact(21, 10), contact(11, 20), contact(21, 11)], fingers, parts).tolist() == [1, 1, 1, 1]
    assert not v4.contact_flags([contact(10, 20, separation=.001), contact(10, 20, impulse=(0, 0, 0)), contact(99, 20)], fingers, parts).any()
    with pytest.raises(ValueError, match='Nonfinite'):
        v4.contact_flags([contact(10, 20, separation=float('nan'))], fingers, parts)


class Scene:
    def __init__(self):
        self.ticks = 0
    def step(self):
        self.ticks += 1
        return 123
    def get_contacts(self):
        return [contact(10, 21)] if self.ticks == 1 else []


def test_action_window_keeps_transient_contact_and_restores_method():
    scene = Scene()
    task = NS(scene=scene, _rlg_contact_groups_v4=({10: 0}, {21: 1}))
    rng = np.random.get_state()
    with v4.observe_control_contacts(task):
        assert scene.step() == 123
        scene.step()
    assert scene.ticks == task._rlg_contact_ticks_v4 == 2
    assert 'step' not in vars(scene)
    assert task._rlg_contact_flags_v4.tolist() == [0, 1, 0, 0]
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
    with v4.observe_control_contacts(task):
        scene.step()
    assert not task._rlg_contact_flags_v4.any()  # previous action is not sticky


def test_zero_tick_and_exception_restore():
    scene = Scene()
    scene.ticks = 1
    original = lambda: 456
    scene.step = original
    task = NS(scene=scene, _rlg_contact_groups_v4=({10: 0}, {21: 1}))
    with v4.observe_control_contacts(task):
        pass
    assert scene.step is original and task._rlg_contact_ticks_v4 == 0
    assert task._rlg_contact_flags_v4[1] == 1
    with pytest.raises(RuntimeError):
        with v4.observe_control_contacts(task):
            raise RuntimeError('simulator failure')
    assert scene.step is original


def test_relative_coordinates_and_vector_order(monkeypatch):
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    parent = NS(get_entity_pose=lambda: NS(to_transformation_matrix=lambda: matrix))
    monkeypatch.setattr(v4, '_resolve_laptop', lambda task: (None, parent, None, None))
    task = NS(robot=NS(get_left_tcp_pose=lambda: [1, 2, 3, 1, 0, 0, 0],
                       get_right_tcp_pose=lambda: [0, 0, 0, 1, 0, 0, 0]),
              laptop=NS(get_contact_point=lambda i: [[1, 3, 3, 1, 0, 0, 0], [2, 2, 3, 1, 0, 0, 0]][i]))
    np.testing.assert_allclose(v4.relative_points(task), [1, 0, 0, 0, -1, 0, 3, -1, 3, 2, -2, 3])
    monkeypatch.setattr(v4, 'read_task_state_v3', lambda task: np.arange(106))
    task._rlg_geometry_centres_v4 = np.arange(6)
    task._rlg_contact_flags_v4 = np.array([1, 0, 0, 1])
    out = v4.read_task_state_v4(task)
    assert out.shape == (128,)
    np.testing.assert_equal(out[106:112], np.arange(6))
    np.testing.assert_equal(out[-4:], [1, 0, 0, 1])
    assert sources.SOURCE_DIMS['open_laptop_robot_v4'] == 128
    with pytest.raises(ValueError):
        sources.validate_sources(['open_laptop_robot_v3', 'open_laptop_robot_v4'], 'open_laptop', 'delta_ee')


def test_real_physx_impulse_contact():
    # No renderer/GPU: verify against actual PhysX contact objects when installed.
    sapien = pytest.importorskip('sapien')
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    ground_builder = scene.create_actor_builder()
    ground_builder.add_box_collision(half_size=[.2, .2, .01])
    ground = ground_builder.build_static(name='ground')
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.02, .02, .02])
    cube = builder.build(name='cube')
    cube.set_pose(sapien.Pose([0, 0, .04]))
    seen = False
    for _ in range(100):
        scene.step()
        flags = v4.contact_flags(scene.get_contacts(), {cube.get_per_scene_id(): 0}, {ground.get_per_scene_id(): 1})
        seen |= bool(flags[1])
    assert seen
