"""Analytic unit/scale/frame tests without a simulator dependency."""
import importlib
import itertools
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

package = types.ModuleType('_laptop_dimensions_test')
package.__path__ = [str(Path(__file__).parents[1] / 'rl_garden/envs/robotwin')]
sys.modules[package.__name__] = package
geometry = importlib.import_module(package.__name__ + '.laptop_dimensions')
sources = importlib.import_module(package.__name__ + '.state_sources')


class Pose:
    def __init__(self, p=(0, 0, 0), angle=0):
        c, s = np.cos(angle), np.sin(angle)
        self.m = np.eye(4)
        self.m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        self.m[:3, 3] = p

    def to_transformation_matrix(self):
        return self.m


class PhysxCollisionShapeConvexMesh:
    def get_vertices(self):
        return np.array(list(itertools.product((-1., 1.), repeat=3)))

    def get_scale(self):
        return np.array([.1, .2, .03])

    def get_local_pose(self):
        return Pose((.4, .2, 0))


class PhysxCollisionShapeBox:
    def get_half_size(self):
        return np.array([.3, .2, .01])

    def get_local_pose(self):
        return Pose()


def test_scale_applied_once_and_box_half_lengths():
    np.testing.assert_allclose(np.ptp(geometry._shape_vertices(PhysxCollisionShapeConvexMesh()), axis=0), [.2, .4, .06])
    np.testing.assert_allclose(np.ptp(geometry._shape_vertices(PhysxCollisionShapeBox()), axis=0), [.6, .4, .02])
    with pytest.raises(ValueError, match='Unsupported'):
        geometry._shape_vertices(object())


@pytest.mark.parametrize('angle', [0., .4, 1.2])
def test_local_part_dimensions_and_reset_cache(monkeypatch, angle):
    hinge = NS(get_name=lambda: 'hinge')
    parent = NS(get_entity_pose=lambda: Pose((1, 2, 3), angle), moving=False,
                get_collision_shapes=lambda: [PhysxCollisionShapeBox()])
    child = NS(get_entity_pose=lambda: Pose((2, 3, 4), angle + .7), moving=True,
               get_collision_shapes=lambda: [PhysxCollisionShapeConvexMesh()])
    task = NS(laptop=NS(actor=NS(get_links=lambda: [parent, child])))
    monkeypatch.setattr(geometry, '_resolve_laptop', lambda task: (hinge, parent, child, []))
    monkeypatch.setattr(geometry, '_chain', lambda link: [(link, hinge)] if link.moving else [])
    dimensions, meta = geometry.measure_laptop_dimensions(task)
    np.testing.assert_allclose(dimensions, [.6, .4, .02, .2, .4, .06], atol=1e-6)
    assert meta['shape_counts'] == [1, 1]
    geometry.prepare_laptop_dimensions(task)
    assert not task._rlg_laptop_dimensions_v3.flags.writeable
    np.testing.assert_allclose(task._rlg_laptop_dimensions_v3, dimensions)
    assert sources.SOURCE_DIMS['open_laptop_robot_v2'] == 100
    assert sources.SOURCE_DIMS['open_laptop_robot_v3'] == 106


def test_v3_no_unprepared_or_mixed_sources():
    with pytest.raises(ValueError, match='prepared'):
        geometry.read_task_state_v3(NS())
    with pytest.raises(ValueError):
        sources.validate_sources(('open_laptop_robot_v2', 'open_laptop_robot_v3'), 'open_laptop', 'delta_ee')
