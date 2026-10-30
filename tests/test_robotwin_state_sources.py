"""State-provider tests use NumPy only, without importing optional simulators."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "state_sources_under_test",
    Path(__file__).parents[1] / "rl_garden/envs/robotwin/state_sources.py",
)
sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sources)


def task_fixture():
    # Scrambled semantic order detects hard-coded qpos[:6] and duplicate entities.
    robot = NS()
    for side, offset in (("left", 0), ("right", 10)):
        joints = [object() for _ in range(10)]
        q = np.arange(10, dtype=float) + offset
        entity = NS(get_active_joints=lambda j=joints: j,
                    get_qpos=lambda q=q: q,
                    get_qvel=lambda q=q: q * .1)
        setattr(robot, side + "_entity", entity)
        setattr(robot, side + "_arm_joints", [joints[i] for i in (7, 1, 9, 3, 2, 0)])
        setattr(robot, side + "_gripper", [(joints[4],), (joints[6],)])
        setattr(robot, "get_" + side + "_tcp_pose", lambda: [1, 2, 3, 1, 0, 0, 0])
        setattr(robot, "get_" + side + "_ee_pose", lambda: [1, 2, 2.88, 1, 0, 0, 0])
    laptop = NS(get_pose=lambda: NS(p=[0, 0, .7], q=[1, 0, 0, 0]),
                get_qpos=lambda: [.3], get_qvel=lambda: [.1],
                get_qlimits=lambda: [[0, 1.5]],
                get_contact_point=lambda i: [i, 0, 1, 1, 0, 0, 0])
    return NS(robot=robot, laptop=laptop, delta_ee_command_reference=True,
              _delta_ee_left_command_pose=None,
              _delta_ee_right_command_pose=np.array([2, 3, 4, -1, 0, 0, 0.]))


def test_dimensions_and_semantic_order():
    out = sources.read_sources(task_fixture(), sources.V1_SOURCES)
    assert {k: sources.SOURCE_DIMS[k] for k in sources.V1_SOURCES} == {"robot_actual_v1": 32, "open_laptop_v1": 25, "delta_ee_v1": 30}
    for name in sources.V1_SOURCES:
        dim = sources.SOURCE_DIMS[name]
        assert out["state_" + name].shape == (dim,)
        assert out["state_" + name].dtype == np.float32
    np.testing.assert_array_equal(out["state_robot_actual_v1"][:8], [7, 1, 9, 3, 2, 0, 4, 6])


def test_read_only_rng_and_command_validity():
    task = task_fixture()
    before = task._delta_ee_right_command_pose.copy()
    rng = np.random.get_state()
    a = sources.read_sources(task, sources.V1_SOURCES)
    b = sources.read_sources(task, sources.V1_SOURCES)
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
    assert rng[2:] == np.random.get_state()[2:]
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(before, task._delta_ee_right_command_pose)
    assert a["state_delta_ee_v1"][14] == 0
    assert a["state_delta_ee_v1"][29] == 1
    assert a["state_delta_ee_v1"][25] == 1  # Canonicalized command qw.


@pytest.mark.parametrize("names,task,control", [
    (["unknown"], "open_laptop", "delta_ee"),
    (["open_laptop_v1"], "place_shoe", "delta_ee"),
    (["delta_ee_v1"], "open_laptop", "joint_pos"),
    (["robot_actual_v1"] * 2, "open_laptop", "delta_ee"),
    ("robot_actual_v1", "open_laptop", "delta_ee"),
])
def test_invalid_requests(names, task, control):
    with pytest.raises(ValueError):
        sources.validate_sources(names, task, control)


def test_missing_and_nonfinite_are_errors():
    task = task_fixture()
    task.laptop.get_qpos = lambda: [np.nan]
    with pytest.raises(ValueError):
        sources.read_sources(task, ["open_laptop_v1"])
    del task._delta_ee_left_command_pose
    with pytest.raises(AttributeError):
        sources.read_sources(task, ["delta_ee_v1"])


def test_empty_request_does_not_touch_runtime():
    assert sources.read_sources(object(), ()) == {}
    sources.validate_sources((), "arbitrary_task", "joint_pos")


def test_joint_schema_rejected():
    task = task_fixture()
    task.robot.left_gripper = []
    with pytest.raises(ValueError):
        sources.read_sources(task, ["robot_actual_v1"])
