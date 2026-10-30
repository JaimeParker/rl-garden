"""142D task observations: geometry centres, relative points, contact history.

Contact bits summarize the preceding control action, not a current grasp label.
Only this scene's step method is temporarily observed; no extra physics steps.
"""
from contextlib import contextmanager

import numpy as np

from .laptop_dimensions import _matrix, prepare_laptop_dimensions, read_task_state_v3
from .state_sources import _vector
from .task_state_v2 import _chain, _resolve_laptop

CONTACT_MAX_SEPARATION_M = 1e-5
CONTACT_MIN_IMPULSE_NS = 1e-8


def _entity_id(body):
    return int(body.entity.get_per_scene_id())


def contact_groups(task):
    """Return numeric scene-local IDs only; never cache physics handles."""
    hinge, _, _, _ = _resolve_laptop(task)
    parts = {}
    for link in task.laptop.actor.get_links():
        moving = any(j.get_name() == hinge.get_name() for _, j in _chain(link))
        parts[_entity_id(link)] = int(moving)
    fingers, names = {}, {}
    for side_index, side in enumerate(('left', 'right')):
        joints = [item[0] for item in getattr(task.robot, side + '_gripper') if item is not None]
        if len(joints) != 2:
            raise ValueError('v4 requires two finger joints per gripper')
        roots = {j.get_name() for j in joints}
        selected = []
        for link in getattr(task.robot, side + '_entity').get_links():
            path = _chain(link)
            if any(j.get_name() in roots for _, j in path):
                idx = _entity_id(link)
                if idx in fingers and fingers[idx] != side_index:
                    raise ValueError('Left/right finger contact groups overlap')
                fingers[idx] = side_index
                selected.append(link.get_name())
        if not selected or any(_entity_id(j.get_child_link()) not in fingers for j in joints):
            raise ValueError('Missing finger contact links')
        names[side] = selected
    if set(parts) & set(fingers):
        raise ValueError('Robot/laptop contact groups overlap')
    if set(parts.values()) != {0, 1}:
        raise ValueError('Missing fixed/moving laptop contact group')
    return fingers, parts, names


def contact_flags(contacts, fingers, parts):
    """Left-base, left-lid, right-base, right-lid; independent of pair order."""
    result = np.zeros(4, dtype=np.float32)
    for contact in contacts:
        a, b = (_entity_id(body) for body in contact.bodies)
        if a in parts and b in fingers:
            a, b = b, a
        if a not in fingers or b not in parts:
            continue
        for point in contact.points:
            separation = float(point.separation)
            impulse = _vector(point.impulse, 3, 'contact impulse')
            if not np.isfinite(separation):
                raise ValueError('Nonfinite contact separation')
            if separation <= CONTACT_MAX_SEPARATION_M and np.linalg.norm(impulse) > CONTACT_MIN_IMPULSE_NS:
                result[2 * fingers[a] + parts[b]] = 1.
                break
    return result


def prepare_task_state_v4(task):
    prepare_laptop_dimensions(task)
    centres = np.asarray(task._rlg_laptop_dimensions_v3_metadata['local_centres_audit_only'], dtype=np.float32)
    task._rlg_geometry_centres_v4 = _vector(centres.reshape(-1), 6, 'geometry centres').astype(np.float32)
    task._rlg_geometry_centres_v4.setflags(write=False)
    fingers, parts, names = contact_groups(task)
    task._rlg_contact_groups_v4 = (fingers, parts)
    task._rlg_contact_metadata_v4 = {'finger_links': names,
        'max_separation_m': CONTACT_MAX_SEPARATION_M, 'min_impulse_Ns': CONTACT_MIN_IMPULSE_NS,
        'window': 'previous_control_action_any_tick; reset=current_snapshot'}
    task._rlg_contact_flags_v4 = contact_flags(task.scene.get_contacts(), fingers, parts)
    task._rlg_contact_ticks_v4 = 0


@contextmanager
def observe_control_contacts(task, *, capture_last=False):
    """Temporary instance-local step observer, restored even on native errors."""
    fingers, parts = task._rlg_contact_groups_v4
    scene = task.scene
    original = scene.step
    had_override = 'step' in vars(scene)
    previous_override = vars(scene).get('step')
    flags = np.zeros(4, dtype=np.float32)
    last = np.zeros(4, dtype=np.float32)
    ticks = 0

    def step(*args, **kwargs):
        nonlocal ticks
        result = original(*args, **kwargs)
        ticks += 1
        last[:] = contact_flags(scene.get_contacts(), fingers, parts)
        np.maximum(flags, last, out=flags)
        return result

    scene.step = step
    try:
        yield
        if not ticks:
            # A no-op/failed planner can produce no physics ticks: observe now.
            flags[:] = contact_flags(scene.get_contacts(), fingers, parts)
            last[:] = flags
        task._rlg_contact_flags_v4 = flags.copy()
        task._rlg_contact_ticks_v4 = ticks
        if capture_last:
            task._rlg_contact_last_flags_v5 = last.copy()
    finally:
        if had_override:
            scene.step = previous_override
        else:
            del scene.step


def relative_points(task):
    _, parent, _, _ = _resolve_laptop(task)
    rotation = _matrix(parent.get_entity_pose())[:3, :3]
    values = []
    for side in ('left', 'right'):
        tcp = _vector(getattr(task.robot, 'get_' + side + '_tcp_pose')(), 7, 'TCP')[:3]
        for index in (0, 1):
            point = _vector(task.laptop.get_contact_point(index), 7, 'operation point')[:3]
            values.extend(rotation.T @ (point - tcp))
    return _vector(values, 12, 'relative points').astype(np.float32)


def read_task_state_v4(task):
    flags = _vector(task._rlg_contact_flags_v4, 4, 'contact flags')
    if not np.isin(flags, (0., 1.)).all():
        raise ValueError('Contact flags must be binary')
    return np.concatenate((read_task_state_v3(task), task._rlg_geometry_centres_v4,
                           relative_points(task), flags))
