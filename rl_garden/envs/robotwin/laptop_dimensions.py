"""Rigid-part collision extents in hinge-parent/child local axes, in metres.

Computed once at reset; no geometry handles are retained and no RNG or stepping
is used. These are collision-shape AABB extents, not visual glass-panel dimensions.
"""
import itertools
import numpy as np

from .task_state_v2 import _chain, _resolve_laptop


def _matrix(pose):
    m = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("Invalid geometry pose")
    return m


def _shape_vertices(shape):
    kind = type(shape).__name__
    if kind in ("PhysxCollisionShapeConvexMesh", "PhysxCollisionShapeTriangleMesh"):
        vertices = np.asarray(shape.get_vertices(), dtype=np.float64)
        scale = np.asarray(shape.get_scale(), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("Missing collision mesh vertices")
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("Invalid collision mesh scale")
        vertices = vertices * scale
    elif kind == "PhysxCollisionShapeBox":
        half = np.asarray(shape.get_half_size(), dtype=np.float64)
        if half.shape != (3,) or np.any(half <= 0):
            raise ValueError("Invalid collision box")
        vertices = np.asarray(list(itertools.product((-1., 1.), repeat=3))) * half
    else:
        raise ValueError(f"Unsupported laptop collision shape: {kind}")
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite collision vertices")
    return vertices


def measure_laptop_dimensions(task):
    hinge, parent, child, _ = _resolve_laptop(task)
    anchors = [np.linalg.inv(_matrix(link.get_entity_pose())) for link in (parent, child)]
    points = [[], []]
    counts = [0, 0]
    for link in task.laptop.actor.get_links():
        moving = any(j.get_name() == hinge.get_name() for _, j in _chain(link))
        index = int(moving)
        for shape in link.get_collision_shapes():
            transform = anchors[index] @ _matrix(link.get_entity_pose()) @ _matrix(shape.get_local_pose())
            vertices = _shape_vertices(shape)
            points[index].append(vertices @ transform[:3, :3].T + transform[:3, 3])
            counts[index] += 1
    dimensions, centres = [], []
    for group in points:
        if not group:
            raise ValueError("Laptop fixed/moving part has no collision geometry")
        vertices = np.concatenate(group)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        extent = hi - lo
        if not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError("Degenerate laptop collision bounds")
        dimensions.extend(extent.tolist())
        centres.append(((lo + hi) / 2).tolist())
    return np.asarray(dimensions, dtype=np.float32), {
        "units": "m", "frame": "hinge_parent_then_child_local_xyz",
        "geometry": "collision_aabb", "shape_counts": counts,
        "local_centres_audit_only": centres,
    }


def prepare_laptop_dimensions(task):
    dimensions, metadata = measure_laptop_dimensions(task)
    dimensions.setflags(write=False)
    # Owned by the freshly reset task, numeric data only; no mesh/link handles.
    task._rlg_laptop_dimensions_v3 = dimensions
    task._rlg_laptop_dimensions_v3_metadata = metadata


def read_task_state_v3(task):
    from .task_state_v2 import read_task_state_v2

    if not hasattr(task, "_rlg_laptop_dimensions_v3"):
        raise ValueError("State v3 dimensions must be prepared during reset")
    return np.concatenate((read_task_state_v2(task), task._rlg_laptop_dimensions_v3))
