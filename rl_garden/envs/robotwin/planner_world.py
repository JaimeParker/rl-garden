"""Validation candidate: lossless supported FCL geometry + native world state.

No SAPIEN scene or CUDA object crosses IPC. Unsupported data fails closed.
This module does NOT install itself in a training process.
"""
import numpy as np
import mplib
from mplib.collision_detection import AllowedCollision
from mplib.collision_detection import fcl


def pose_out(p):
    return {'p': np.asarray(p.p).copy(), 'q': np.asarray(p.q).copy()}


def pose_in(p):
    return mplib.Pose(p['p'], p['q'])


def geometry_out(g):
    name = type(g).__name__
    fields = {'Box': ('side',), 'Capsule': ('radius', 'lz'),
              'Cylinder': ('radius', 'lz'), 'Sphere': ('radius',),
              'Halfspace': ('n', 'd')}
    if name in fields:
        values = {k: np.asarray(getattr(g, k)).copy() for k in fields[name]}
    elif name in ('Convex', 'BVHModel'):
        values = {'vertices': np.asarray(g.get_vertices()).copy(),
                  'faces': np.asarray(g.get_faces()).copy()}
        if name == 'Convex':
            values['num_faces'] = g.get_face_count()
    else:
        raise TypeError('Unsupported FCL geometry: '+name)
    return {'type': name, 'values': values, 'cost_density': g.cost_density}


def geometry_in(spec):
    values = spec['values']
    if spec['type'] == 'BVHModel':
        g = fcl.BVHModel()
        g.begin_model()
        g.add_sub_model(**values)
        g.end_model()
    else:
        if spec['type'] not in ('Box', 'Capsule', 'Cylinder', 'Sphere', 'Halfspace', 'Convex'):
            raise TypeError(spec['type'])
        if spec['type'] == 'Convex':
            faces = values['faces']
            if faces.size != values['num_faces'] * 4 or not np.all(faces.reshape(-1, 4)[:, 0] == 3):
                raise ValueError('Non-triangular convex faces are not supported by installed binding')
            g = fcl.Convex(values['vertices'], np.ascontiguousarray(faces.reshape(-1, 4)[:, 1:], dtype=np.int32))
        else:
            g = getattr(fcl, spec['type'])(**values)
    g.cost_density = spec['cost_density']
    return g


def object_out(obj):
    return {'name': obj.name, 'pose': pose_out(obj.pose),
            'shapes': [geometry_out(s.get_collision_geometry()) for s in obj.shapes],
            'shape_poses': [pose_out(p) for p in obj.shape_poses]}


def object_in(spec):
    return fcl.FCLObject(spec['name'], pose_in(spec['pose']),
                         [fcl.CollisionObject(geometry_in(s)) for s in spec['shapes']],
                         [pose_in(p) for p in spec['shape_poses']])


def collision_bool(value):
    if value == AllowedCollision.ALWAYS:
        return True
    if value == AllowedCollision.NEVER:
        return False
    raise ValueError('Conditional collision rules cannot be serialized safely')


def acm_out(acm):
    names = sorted(acm.get_all_entry_names())
    entries, defaults = [], []
    for i, a in enumerate(names):
        val = acm.get_default_entry(a)
        if val is not None:
            defaults.append((a, collision_bool(val)))
        for b in names[i:]:
            val = acm.get_entry(a, b)
            if val is not None:
                entries.append((a, b, collision_bool(val)))
    return {'names': names, 'entries': entries, 'defaults': defaults}


def acm_in(acm, spec):
    acm.clear()
    for a, b, value in spec['entries']:
        acm.set_entry(a, b, value)
    for a, value in spec['defaults']:
        acm.set_default_entry(a, value)
    if acm_out(acm) != spec:
        raise ValueError('Collision matrix did not round-trip')


def capture(planner):
    """Capture the actual planning world; do NOT update it from simulation."""
    from mplib.sapien_utils.conversion import convert_object_name
    from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
    from mplib.sapien_utils.srdf_exporter import export_srdf
    world = planner.planning_world
    scene = world._sim_scene
    sim_arts = {convert_object_name(a): a for a in scene.get_all_articulations()}
    names = list(world.get_articulation_names())
    if not set(names) <= set(sim_arts):
        raise ValueError('Planning articulation has no corresponding kinematic source')
    arts = []
    for name in names:
        native, sim = world.get_articulation(name), sim_arts[name]
        model = native.get_fcl_model()
        arts.append({'name': name, 'urdf': export_kinematic_chain_urdf(sim),
                     'srdf': export_srdf(sim),
                     'gravity': np.asarray(scene.get_physx_system().config.gravity).copy(),
                     'links': list(native.get_user_link_names()),
                     'joints': list(native.get_user_joint_names()),
                     'collision_links': [object_out(o) for o in model.get_collision_objects()],
                     'collision_pairs': model.get_collision_pairs(),
                     'base': pose_out(native.get_base_pose()), 'qpos': native.get_qpos().copy(),
                     'planned': world.is_articulation_planned(name),
                     'move_group': list(native.get_move_group_end_effectors())})
    objects = []
    for name in world.get_object_names():
        if world.is_object_attached(name):
            raise ValueError('Attached object unsupported: '+name)
        objects.append(object_out(world.get_object(name)))
    return {'arts': arts, 'objects': objects,
            'scene_articulations_not_in_original_world': sorted(set(sim_arts) - set(names)),
            'acm': acm_out(world.get_allowed_collision_matrix()),
            'move_group': planner.move_group,
            'joint_vel_limits': np.asarray(planner.joint_vel_limits).copy(),
            'joint_acc_limits': np.asarray(planner.joint_acc_limits).copy()}


def rebuild(spec):
    from mplib.sapien_utils import SapienPlanner
    world = mplib.PlanningWorld([])
    for a in spec['arts']:
        art = mplib.ArticulatedModel.create_from_urdf_string(
            a['urdf'], a['srdf'], [object_in(o) for o in a['collision_links']],
            name=a['name'], gravity=a['gravity'], link_names=a['links'], joint_names=a['joints'])
        art.set_base_pose(pose_in(a['base']))
        art.set_qpos(a['qpos'], full=True)
        if a['move_group']:
            art.set_move_group(a['move_group'])
        if art.get_fcl_model().get_collision_pairs() != a['collision_pairs']:
            raise ValueError('Internal collision pairs did not round-trip: '+a['name'])
        world.add_articulation(art)
        world.set_articulation_planned(a['name'], a['planned'])
    for obj in spec['objects']:
        world.add_object(object_in(obj))
    acm_in(world.get_allowed_collision_matrix(), spec['acm'])
    return SapienPlanner(world, spec['move_group'],
                         joint_vel_limits=spec['joint_vel_limits'],
                         joint_acc_limits=spec['joint_acc_limits'])


def collision_signature(world):
    return sorted((x.collision_type, x.object_name1, x.object_name2,
                   x.link_name1, x.link_name2) for x in world.check_collision())
