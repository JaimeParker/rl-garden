"""Run MPlib screw planning in an isolated process."""
import numpy as np
from .planner_process import PlannerProcess


def install_planner(*, planner_source, planner_sha256):
    from rl_garden.encoders.laptop.checkpoint import sha256
    if not planner_sha256 or sha256(planner_source) != planner_sha256:
        raise ValueError('Planner source SHA256 mismatch')
    from . import planner_world as codec
    from envs.robot.planner import MplibPlanner
    import rl_garden.envs.robotwin.executor as executor_module
    worker = PlannerProcess(timeout=180, grace=60, retries=0,
                            planner_source=planner_source, planner_sha256=planner_sha256)
    counters = {'calls': 0, 'success': 0, 'failure': 0, 'retries': 0, 'timeouts': 0}
    original_plan = MplibPlanner.plan_path
    if getattr(original_plan, '_isolated', False):
        raise RuntimeError('An isolated planner is already installed')
    original_wait = executor_module.ThreadedRoboTwinExecutor._wait

    def wait(self, future, *, operation):
        # Permit the same bounded planning attempts before reporting a stuck executor.
        timeout = self.timeout_seconds + (900 if operation.startswith('step') else 180)
        return future.result(timeout=timeout)

    executor_module.ThreadedRoboTwinExecutor._wait = wait

    def plan(self, now_qpos, target_pose, use_point_cloud=False, use_attach=False, arms_tag=None, log=True):
        if self.planner_type != 'mplib_screw' or use_point_cloud or use_attach:
            raise ValueError('Isolated planning requires mplib_screw without point clouds or attachments')
        request = {'world': codec.capture(self.planner), 'planner_type': self.planner_type,
                   'plan_step_lim': self.plan_step_lim, 'qpos': np.asarray(now_qpos).copy(),
                   'target': codec.pose_out(target_pose), 'arm': arms_tag}
        response = None
        for attempt in range(2):
            try:
                response = worker.call(request)
                result = response['result']
            except TimeoutError:
                counters['timeouts'] += 1
                result = {'status': 'Fail', 'error': 'isolated planning timeout after grace'}
                response = None
            if result.get('status') == 'Success': break
            if attempt == 0: counters['retries'] += 1
        if not isinstance(result, dict) or 'status' not in result:
            raise ValueError('Invalid isolated planning result')
        if result['status'] == 'Success':
            for key in ('position', 'velocity'):
                arr = np.asarray(result[key])
                if arr.ndim != 2 or arr.shape[0] == 0 or not np.isfinite(arr).all():
                    raise ValueError('Invalid trajectory: '+key)
        world = self.planner.planning_world
        if response is not None and set(response['qpos_after']) != set(world.get_articulation_names()):
            raise ValueError('Worker articulation identity mismatch')
        # Preserve only the original planner's internal kinematic state.
        # These are NOT SAPIEN articulations; this never advances physics.
        for name, qpos in (response['qpos_after'].items() if response is not None else ()):
            world.get_articulation(name).set_qpos(qpos, full=True)
        counters['calls'] += 1
        counters['success' if result['status']=='Success' else 'failure'] += 1
        return result

    plan._isolated = True
    MplibPlanner.plan_path = plan
    def close():
        if worker.lock.locked():
            raise RuntimeError('Cannot close a planner with an in-flight request')
        worker.close()
        if MplibPlanner.plan_path is plan:
            MplibPlanner.plan_path = original_plan
        if executor_module.ThreadedRoboTwinExecutor._wait is wait:
            executor_module.ThreadedRoboTwinExecutor._wait = original_wait
    return worker, counters, close
