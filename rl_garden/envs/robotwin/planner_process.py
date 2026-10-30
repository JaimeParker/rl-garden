"""Candidate persistent, exec-created CPU planner. Never forks live CUDA state.

Only the parent executes physics. One in-flight request; terminate and reap the
old worker before retry. All transport deadlines include complete message I/O.
"""
import atexit
import os
import pickle
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import ast
import hashlib


def wait_io(sock, deadline, writing=False):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('planner IPC deadline')
    ready = select.select([] if writing else [sock], [sock] if writing else [], [], remaining)
    if not (ready[1] if writing else ready[0]):
        raise TimeoutError('planner IPC deadline')


def send(sock, value, deadline):
    data = pickle.dumps(value, protocol=5)
    view = memoryview(struct.pack('!Q', len(data))+data)
    while view:
        wait_io(sock, deadline, True)
        count = sock.send(view)
        if count == 0: raise EOFError('planner socket closed')
        view = view[count:]


def receive(sock, deadline):
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            wait_io(sock, deadline)
            part = sock.recv(min(size-len(chunks), 1048576))
            if not part: raise EOFError('planner socket closed')
            chunks.extend(part)
        return chunks
    size, = struct.unpack('!Q', exact(8))
    if size > 128*1024*1024: raise ValueError('planner message exceeds safety limit')
    return pickle.loads(exact(size))


class PlannerProcess:
    def __init__(self, timeout=60., retries=1, worker_mode='native', grace=0., warn=None, planner_source=None, planner_sha256=None):
        if timeout <= 0 or grace < 0 or retries not in (0, 1): raise ValueError('invalid planner bounds')
        self.timeout, self.retries, self.worker_mode = timeout, retries, worker_mode
        self.grace, self.warn = grace, warn
        self.planner_source = str(Path(planner_source).resolve()) if planner_source else ''
        self.planner_sha256 = planner_sha256 or ''
        if worker_mode == 'native' and (not self.planner_source or len(self.planner_sha256) != 64):
            raise ValueError('Explicit planner source and SHA256 required')
        self.process = self.sock = None
        self.lock = threading.Lock()
        self.events = []
        atexit.register(self.close)

    def start(self):
        assert self.process is None
        parent, child = socket.socketpair()
        parent.setblocking(False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            self.process = subprocess.Popen([sys.executable, '-m', 'rl_garden.envs.robotwin.planner_process',
                '--worker', str(child.fileno()), self.worker_mode, str(os.getpid()), self.planner_source, self.planner_sha256], pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, env=env)
            self.sock = parent
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.events.append(('started', self.process.pid))

    def close(self):
        process, sock = self.process, self.sock
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self.events.append(('reaped', process.pid, process.returncode))
            self.process = None
        if sock is not None:
            sock.close()
            self.sock = None

    def call(self, request):
        with self.lock:
            for attempt in range(self.retries+1):
                if self.process is None: self.start()
                deadline = time.monotonic()+self.timeout+self.grace
                timer = None
                if self.grace > 0 and self.warn is not None:
                    timer = threading.Timer(self.timeout, self.warn)
                    timer.daemon = True
                    timer.start()
                try:
                    send(self.sock, request, deadline)
                    response = receive(self.sock, deadline)
                except (TimeoutError, EOFError, BrokenPipeError, ConnectionResetError):
                    self.close()  # Must finish reaping before another start.
                    if attempt == self.retries: raise
                    continue
                finally:
                    if timer is not None: timer.cancel()
                if 'error' in response:
                    raise RuntimeError('Isolated planning error: '+response['error'])
                return response


def worker(fd, mode, parent_pid, planner_source, planner_sha256):
    # The outer simulator watchdog may use os._exit. Do not leave a hung
    # native planner orphaned when its parent dies without running atexit.
    import ctypes
    import signal
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != parent_pid:
        return
    channel = socket.socket(fileno=fd)
    channel.setblocking(False)
    if mode == 'native':
        from . import planner_world as codec
        # Importing the whole RoboTwin planner module also imports Curobo,
        # which tries to initialize CUDA. Load the exact verified MPlib class
        # AST only, with its original numeric dependencies; do not edit methods.
        import numpy as np
        import mplib
        source = Path(planner_source).read_bytes()
        expected = planner_sha256
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError('RoboTwin planner source identity changed')
        node, = [x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == 'MplibPlanner']
        scope = {'np': np, 'mplib': mplib}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<verified-MplibPlanner>', 'exec'), scope)
        MplibPlanner = scope['MplibPlanner']
    while True:
        try: request = receive(channel, time.monotonic()+86400)
        except EOFError: return
        try:
            if '_boundary_rng' in request:
                import random
                import numpy as rng_numpy
                if request['_boundary_rng'] == 'get':
                    response = dict(python=random.getstate(), numpy=rng_numpy.random.get_state())
                elif request['_boundary_rng'] == 'set':
                    random.setstate(request['state']['python'])
                    rng_numpy.random.set_state(request['state']['numpy'])
                    response = dict(restored=True)
                else:
                    raise ValueError('Unknown RNG control operation')
                send(channel, response, time.monotonic()+60)
                continue
            if mode == 'echo':
                time.sleep(request.get('delay', 0))
                response = {'value': request, 'pid': os.getpid()}
            elif mode == 'hang':
                while True: time.sleep(60)
            elif mode == 'native':
                if request['planner_type'] != 'mplib_screw':
                    raise ValueError('Only validated screw planner allowed; RRT RNG state is not portable')
                obj = MplibPlanner.__new__(MplibPlanner)
                obj.planner = codec.rebuild(request['world'])
                obj.planner_type = request['planner_type']
                obj.plan_step_lim = request['plan_step_lim']
                obj.TOPP = obj.planner.TOPP
                result = obj.plan_path(request['qpos'], codec.pose_in(request['target']),
                                       arms_tag=request['arm'], log=False)
                world = obj.planner.planning_world
                response = {'result': result, 'pid': os.getpid(),
                            'qpos_after': {n: world.get_articulation(n).get_qpos().copy()
                                           for n in world.get_articulation_names()}}
            else: raise ValueError('unknown worker mode')
        except Exception:
            response = {'error': traceback.format_exc()}
        send(channel, response, time.monotonic()+60)


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
