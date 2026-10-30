import pickle
import sys

import pytest

from rl_garden.envs.robotwin.planner_process import PlannerProcess


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux isolated planner')
def test_process_reuse_and_boundary_rng_roundtrip():
    worker = PlannerProcess(timeout=10, retries=0, worker_mode='echo')
    try:
        first = worker.call({'value': 1})
        state = worker.call({'_boundary_rng': 'get'})
        assert worker.call({'_boundary_rng': 'set', 'state': state})['restored']
        assert pickle.dumps(worker.call({'_boundary_rng': 'get'})) == pickle.dumps(state)
        assert worker.call({'value': 2})['pid'] == first['pid']
    finally:
        worker.close()
    assert worker.process is None


def test_native_requires_explicit_source_identity():
    with pytest.raises(ValueError, match='source and SHA256'):
        PlannerProcess()


def test_bridge_rejects_changed_source_before_simulator_import(tmp_path):
    # A changed planner must be rejected before creating CUDA or a simulator.
    from rl_garden.envs.robotwin.isolated_planner import install_planner
    source = tmp_path / 'planner.py'
    source.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        install_planner(planner_source=source, planner_sha256='0' * 64)
