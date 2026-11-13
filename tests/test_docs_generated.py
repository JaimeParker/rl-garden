"""Check registry documentation in a fresh, simulator-free interpreter."""
from pathlib import Path
import subprocess
import sys


def test_generated_tables_are_current_without_optional_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    # Block imports rather than depending on whether CI happens to install a simulator.
    script = """
import importlib.abc
import runpy
import sys

class NoOptionalBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        optional = {
            'mani_skill', 'sapien', 'mujoco', 'mujoco_warp', 'dm_control',
            'isaaclab', 'isaacsim', 'robosuite', 'robomimic', 'ogbench',
            'metaworld', 'rlbench', 'pyrep', 'minari', 'd4rl', 'mujoco_py',
        }
        if fullname.split('.')[0] in optional:
            raise AssertionError('Registry documentation imported optional backend: ' + fullname)

sys.meta_path.insert(0, NoOptionalBackends())
sys.argv = ['tools/docs/gen_tables.py', '--check']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Generated tables are current" in result.stdout
