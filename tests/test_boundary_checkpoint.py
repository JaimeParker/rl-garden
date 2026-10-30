"""Standard-library logic tests; no training/renderer or branch comparison."""
import ast
from collections import defaultdict
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("boundary", ROOT / "rl_garden/common/boundary_checkpoint.py")
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = boundary.BoundaryStore(Path(self.tmp.name) / "bundles", "test-run")

    def save(self, step, pins=()):
        return self.store.publish(step, pins, {}, lambda p: (p / "model").write_bytes(b"test"))

    def test_rolling_replaced_after_commit(self):
        first = self.save(50008)
        second = self.save(60004)
        self.assertFalse((self.store.root / first).exists())
        self.assertEqual(self.store.index()["latest"], second)
        self.store.verify(second)

    def test_all_comparison_points_retained(self):
        names = [self.save(n + 12, (n,)) for n in (150000, 175000, 200000, 220000)]
        self.save(230001)
        self.save(240002)
        self.assertEqual(len(list(self.store.root.glob("gen-*"))), 5)
        for name in names:
            self.store.verify(name)

    def test_failed_write_keeps_previous(self):
        first = self.save(50000)
        def fail(p):
            (p / "partial").write_bytes(b"partial")
            raise OSError("injected disk failure")
        with self.assertRaises(OSError):
            self.store.publish(60000, [], {}, fail)
        self.assertEqual(self.store.index()["latest"], first)
        self.store.verify(first)

    def test_pin_cannot_be_overwritten(self):
        self.save(150010, (150000,))
        with self.assertRaises(ValueError):
            self.save(150020, (150000,))

    def test_wrong_owner_rejected(self):
        with self.assertRaises(ValueError):
            boundary.BoundaryStore(self.store.root, "other-run")

    def test_corruption_rejected(self):
        name = self.save(50000)
        (self.store.root / name / "model").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.store.verify(name)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.verify("../other")

    def test_does_not_consume_python_rng(self):
        state = random.getstate()
        self.save(50000)
        self.assertEqual(random.getstate(), state)

    def test_second_writer_rejected(self):
        (self.store.root / ".writer.lock").touch()
        with self.assertRaises(FileExistsError):
            self.save(50000)
        self.assertTrue((self.store.root / ".writer.lock").exists())

    def test_schedule_waits_for_done_and_includes_175k(self):
        codec = NS(prepare_boundary=lambda a: None,
                   save=lambda a, p, c: (p / "loop.json").write_text(json.dumps(c)))
        mgr = boundary.BoundaryCheckpoints(self.store, codec, 220000)
        agent = NS(_global_step=60009, env=NS(boundary_reset_pending=False))
        mgr.after_transition(agent, {})
        self.assertIsNone(self.store.index()["latest"])
        agent.env.boundary_reset_pending = True
        mgr.after_transition(agent, {})
        self.assertEqual(mgr.completed, [50000, 60000])
        agent._global_step = 175011
        mgr.after_transition(agent, {})
        self.assertIn("175000", self.store.index()["pins"])
        agent._global_step = 220009
        self.assertTrue(mgr.after_transition(agent, {}))
        self.assertEqual(self.store.index()["final"], self.store.index()["latest"])


def method(path, class_name, method_name, globals_):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), globals_)
    return globals_[method_name]


class Scalar:
    def __init__(self, value): self.value = value
    def clone(self): return Scalar(self.value)
    def float(self): return self
    def sum(self): return self
    def item(self): return self.value
    def numel(self): return 1


class LoopTests(unittest.TestCase):
    def test_real_env_defer_does_not_call_reset_or_draw_seed(self):
        handle = method("rl_garden/envs/robotwin/env.py", "RoboTwinEnv", "_handle_auto_reset",
                        {"_clone_info": copy.deepcopy})
        env = NS(_boundary_checkpoints_enabled=True)
        obs = {"state": Scalar(3)}
        returned, infos = handle(env, True, obs, {"episode": {"return": 7}})
        self.assertTrue(env.boundary_reset_pending)
        self.assertIs(returned, obs)
        self.assertEqual(infos["final_info"]["episode"]["return"], 7)
        self.assertIsNot(infos["final_observation"]["state"], obs["state"])

    def test_real_loop_partial_block_and_no_initial_reseed(self):
        learn = method("rl_garden/algorithms/off_policy.py", "OffPolicyAlgorithm", "learn",
                       {"defaultdict": defaultdict, "time": time})
        events = []
        env = NS(boundary_reset_pending=True)
        def reset_pending():
            events.append("reset")
            env.boundary_reset_pending = False
            return "new-scene"
        def step(action):
            events.append("step")
            env.boundary_reset_pending = True
            return "terminal", Scalar(1), True, False, {}
        env.complete_boundary_reset = reset_pending
        env.step = step
        restored = dict(previous_step=49984, rollout_offset=63, learning_has_started=True,
                        cumulative={}, rollout_reward_sum=0, rollout_reward_count=63,
                        rollout_episode_metrics={})
        manager = NS(finished=False, begin=lambda a, t: restored)
        def after(a, context):
            events.append("save")
            self.assertEqual(context["rollout_offset"], 64)
            self.assertEqual(events[-2], "post")
            manager.finished = True
            return True
        manager.after_transition = after
        agent = NS(boundary_checkpoints=manager, env=env, _global_step=50047,
                   learning_starts=1, steps_per_env=64, num_envs=1, eval_freq=0,
                   logger=None, training_freq=64,
                   _on_training_start=lambda n: None,
                   _rollout_action=lambda o, l: (0, 0, None),
                   _on_env_step=lambda *a: None,
                   _clone_obs=lambda o: o,
                   _compute_done_masks=lambda *a: (False, False),
                   _write_final_obs=lambda *a: None,
                   _replay_buffer_add_kwargs=lambda *a: {},
                   _replay_buffer_step_kwargs=lambda *a: {},
                   replay_buffer=NS(add=lambda *a, **kw: events.append("replay")),
                   _post_rollout_step=lambda *a: events.append("post"))
        learn(agent, 50000)
        self.assertEqual(events, ["reset", "step", "replay", "post", "save"])


if __name__ == "__main__":
    unittest.main()
