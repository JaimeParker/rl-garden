"""Opt-in, single-environment pre-reset checkpoint scheduling and publication.

No simulator or torch imports: the training-specific codec supplies synchronous
save/load callbacks. A fresh output directory belongs to ONE run/one writer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


def _sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BoundaryStore:
    """Atomic index switches only after a complete, verified generation exists.

    Pinned generations are immutable. Ordinary generations are removed ONLY
    after publication, and ONLY when named in our old index and owned by us.
    An interrupted write cannot invalidate the preceding latest checkpoint.
    """

    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        self.identity = str(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.root / "owner.json"
        if owner.exists():
            if json.loads(owner.read_text())["identity"] != self.identity:
                raise ValueError("Checkpoint directory belongs to another run")
        else:
            if any(self.root.iterdir()):
                raise ValueError("Use a new, empty boundary checkpoint directory")
            with owner.open("x", encoding="utf-8") as f:
                json.dump({"identity": self.identity, "version": 1}, f)
                f.flush()
                os.fsync(f.fileno())
            _sync_dir(self.root)

    def index(self):
        p = self.root / "index.json"
        return json.loads(p.read_text()) if p.exists() else {"latest": None, "pins": {}}

    def _generation(self, name):
        if not re.fullmatch(r"gen-[0-9]+-[0-9a-f]{32}", name):
            raise ValueError("Invalid owned generation name")
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("Generation escaped checkpoint directory")
        return path

    def verify(self, name):
        path = self._generation(name)
        if (path / "manifest.json").is_symlink():
            raise ValueError("Unsafe manifest")
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["identity"] != self.identity:
            raise ValueError("Generation owner mismatch")
        if set(p.name for p in path.iterdir()) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Unexpected or missing bundle files")
        for name, info in manifest["files"].items():
            p = path / name
            if Path(name).name != name or p.is_symlink() or not p.is_file():
                raise ValueError("Unsafe bundle member")
            if p.stat().st_size != info["bytes"] or _hash(p) != info["sha256"]:
                raise ValueError("Corrupt checkpoint bundle")
        return path, manifest

    def publish(self, step, pins, metadata, writer):
        lock = self.root / ".writer.lock"
        # Fail closed on a stale lock; don't steal a lock from a saving process.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            old = self.index()
            if set(map(str, pins)) & set(old["pins"]):
                raise ValueError("Refusing to replace a pinned comparison point")
            name = f"gen-{int(step)}-{uuid.uuid4().hex}"
            staging = self.root / (".pending-" + name)
            staging.mkdir()
            started = time.monotonic()
            writer(staging)  # synchronous, learner/replay remain frozen
            files = {}
            for p in staging.iterdir():
                if p.is_symlink() or not p.is_file() or p.name == "manifest.json":
                    raise ValueError("Bundle must contain plain, self-contained files")
                with p.open("rb+") as f:
                    os.fsync(f.fileno())
                files[p.name] = {"bytes": p.stat().st_size, "sha256": _hash(p)}
            if not files:
                raise ValueError("Empty checkpoint")
            manifest = dict(version=1, identity=self.identity, step=int(step),
                            pins=list(pins), metadata=metadata, files=files,
                            save_seconds=time.monotonic() - started)
            with (staging / "manifest.json").open("x", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _sync_dir(staging)
            destination = self._generation(name)
            os.replace(staging, destination)
            _sync_dir(self.root)
            self.verify(name)
            index = {"latest": name, "pins": dict(old["pins"])}
            index["pins"].update({str(p): name for p in pins})
            if metadata.get("finished"):
                index["final"] = name
            temporary = self.root / ".index.next"
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.root / "index.json")
            _sync_dir(self.root)
            obsolete = old["latest"]
            if obsolete and obsolete not in index["pins"].values():
                target, _ = self.verify(obsolete)
                # Only the previous committed, owned, non-pinned flat bundle.
                shutil.rmtree(target)
                _sync_dir(self.root)
            return name
        finally:
            lock.unlink()


class BoundaryCheckpoints:
    """Manager for the opt-in OffPolicyAlgorithm loop hooks.

    Codec contract: validate(agent), save(agent,path,context),
    load(agent,path)->context. The codec must include RNG/private runtime state.
    Finalization waits for natural done at/after budget; no artificial truncation.
    """

    def __init__(self, store, codec, total_steps, resume=None):
        self.store, self.codec = store, codec
        self.total_steps = int(total_steps)
        if self.total_steps < 50000:
            raise ValueError("Boundary schedule starts at 50000")
        self.pins = {s for s in (150000, 175000, 200000, 220000) if s <= self.total_steps}
        self.thresholds = sorted(set(range(50000, self.total_steps + 1, 10000))
                                 | self.pins | {self.total_steps})
        self.completed = []
        self.resume = resume
        self.finished = False

    def begin(self, agent, total_steps):
        if total_steps != self.total_steps or agent.num_envs != 1:
            raise ValueError("Budget mismatch or unsupported multi-env boundary")
        if not getattr(agent.env, "boundary_checkpoint_capable", False):
            raise ValueError("Environment lacks deferred-reset support")
        self.codec.validate(agent)
        agent.env.enable_boundary_checkpoints()
        if self.resume is None:
            if agent._global_step or self.store.index()["latest"]:
                raise ValueError("Explicit boundary resume required for an existing run")
            return None
        path, manifest = self.store.verify(self.resume)
        if manifest["metadata"]["total_steps"] != total_steps:
            raise ValueError("Cannot change training budget on boundary resume")
        context = self.codec.load(agent, path)
        if agent._global_step != manifest["step"]:
            raise ValueError("Model/context step mismatch")
        offset = context["rollout_offset"]
        if not 1 <= offset <= agent.steps_per_env:
            raise ValueError("Invalid saved rollout block position")
        if context["previous_step"] + offset * agent.num_envs != agent._global_step:
            raise ValueError("Rollout block and learner counters disagree")
        expected = [s for s in self.thresholds if s <= agent._global_step]
        if context["completed_thresholds"] != expected:
            raise ValueError("Threshold schedule does not match saved training step")
        if context["finished"] != (agent._global_step >= total_steps):
            raise ValueError("Invalid final-boundary marker")
        self.completed = context["completed_thresholds"]
        self.finished = context["finished"]
        return context

    def after_transition(self, agent, context):
        if not agent.env.boundary_reset_pending:
            return False
        self.codec.prepare_boundary(agent)
        due = [s for s in self.thresholds if s <= agent._global_step and s not in self.completed]
        self.finished = agent._global_step >= self.total_steps
        if due:
            context = dict(context, completed_thresholds=self.completed + due,
                           finished=self.finished, total_steps=self.total_steps)
            self.store.publish(agent._global_step, sorted(set(due) & self.pins),
                               {"total_steps": self.total_steps, "thresholds": due,
                                "boundary": "transition_committed_before_reset",
                                "finished": self.finished},
                               lambda p: self.codec.save(agent, p, context))
            self.completed += due
        return self.finished
