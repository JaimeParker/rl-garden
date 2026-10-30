"""Torch codec for boundary bundles; imported only by opted-in launchers.

Private RNG inventory and any external planner/sampler state are EXPLICIT.
Uninventoried external state must fail validation, not silently disappear.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch


class TorchBoundaryCodec:
    def __init__(self, runtime, *, torch_generators, numpy_generators, fingerprint):
        self.runtime = runtime
        self.torch_generators = dict(torch_generators)
        self.numpy_generators = dict(numpy_generators)
        self.fingerprint = str(fingerprint)

    def validate(self, agent):
        if not self.fingerprint:
            raise ValueError("Frozen configuration/source/ACT fingerprint required")
        if agent.env._generator not in self.torch_generators.values():
            raise ValueError("Environment private generator missing from inventory")
        self.runtime.validate(agent)

    def prepare_boundary(self, agent):
        self.runtime.prepare_boundary(agent)

    def capture_rng(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                    private_torch={k: (str(v.device), v.get_state()) for k, v in self.torch_generators.items()},
                    private_numpy={k: (type(v.bit_generator).__name__, copy.deepcopy(v.bit_generator.state))
                                   for k, v in self.numpy_generators.items()})

    def restore_rng(self, rng):
        if set(rng["private_torch"]) != set(self.torch_generators) or set(rng["private_numpy"]) != set(self.numpy_generators):
            raise ValueError("Private RNG inventory mismatch")
        cuda_count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
        if len(rng["cuda"]) != cuda_count:
            raise ValueError("CUDA RNG device inventory mismatch")
        for k, g in self.torch_generators.items():
            if str(g.device) != rng["private_torch"][k][0]:
                raise ValueError("Private torch generator device mismatch")
        for k, g in self.numpy_generators.items():
            if type(g.bit_generator).__name__ != rng["private_numpy"][k][0]:
                raise ValueError("NumPy bit-generator mismatch")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"].cpu())
        if cuda_count:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        for k, g in self.torch_generators.items():
            g.set_state(rng["private_torch"][k][1].cpu())
        for k, g in self.numpy_generators.items():
            g.bit_generator.state = copy.deepcopy(rng["private_numpy"][k][1])

    def save(self, agent, path, context):
        for i in range(torch.cuda.device_count() if torch.cuda.is_initialized() else 0):
            torch.cuda.synchronize(i)
        runtime_state = self.runtime.capture(agent)
        rng = self.capture_rng()
        # Direct stock save avoids an older instance-level checkpoint wrapper.
        from rl_garden.algorithms.base_algorithm import BaseAlgorithm
        try:
            BaseAlgorithm.save(agent, path / "checkpoint.pt", include_replay_buffer=True)
            torch.save(dict(version=1, fingerprint=self.fingerprint, rng=rng,
                            runtime=runtime_state, loop=context,
                            step=agent._global_step, update=agent._global_update,
                            replay_pos=agent.replay_buffer.pos, replay_full=agent.replay_buffer.full),
                       path / "boundary.pt")
        finally:
            # Serialization must not advance training RNG, even on an I/O error.
            self.restore_rng(rng)

    def load(self, agent, path):
        from rl_garden.algorithms.base_algorithm import BaseAlgorithm
        saved = torch.load(path / "boundary.pt", map_location="cpu", weights_only=False)
        if saved["version"] != 1 or saved["fingerprint"] != self.fingerprint:
            raise ValueError("Boundary config/source fingerprint mismatch")
        BaseAlgorithm.load(agent, path / "checkpoint.pt", load_replay_buffer=True, load_optimizers=True)
        if (agent._global_step, agent._global_update, agent.replay_buffer.pos, agent.replay_buffer.full) != (
                saved["step"], saved["update"], saved["replay_pos"], saved["replay_full"]):
            raise ValueError("Learner/replay/boundary counters disagree")
        self.runtime.restore(agent, saved["runtime"])
        self.restore_rng(saved["rng"])  # LAST: construction/loading may consume RNG
        return saved["loop"]


class RoboTwinBoundaryRuntime:
    """Single-env V2 reset runtime with audited external components.

    external must contain 'planner' and 'sampler' components implementing
    validate/prepare_boundary/state_dict/load_state_dict. Stateless components
    still require an explicit implementation and code audit at the launcher.
    This is intentional: no guessed private-process RNG recovery.
    """

    ENV_FIELDS = ("_current_seed_index", "_success_seeds", "reset_state_ids")
    ADAPTER_FIELDS = ("reset_count", "_eval_video_index", "env_seed")

    def __init__(self, external):
        self.external = dict(external)

    def validate(self, agent):
        if agent.num_envs != 1 or agent.env.cfg.clear_cache_freq != 1:
            raise ValueError("Prepared runtime requires single-env V2 cache cleanup every reset")
        if not callable(getattr(agent.env.executor, "prepare_boundary_reset", None)):
            raise ValueError("Executor missing pre-save task release hook")
        if not {"planner", "sampler"} <= self.external.keys():
            raise ValueError("Explicit planner and sampler state inventory required")
        for component in self.external.values():
            for name in ("validate", "prepare_boundary", "state_dict", "load_state_dict"):
                if not callable(getattr(component, name, None)):
                    raise ValueError("Incomplete external state protocol: " + name)
            component.validate(agent)

    def prepare_boundary(self, agent):
        for component in self.external.values():
            component.prepare_boundary(agent)
        agent.env.executor.prepare_boundary_reset()

    def capture(self, agent):
        env = agent.env
        if not env.boundary_reset_pending or any(e.adapter.task is not None for e in env.executor.envs):
            raise ValueError("Checkpoint is not at released-task pre-reset boundary")
        if getattr(agent, "_cached_base_actions", None) is not None:
            raise ValueError("ACT next-action cache was not cleared at episode end")
        return dict(env={k: copy.deepcopy(getattr(env, k)) for k in self.ENV_FIELDS},
                    adapters=[{k: copy.deepcopy(getattr(e.adapter, k)) for k in self.ADAPTER_FIELDS}
                              for e in env.executor.envs],
                    warmup_use_policy=copy.deepcopy(getattr(agent, "_warmup_use_policy", None)),
                    external={k: v.state_dict() for k, v in self.external.items()})

    def restore(self, agent, state):
        env = agent.env
        if any(e.adapter.task is not None for e in env.executor.envs):
            raise ValueError("Construct a fresh, not-yet-reset environment for resume")
        if set(state["external"]) != set(self.external):
            raise ValueError("External runtime inventory mismatch")
        for k, value in state["env"].items():
            old = getattr(env, k)
            setattr(env, k, value.to(old.device) if torch.is_tensor(old) else value)
        for e, fields in zip(env.executor.envs, state["adapters"]):
            for k, value in fields.items():
                setattr(e.adapter, k, value)
        if hasattr(agent, "base_action_provider"):
            agent.base_action_provider.reset()
            agent._cached_base_actions = None
            flag = state["warmup_use_policy"]
            agent._warmup_use_policy = flag.to(agent.device) if flag is not None else None
        for k, component in self.external.items():
            component.load_state_dict(state["external"][k])
        env.boundary_reset_pending = True
