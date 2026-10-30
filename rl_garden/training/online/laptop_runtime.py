"""Explicit external state for single-environment episode-boundary continuation."""
import torch
import numpy as np

def inventory(agent):
    tg,ng={},{}; seen=set()
    def visit(obj,path,depth=0):
        if obj is None or id(obj) in seen or depth>6:return
        seen.add(id(obj))
        for name,value in vars(obj).items() if hasattr(obj,'__dict__') else []:
            key=path+'.'+name
            if isinstance(value,torch.Generator):tg[key]=value
            elif isinstance(value,np.random.Generator):ng[key]=value
            elif isinstance(value,np.random.RandomState):raise RuntimeError('Unregistered legacy private RandomState: '+key)
            elif name in ('env','venv','unwrapped','action_space','observation_space','single_action_space','single_observation_space','base_action_provider','provider','replay_buffer'):
                visit(value,key,depth+1)
            elif name=='spaces' and isinstance(value,dict):
                for k,v in value.items():visit(v,key+'.'+k,depth+1)
    visit(agent,'agent')
    assert any(k.endswith('._generator') for k in tg),'Missing RoboTwin scene generator'
    return tg,ng


class PlannerState:
    def __init__(self, worker, counters):
        self.worker, self.counters = worker, counters

    def validate(self, agent):
        if (agent.env.cfg.planner_backend != 'mplib' or
                agent.env.cfg.delta_ee_planner_type != 'mplib_screw' or
                self.worker.worker_mode != 'native' or self.worker.retries != 0):
            raise ValueError('Only the isolated screw-planner protocol is supported')

    def prepare_boundary(self, agent):
        if self.worker.lock.locked():
            raise RuntimeError('Planner request still in flight')

    def state_dict(self):
        return dict(rng=self.worker.call({'_boundary_rng': 'get'}), counters=dict(self.counters))

    def load_state_dict(self, state):
        if not self.worker.call({'_boundary_rng': 'set', 'state': state['rng']})['restored']:
            raise RuntimeError('Planner RNG restore failed')
        self.counters.clear()
        self.counters.update(state['counters'])


class SamplerState:
    def validate(self, agent):
        # This recipe uses the existing global torch replay sampler, no DataLoader.
        import inspect
        source = inspect.getsource(type(agent.replay_buffer).sample)
        if 'torch.randint' not in source or 'generator=' in source:
            raise ValueError('Replay RNG implementation changed; inventory required')

    def prepare_boundary(self, agent):
        pass

    def state_dict(self):
        return {'backend': 'global-torch-randint'}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('Replay sampler identity mismatch')
