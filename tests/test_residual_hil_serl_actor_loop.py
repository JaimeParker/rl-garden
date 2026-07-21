from __future__ import annotations

import torch

from rl_garden.training.hitl.residual_hil_serl import ResidualHilSerlActorLoop


class _Policy:
    def __init__(self) -> None:
        self.loaded = None

    def eval(self):
        pass

    def load_state_dict(self, params):
        self.loaded = params

    def predict(self, obs, *, base_actions, deterministic=False):
        del obs, deterministic
        return torch.full_like(base_actions, 0.25)


class _Scaler:
    def unscale(self, action):
        return action * 10.0

    def scale(self, action):
        return action / 10.0


class _Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = _Policy()
        self.action_scaler = _Scaler()
        self._cached_base_actions = None
        self.reset_calls = 0
        self.base_calls = 0

    def _obs_to_policy_device(self, obs):
        return obs

    def _base_naction(self, obs):
        del obs
        self.base_calls += 1
        return torch.full((1, 2), 0.1 * self.base_calls)

    def _combine_base_residual(self, base_actions, unit_residual):
        return base_actions + unit_residual

    def _on_env_reset(self, obs):
        del obs
        self.reset_calls += 1
        self._cached_base_actions = None


class _Env:
    num_envs = 1

    def __init__(self, info=None, done=False):
        self.info = {} if info is None else info
        self.done = done
        self.actions = []
        self.reset_calls = 0

    def reset(self, seed=None):
        del seed
        self.reset_calls += 1
        return torch.zeros(1, 3), {}

    def step(self, action):
        self.actions.append(action.detach().clone())
        return (
            torch.ones(1, 3),
            torch.ones(1),
            torch.tensor([self.done]),
            torch.tensor([False]),
            dict(self.info),
        )


class _Sync:
    def __init__(self):
        self.transitions = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def latest_policy_params(self):
        return None

    def push_transition(self, transition):
        self.transitions.append(transition)


def test_actor_loop_pushes_residual_replay_fields_without_intervention():
    env = _Env()
    agent = _Agent()
    sync = _Sync()

    ResidualHilSerlActorLoop(env, agent, sync, control_hz=1_000_000).run(total_steps=1)

    transition = sync.transitions[0]
    torch.testing.assert_close(env.actions[0], torch.full((1, 2), 3.5))
    torch.testing.assert_close(transition["action"], torch.full((1, 2), 0.35))
    torch.testing.assert_close(transition["base_actions"], torch.full((1, 2), 0.1))
    torch.testing.assert_close(transition["next_base_actions"], torch.full((1, 2), 0.2))
    assert transition["intervened"] is False
    assert sync.started is True
    assert sync.stopped is True


def test_actor_loop_scales_intervention_action_for_replay():
    env = _Env(info={"intervene_action": torch.full((1, 2), 6.0)})
    agent = _Agent()
    sync = _Sync()

    ResidualHilSerlActorLoop(env, agent, sync, control_hz=1_000_000).run(total_steps=1)

    transition = sync.transitions[0]
    torch.testing.assert_close(transition["action"], torch.full((1, 2), 0.6))
    assert transition["intervened"] is True
