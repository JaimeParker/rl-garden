import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.envs.wrappers.dict_state import DictStateObservationWrapper


class _VectorizedStateEnv(gym.Env):
    def __init__(self):
        self.single_observation_space = spaces.Box(-1.0, 1.0, (3,))
        self.observation_space = spaces.Box(-1.0, 1.0, (2, 3))
        self.action_space = spaces.Box(-1.0, 1.0, (2, 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((2, 3), dtype=np.float32), {}


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_single_observation_space_is_preserved_through_wrapper_stack(wrapper_depth):
    env = _VectorizedStateEnv()
    for _ in range(wrapper_depth):
        env = gym.Wrapper(env)

    wrapped = DictStateObservationWrapper(env)
    try:
        assert wrapped.single_observation_space == spaces.Dict(
            {"state": spaces.Box(-1.0, 1.0, (3,))}
        )
        obs, _ = wrapped.reset(seed=0)
        assert wrapped.observation_space.contains(obs)
        assert wrapped.single_observation_space.contains({"state": obs["state"][0]})
    finally:
        wrapped.close()
