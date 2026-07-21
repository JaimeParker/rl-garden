# tests/test_obs_space_utils.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_garden.envs.wrappers.obs_space_utils import drop_dict_keys


def _space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "privileged": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        }
    )


def test_drop_dict_keys_removes_named_key():
    out = drop_dict_keys(_space(), ("privileged",))
    assert set(out.spaces) == {"rgb", "state"}


def test_drop_dict_keys_empty_keys_is_noop():
    space = _space()
    out = drop_dict_keys(space, ())
    assert set(out.spaces) == set(space.spaces)
    assert out is not space  # always returns a new Dict, never mutates the input


def test_drop_dict_keys_ignores_absent_key():
    out = drop_dict_keys(_space(), ("does_not_exist",))
    assert set(out.spaces) == {"rgb", "state", "privileged"}
