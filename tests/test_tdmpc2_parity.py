"""Numeric-parity regression test for TD-MPC2's ``_gradient_step()``.

Builds a small ``TDMPC2`` deterministically, fills its replay buffer with a
fixed synthetic trajectory (no environment/simulator involved), runs
``N=5`` ``_gradient_step()`` calls, and compares the recorded loss/scale
metrics against a golden JSON fixture
(``tests/fixtures/tdmpc2_parity.json``) with ``atol=1e-5, rtol=1e-5``.

RNG sources enumerated and controlled:

- Init RNG (``LatentConsistencyModel.apply_init()`` + ``TDMPC2Policy
  .__init__``'s actor/critic/critic_target ``trunc_normal_`` draws, run in
  that exact order for RNG-stream parity with the pre-split single-class
  design -- see ``TDMPC2Policy.__init__``'s docstring): seeded via
  ``torch.manual_seed()`` *before* ``TDMPC2(...)`` is constructed.
- Q-head ``randperm`` (``TDMPC2Policy.Q()``) and policy ``randn_like``
  (``TDMPC2Policy.pi()``): consume the same global torch RNG stream, seeded
  once above -- no reseed between gradient steps, so the fixture captures
  one continuous deterministic stream.
- Dropout: set to ``dropout=0.0`` in the parity config (recorded in the
  fixture's ``config`` block) -- this removes dropout's RNG draws from the
  stream entirely rather than trying to control them.
- Replay-buffer window sampling (``torch.randint`` in
  ``StrictWindowSamplingMixin._sample_valid_window_starts``): same global
  stream.
- Planner noise / Gumbel-softmax sampling: not exercised by
  ``_gradient_step()`` (the planner is only used for env rollout action
  selection), so left uncontrolled here.

The fixture's synthetic trajectory has no episode boundaries in it, so this
test pins only the interior (never-crosses-an-episode) sampling path of
``SequenceReplayBuffer``'s strict mode; boundary/tail-step behavior (a
window ending exactly at an episode end) is not exercised here and is
instead covered functionally by ``tests/test_sequence_replay_buffer.py``.

The fixture is regenerated only via ``generate_fixture()``, gated by the
``TDMPC2_PARITY_REGEN=1`` environment variable (never implicitly during a
normal test run) -- see ``.agents``/model-based-base plan section 1.0.
Regenerate on 6017 (after any intentional numeric fix) with:

    TDMPC2_PARITY_REGEN=1 python -m pytest tests/test_tdmpc2_parity.py -q -p no:cacheprovider

The exact float values are only reproducible on the torch build/platform
that generated them (CPU reduction order differs across torch versions and
BLAS backends) -- the fixture records ``torch_version``/``platform`` and the
comparison test checks those first, failing with an explicit "regenerate"
message instead of a wall of numeric diffs on a version mismatch.
"""
from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.tdmpc2 import TDMPC2

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tdmpc2_parity.json"

_PARITY_SEED = 20260914
_TRAJECTORY_LEN = 200
_OBS_DIM = 8
_ACTION_DIM = 3
_NUM_GRADIENT_STEPS = 5
_ATOL = 1e-5
_RTOL = 1e-5

# Small, fixed hyperparameters for a fast, deterministic CPU test.
# dropout=0.0 removes dropout's RNG draws from the stream entirely (see
# module docstring) instead of trying to control them.
_CONFIG: dict = dict(
    device="cpu",
    buffer_device="cpu",
    buffer_size=_TRAJECTORY_LEN,
    batch_size=8,
    episode_length=_TRAJECTORY_LEN,
    seed_steps=1,
    horizon=3,
    num_samples=8,
    num_elites=4,
    num_pi_trajs=2,
    iterations=1,
    latent_dim=16,
    mlp_dim=16,
    simnorm_dim=4,
    num_q=2,
    num_bins=11,
    vmin=-10.0,
    vmax=10.0,
    dropout=0.0,
    episodic=False,
    lr=3e-4,
    enc_lr_scale=0.3,
    grad_clip_norm=20.0,
    tau=0.01,
    rho=0.5,
    consistency_coef=20.0,
    reward_coef=0.1,
    value_coef=0.1,
    seed=_PARITY_SEED,
)

_METRIC_KEYS = (
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "pi_loss",
    "total_loss",
    "grad_norm",
    "pi_grad_norm",
    "pi_scale",
)


class _FixedDimsEnv:
    """A no-op env: only its spaces are read (world model + buffer
    construction). Rollout is never exercised -- the buffer is filled
    directly by ``_fill_buffer`` below."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, action_dim), dtype=np.float32
        )


def _synthetic_trajectory(
    obs_dim: int, action_dim: int, length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed sin/cos-formula trajectory -- deliberately not RNG-based, so
    filling the buffer never touches the torch RNG stream the gradient
    steps' Q-index/policy sampling depend on."""
    t = torch.arange(length, dtype=torch.float32)
    obs_phase = torch.arange(obs_dim, dtype=torch.float32) * 0.37
    obs = torch.sin(0.1 * t.unsqueeze(-1) + obs_phase.unsqueeze(0))  # (length, obs_dim)
    action_phase = torch.arange(action_dim, dtype=torch.float32) * 0.53
    action = torch.cos(0.2 * t.unsqueeze(-1) + action_phase.unsqueeze(0))  # (length, action_dim), in [-1, 1]
    reward = torch.sin(0.05 * t) + 0.5 * torch.cos(0.11 * t)  # (length,)
    return obs, action, reward


def _fill_buffer(agent: TDMPC2) -> None:
    obs, action, reward = _synthetic_trajectory(_OBS_DIM, _ACTION_DIM, _TRAJECTORY_LEN)
    terminated = torch.zeros(1, dtype=torch.bool)
    episode_end = torch.zeros(1, dtype=torch.bool)  # single continuous episode
    for t in range(_TRAJECTORY_LEN):
        obs_t = {"state": obs[t].unsqueeze(0)}
        next_obs_t = {"state": obs[min(t + 1, _TRAJECTORY_LEN - 1)].unsqueeze(0)}
        agent.replay_buffer.add(
            obs_t, next_obs_t, action[t].unsqueeze(0), reward[t : t + 1], terminated, episode_end
        )


def _seed_and_configure_determinism() -> tuple[int, bool]:
    """Seeds every RNG source and returns the (num_threads,
    deterministic_algorithms_state) this process had before, so callers can
    restore it -- both settings are process-global and must not leak into
    other test files run in the same pytest session."""
    random.seed(_PARITY_SEED)
    np.random.seed(_PARITY_SEED)
    torch.manual_seed(_PARITY_SEED)
    previous_num_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    # Eliminates BLAS/reduction thread-order nondeterminism on CPU.
    torch.set_num_threads(1)
    # warn_only=True: fall back to a warning instead of a hard error for any
    # CPU op without a deterministic implementation, rather than blocking
    # the whole parity test on one op ("if the ops allow", plan 1.0).
    torch.use_deterministic_algorithms(True, warn_only=True)
    return previous_num_threads, previous_deterministic


def _build_and_run() -> list[dict[str, float]]:
    """Seed, construct TDMPC2 deterministically, fill the buffer, and run
    ``_NUM_GRADIENT_STEPS`` gradient steps. Returns the recorded metrics."""
    previous_num_threads, previous_deterministic = _seed_and_configure_determinism()
    try:
        env = _FixedDimsEnv(_OBS_DIM, _ACTION_DIM)
        agent = TDMPC2(env=env, **_CONFIG)
        _fill_buffer(agent)

        steps = []
        for _ in range(_NUM_GRADIENT_STEPS):
            info = agent._gradient_step()
            steps.append({key: float(info[key]) for key in _METRIC_KEYS})
        return steps
    finally:
        torch.set_num_threads(previous_num_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)


def generate_fixture() -> dict:
    return {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "provenance": (
            "Generated on 6017 after fixing tdmpc2 bugs 1(a) (target_Q cloned "
            "after _apply_init) and 1(b) (planner _estimate_value bootstraps "
            "with the online Q, not target_Q) -- model-based-base plan 1.0."
        ),
        "config": _CONFIG,
        "metrics": _build_and_run(),
    }


def _assert_close(actual: float, expected: float, *, where: str) -> None:
    diff = abs(actual - expected)
    tol = _ATOL + _RTOL * abs(expected)
    assert diff <= tol, f"{where}: got {actual!r}, want {expected!r} (diff={diff!r} > tol={tol!r})"


def test_gradient_step_matches_golden_fixture() -> None:
    if not FIXTURE_PATH.exists():
        raise AssertionError(
            f"Missing golden fixture {FIXTURE_PATH!s}. Generate it on 6017 with "
            "TDMPC2_PARITY_REGEN=1 (see this module's docstring), sync it back, "
            "and commit it."
        )
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture.get("torch_version") == torch.__version__, (
        f"Fixture was generated with torch {fixture.get('torch_version')!r}, this run is "
        f"torch {torch.__version__!r} -- CPU float reduction order differs across torch "
        "builds, so a version mismatch alone explains any numeric diff. Regenerate on 6017 "
        "with TDMPC2_PARITY_REGEN=1 (see this module's docstring) rather than chasing the "
        "diff."
    )
    assert fixture["config"] == _CONFIG, (
        "Fixture was generated from a different config than this test's _CONFIG "
        "-- regenerate with TDMPC2_PARITY_REGEN=1."
    )

    steps = _build_and_run()
    golden_steps = fixture["metrics"]
    assert len(steps) == len(golden_steps) == _NUM_GRADIENT_STEPS

    for step_idx, (actual, golden) in enumerate(zip(steps, golden_steps)):
        for key in _METRIC_KEYS:
            _assert_close(actual[key], golden[key], where=f"step {step_idx} {key!r}")


# Regeneration is opt-in only: `TDMPC2_PARITY_REGEN=1 python -m pytest
# tests/test_tdmpc2_parity.py -q` (or `python tests/test_tdmpc2_parity.py`
# directly) writes a fresh fixture at import time; a normal test run never
# touches this branch.
if os.environ.get("TDMPC2_PARITY_REGEN") == "1":
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(generate_fixture(), indent=2) + "\n")
    print(f"[tdmpc2 parity] wrote fixture to {FIXTURE_PATH}")
