"""Seeded training windows of the PPO family that the snapshots of
``test_ppo_seams.py`` and ``test_ppo_train_snapshot.py`` do not reach, pinned
bit for bit: ``GAIL`` (its discriminator included), decision-level
``RecurrentPPO`` with ``bootstrap_at_done="never"``, ``PPO`` whose
``target_kl`` early stop fires, and the clipped value-loss and entropy terms
with and without a decision mask.

Each case pins the ``train()`` metrics, the number of minibatch updates and
early stops, and the sha256 of the trained modules' ``state_dict``. The values
were recorded on CPU (torch 2.6, one thread) before ``learn()``, ``train()``
and ``_ppo_minibatch_update`` were split into overridable methods; the split
must reproduce them bit for bit. A different torch build or BLAS may round
differently: re-record there instead of loosening the comparison.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable
from unittest import mock

import pytest
import torch
from _decision_fakes import NUM_STEPS, _FixedHoldVecEnv
from test_gail import _fake_demo_loader, _FakeBoxEnv, _gail_kwargs
from test_ppo_seams import state_dict_digest

from rl_garden.algorithms import GAIL, PPO, RecurrentPPO

COMMON_KWARGS = {
    "device": "cpu",
    "num_minibatches": 2,
    "update_epochs": 2,
    "eval_freq": 0,
    "log_freq": 0,
    "std_log": False,
    "seed": 0,
}


def _build_ppo(agent_cls: type[PPO], env_factory: Callable[[], Any], **kwargs: Any):
    def build() -> PPO:
        return agent_cls(
            env_factory(),
            num_steps=NUM_STEPS,
            net_arch=[8],
            **{"target_kl": None, **COMMON_KWARGS, **kwargs},
        )

    return build


def _build_gail() -> GAIL:
    with mock.patch(
        "rl_garden.buffers.d4rl_legacy_dataset.load_d4rl_legacy_dataset_to_replay_buffer",
        _fake_demo_loader,
    ):
        return GAIL(_FakeBoxEnv(), **{**_gail_kwargs(), **COMMON_KWARGS})


def _plain_env() -> _FixedHoldVecEnv:
    return _FixedHoldVecEnv(emit_decisions=False)


#: case -> (agent builder, the attributes whose ``state_dict`` is digested).
CASES: dict[str, tuple[Callable[[], PPO], tuple[str, ...]]] = {
    "gail": (_build_gail, ("policy", "discriminator")),
    "gru-decision-bootstrap_never": (
        _build_ppo(
            RecurrentPPO,
            _FixedHoldVecEnv,
            rnn_hidden_size=8,
            decision_level=True,
            bootstrap_at_done="never",
        ),
        ("policy",),
    ),
    "ppo-target_kl": (_build_ppo(PPO, _plain_env, target_kl=1e-6), ("policy",)),
    "ppo-clip_vloss-ent_coef": (
        _build_ppo(PPO, _plain_env, clip_vloss=True, ent_coef=0.01),
        ("policy",),
    ),
    "ppo-decision-critic_on_decisions-clip_vloss": (
        _build_ppo(
            PPO,
            _FixedHoldVecEnv,
            decision_level=True,
            per_step_critic=False,
            clip_vloss=True,
            ent_coef=0.01,
        ),
        ("policy",),
    ),
}

#: The build the values below were recorded on (CPU only).
SNAPSHOT_TORCH_VERSION = "2.6.0+cu124"
pytestmark = pytest.mark.skipif(
    torch.__version__ != SNAPSHOT_TORCH_VERSION,
    reason=f"snapshot recorded on torch {SNAPSHOT_TORCH_VERSION} CPU; re-record",
)

#: case -> {"metrics", "updates", "stops", "state"}.
SNAPSHOT: dict[str, dict[str, Any]] = {
    "gail": {
        "metrics": {
            "approx_kl": 7.009754824593983e-06,
            "clip_coef": 0.2,
            "clipfrac": 0.0,
            "disc_acc": 0.5,
            "disc_loss": 0.7111062407493591,
            "entropy": 1.8377845883369446,
            "explained_variance": -0.31495440006256104,
            "learning_rate": 0.0003,
            "loss": 0.8079663217067719,
            "old_approx_kl": 0.0010309567005606368,
            "policy_loss": -0.0010467489822376663,
            "value_loss": 1.6180261373519897,
        },
        "state": "e03e4da2fed58a56fce278da9682bbb3bd03f7752ce62711acb06f84aba0abeb",
        "stops": 0,
        "updates": 4,
    },
    "gru-decision-bootstrap_never": {
        "metrics": {
            "approx_kl": 3.606241776310526e-06,
            "clip_coef": 0.2,
            "clipfrac": 0.0,
            "decisions_per_minibatch_mean": 13.5,
            "decisions_per_minibatch_min": 10.0,
            "entropy": 1.8381154835224152,
            "explained_variance": 0.0033271312713623047,
            "learning_rate": 0.0003,
            "loss": 5.423378944396973,
            "old_approx_kl": -0.00016747590370869148,
            "policy_loss": -0.0003910520950736185,
            "value_loss": 10.847540140151978,
        },
        "state": "aaeb7ec83941674bdae397f84bacc0f396934a8c4d8bcb5c1017513cc532b7dd",
        "stops": 0,
        "updates": 4,
    },
    "ppo-clip_vloss-ent_coef": {
        "metrics": {
            "approx_kl": 3.5054982561177894e-06,
            "clip_coef": 0.2,
            "clipfrac": 0.0,
            "entropy": 1.8378483057022095,
            "explained_variance": 0.017375409603118896,
            "learning_rate": 0.0003,
            "loss": 4.557312965393066,
            "old_approx_kl": 0.0001918480981354942,
            "policy_loss": -0.00042888149497832817,
            "value_loss": 9.152240872383118,
        },
        "state": "6e4ad7684c8b96fd38a68ef1e3d358313f94422d8029afe2a320903b90dbb8c1",
        "stops": 0,
        "updates": 4,
    },
    "ppo-decision-critic_on_decisions-clip_vloss": {
        "metrics": {
            "approx_kl": 4.546851926079398e-06,
            "clip_coef": 0.2,
            "clipfrac": 0.0,
            "decisions_per_minibatch_mean": 13.5,
            "decisions_per_minibatch_min": 12.0,
            "entropy": 1.8372746706008911,
            "explained_variance": 0.016781628131866455,
            "learning_rate": 0.0003,
            "loss": 3.8309162855148315,
            "old_approx_kl": 0.0003548628847056534,
            "policy_loss": -0.0008573999026948442,
            "value_loss": 7.7002928256988525,
        },
        "state": "89cbccc3f18d876843791d79e40cb2f30ee9c4f73942c19b71b2f7ccd3801cd4",
        "stops": 0,
        "updates": 4,
    },
    "ppo-target_kl": {
        "metrics": {
            "approx_kl": 0.0,
            "clip_coef": 0.2,
            "clipfrac": 0.0,
            "entropy": 1.837876796722412,
            "explained_variance": 0.017375409603118896,
            "learning_rate": 0.0003,
            "loss": 5.513712406158447,
            "old_approx_kl": 2.4835269396561444e-09,
            "policy_loss": 4.967053879312289e-09,
            "value_loss": 11.027424812316895,
        },
        "state": "60d283c9c541bed8dd0ea61d957aa04f3351e1396542f8671c865368a7fe355a",
        "stops": 1,
        "updates": 2,
    },
}


def train_window(case: str) -> dict[str, Any]:
    """One seeded ``learn()`` window of ``case``: ``train()``'s metrics, the
    number of minibatch updates and of early stops, and the state digest."""
    build, module_names = CASES[case]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = build()
        recorded: list[dict[str, float]] = []
        stops: list[bool] = []
        train = agent.train
        minibatch_update = agent._ppo_minibatch_update

        def recording_train() -> dict[str, float]:
            recorded.append(train())
            return recorded[-1]

        def recording_minibatch_update(**kwargs):
            result = minibatch_update(**kwargs)
            stops.append(bool(result["stop"]))
            return result

        agent.train = recording_train
        agent._ppo_minibatch_update = recording_minibatch_update
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
    finally:
        torch.set_num_threads(threads)
    assert len(recorded) == 1
    state = {
        f"{module_name}.{key}": value
        for module_name in module_names
        for key, value in getattr(agent, module_name).state_dict().items()
    }
    return {
        "metrics": recorded[0],
        "updates": len(stops),
        "stops": sum(stops),
        "state": state_dict_digest(state),
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_seeded_window_matches_the_recorded_snapshot(case):
    window = train_window(case)
    expected = SNAPSHOT[case]
    assert (window["updates"], window["stops"]) == (expected["updates"], expected["stops"])
    assert window["state"] == expected["state"]
    assert window["metrics"].keys() == expected["metrics"].keys()
    for key, value in expected["metrics"].items():
        if math.isnan(value):
            assert math.isnan(window["metrics"][key]), key
            continue
        torch.testing.assert_close(
            torch.tensor(window["metrics"][key], dtype=torch.float64),
            torch.tensor(value, dtype=torch.float64),
            rtol=0,
            atol=0,
            msg=f"{case}: {key}",
        )


def test_the_target_kl_case_stops_early_after_at_least_one_update():
    expected = SNAPSHOT["ppo-target_kl"]
    assert expected["stops"] == 1
    assert expected["updates"] > 1
