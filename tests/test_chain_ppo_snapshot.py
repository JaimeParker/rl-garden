"""Seeded training windows of the chain-PPO family (``ChainPPOCore``:
``DPPO``, ``FlowPPO`` and ``DiffusionCMDistillOnline``), pinned bit for bit.

The baseline is the code AFTER ``DPPO``/``FlowPPO`` moved onto
``ChainPPOCore`` (advantages normalized and quantile-clamped per minibatch
for DPPO and once over the rollout for FlowPPO, each as its reference does;
the shared value loss and actor gradient clip), not the code before it:
their numerics changed deliberately there. Each case pins the
``train()`` metrics, the number of minibatch gradient steps and the sha256
of the trained modules' ``state_dict``, on CPU (torch 2.6, one thread),
like ``test_ppo_loop_snapshot.py``. A different torch build or BLAS may
round differently: re-record there instead of loosening the comparison.

Recording: on the snapshot build (``SNAPSHOT_TORCH_VERSION``), from
``tests/``, run ``python test_chain_ppo_snapshot.py`` and paste the printed
dict over ``SNAPSHOT``.
"""

from __future__ import annotations

import math
import pprint
import warnings
from typing import Any, Callable

import pytest
import torch
from test_dppo_smoke import _make_env as _make_dppo_env
from test_flow_ppo import _make_env as _make_flow_env
from test_ppo_seams import state_dict_digest

from rl_garden.algorithms import DPPO, DiffusionCMDistillOnline, FlowPPO
from rl_garden.algorithms.chain_ppo_core import ChainPPOCore

NUM_ENVS = 4
COMMON_KWARGS = {
    "num_steps": 3,
    "actor_mlp_dims": [16, 16, 16],
    "critic_mlp_dims": [16, 16, 16],
    "update_epochs": 2,
    "update_batch_size": 8,
    "clip_advantage_lower_quantile": 0.1,
    "clip_advantage_upper_quantile": 0.9,
    "grad_clip_norm": 1.0,
    "target_kl": None,
    "device": "cpu",
    "eval_freq": 0,
    "log_freq": 0,
    "std_log": False,
    "seed": 0,
}
DIFFUSION_KWARGS = {
    "horizon_steps": 2,
    "act_steps": 2,
    "denoising_steps": 5,
    "ft_denoising_steps": 3,
}


def _build_dppo(**kwargs: Any) -> Callable[[], ChainPPOCore]:
    def build() -> DPPO:
        return DPPO(
            env=_make_dppo_env(NUM_ENVS, act_steps=2),
            **{**COMMON_KWARGS, **DIFFUSION_KWARGS, **kwargs},
        )

    return build


def _build_flow_ppo(**kwargs: Any) -> Callable[[], ChainPPOCore]:
    def build() -> FlowPPO:
        return FlowPPO(
            env=_make_flow_env(NUM_ENVS, horizon_length=2),
            **{**COMMON_KWARGS, "horizon_length": 2, "flow_steps": 4, **kwargs},
        )

    return build


def _build_cm_distill() -> DiffusionCMDistillOnline:
    return DiffusionCMDistillOnline(
        env=_make_dppo_env(NUM_ENVS, act_steps=2),
        cm_mlp_dims=[16, 16, 16],
        cm_lr=1e-3,
        **{**COMMON_KWARGS, **DIFFUSION_KWARGS},
    )


#: case -> (agent builder, the attributes whose ``state_dict`` is digested).
CASES: dict[str, tuple[Callable[[], ChainPPOCore], tuple[str, ...]]] = {
    "dppo": (_build_dppo(clip_vloss_coef=0.5), ("policy",)),
    "dppo-critic_warmup": (_build_dppo(critic_warmup_updates=1), ("policy",)),
    "dppo-target_kl": (_build_dppo(target_kl=1e-12), ("policy",)),
    "flow_ppo": (_build_flow_ppo(clip_vloss_coef=0.5), ("policy",)),
    "flow_ppo-sde-pseudo": (
        _build_flow_ppo(sde_type="sde", logprob_mode="pseudo"),
        ("policy",),
    ),
    "diffusion_cm_distill_online": (
        _build_cm_distill,
        ("policy", "cm_student", "cm_target"),
    ),
}

#: The build the values below are recorded on (CPU only).
SNAPSHOT_TORCH_VERSION = "2.6.0+cu124"
pytestmark = pytest.mark.skipif(
    torch.__version__ != SNAPSHOT_TORCH_VERSION,
    reason=f"snapshot recorded on torch {SNAPSHOT_TORCH_VERSION} CPU; re-record",
)

#: case -> {"metrics", "updates", "actor_updates", "state"}.
SNAPSHOT: dict[str, dict[str, Any]] = {
    "diffusion_cm_distill_online": {
        "actor_updates": 8,
        "metrics": {
            "approx_kl": 9.021488949656487e-06,
            "clipfrac": 0.0625,
            "cm_distill_loss": 5433.47119140625,
            "explained_variance": -0.2752112150192261,
            "loss": 4.188044488430023,
            "pg_loss": 0.0020273677073419094,
            "ratio": 1.0008177682757378,
            "v_loss": 8.372034311294556,
        },
        "state": "a4cea6833c296234b035b4b07a3af737508e0e9dace90322b67a255ceac43cdb",
        "updates": 8,
    },
    "dppo": {
        "actor_updates": 8,
        "metrics": {
            "approx_kl": 8.311821147799492e-06,
            "clipfrac": 0.03125,
            "explained_variance": 0.10589200258255005,
            "loss": 3.7627177834510803,
            "pg_loss": -0.008737803436815739,
            "ratio": 1.0006594583392143,
            "v_loss": 7.542911171913147,
        },
        "state": "799d1c02441bdc2c520cd7d85323ed5149c7f62959031d667fc69e94aa073898",
        "updates": 8,
    },
    "dppo-critic_warmup": {
        "actor_updates": 0,
        "metrics": {
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "explained_variance": 0.10589200258255005,
            "loss": 3.7629001140594482,
            "pg_loss": -0.008555472129955888,
            "ratio": 1.0,
            "v_loss": 7.542911171913147,
        },
        "state": "f8ac0f34adcf0abfe4096d0faedb2c86f7f732218adac7c8e04ccc76ad5aea11",
        "updates": 8,
    },
    "dppo-target_kl": {
        "actor_updates": 2,
        "metrics": {
            "approx_kl": 9.872019290924072e-08,
            "clipfrac": 0.0,
            "explained_variance": 0.10589200258255005,
            "loss": 3.270872950553894,
            "pg_loss": -0.015675634145736694,
            "ratio": 0.9997782409191132,
            "v_loss": 6.573097229003906,
        },
        "state": "49f0772f91e0e3425c6ebddf8128ee877633f4c3aa803e63921ebc609b3aa1e3",
        "updates": 2,
    },
    "flow_ppo": {
        "actor_updates": 12,
        "metrics": {
            "approx_kl": 0.0004020308454831441,
            "clipfrac": 0.0,
            "explained_variance": -0.009032249450683594,
            "loss": 4.525807758172353,
            "pg_loss": -0.004825264836351077,
            "ratio": 0.9999725371599197,
            "v_loss": 9.06126606464386,
        },
        "state": "11facbca5a68eeff40f0499e7856dc32bde9be80b981f62c15a8181edcb3ac2a",
        "updates": 12,
    },
    "flow_ppo-sde-pseudo": {
        "actor_updates": 12,
        "metrics": {
            "approx_kl": 1.9763926199326914e-08,
            "clipfrac": 0.0,
            "explained_variance": -0.009032249450683594,
            "loss": 4.529224077860515,
            "pg_loss": -0.0014089631537596385,
            "ratio": 0.9999797890583674,
            "v_loss": 9.06126606464386,
        },
        "state": "1c7df3a6b5bab0575796b650c68c3aea99cd817aa5f11b4e5a82a1da7c3a5521",
        "updates": 12,
    },
}


def train_window(case: str) -> dict[str, Any]:
    """One seeded ``learn()`` window of ``case``: ``train()``'s metrics, the
    number of minibatch gradient steps and the state digest."""
    build, module_names = CASES[case]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = build()
        recorded: list[dict[str, float]] = []
        steps: list[bool] = []
        train = agent.train
        optimizer_step = agent._chain_optimizer_step

        def recording_train() -> dict[str, float]:
            recorded.append(train())
            return recorded[-1]

        def recording_optimizer_step(loss: torch.Tensor, train_actor: bool) -> None:
            steps.append(train_actor)
            optimizer_step(loss, train_actor)

        agent.train = recording_train
        agent._chain_optimizer_step = recording_optimizer_step
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
        "updates": len(steps),
        "actor_updates": sum(steps),
        "state": state_dict_digest(state),
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_seeded_window_matches_the_recorded_snapshot(case):
    expected = SNAPSHOT[case]
    window = train_window(case)
    assert (window["updates"], window["actor_updates"]) == (
        expected["updates"],
        expected["actor_updates"],
    )
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


def test_the_recorded_cases_exercise_their_branches():
    """The warmup case never steps the actor; the target_kl case stops
    before the full ``update_epochs`` passes (it checks after each step).
    Checked on the recorded values."""
    full, warmup, target_kl = (
        SNAPSHOT["dppo"],
        SNAPSHOT["dppo-critic_warmup"],
        SNAPSHOT["dppo-target_kl"],
    )
    assert warmup["actor_updates"] == 0 < warmup["updates"] == full["updates"]
    assert 1 <= target_kl["updates"] < full["updates"]


def record() -> None:
    """Print every case's window, ready to paste over ``SNAPSHOT``."""
    pprint.pprint({case: train_window(case) for case in sorted(CASES)}, sort_dicts=True)


if __name__ == "__main__":
    record()
