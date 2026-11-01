"""Seeded ``train()`` metrics of the PPO family pinned to exact values.

The values were recorded on CPU (torch 2.6, one thread) before the per-class
minibatch evaluation was consolidated into one ``_evaluate_minibatch``
returning ``MinibatchEval``; the refactor must reproduce them bit for bit.
Every evaluation branch is covered: plain and ``adaptive_kl``, each with and
without a per-component decision mask, for ``PPO`` and ``RecurrentPPO``,
plus ``TransformerPPO``. A different torch build or BLAS may round
differently; re-record the values there instead of loosening the tolerance.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from _decision_fakes import NUM_STEPS, _ComponentMaskVecEnv, _FixedHoldVecEnv

from rl_garden.algorithms import PPO, RecurrentPPO, TransformerPPO

TRANSFORMER_KWARGS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_transformer_layers": 1,
    "mlp_num": 1,
    "memory_len": 2,
}
AGENTS = {
    "ppo": (PPO, {}),
    "gru": (RecurrentPPO, {"rnn_hidden_size": 8}),
    "gtrxl": (TransformerPPO, TRANSFORMER_KWARGS),
}
CASES = {
    "ppo": ("ppo", "constant", False),
    "ppo-adaptive_kl": ("ppo", "adaptive_kl", False),
    "ppo-components": ("ppo", "constant", True),
    "ppo-components-adaptive_kl": ("ppo", "adaptive_kl", True),
    "gru": ("gru", "constant", False),
    "gru-adaptive_kl": ("gru", "adaptive_kl", False),
    "gru-components": ("gru", "constant", True),
    "gru-components-adaptive_kl": ("gru", "adaptive_kl", True),
    "gtrxl": ("gtrxl", "constant", False),
}

#: The build the values below were recorded on (CPU only).
SNAPSHOT_TORCH_VERSION = "2.6.0+cu124"
pytestmark = pytest.mark.skipif(
    torch.__version__ != SNAPSHOT_TORCH_VERSION,
    reason=(
        f"snapshot recorded on torch {SNAPSHOT_TORCH_VERSION} CPU; re-record on other "
        "builds, see module docstring"
    ),
)

SNAPSHOT: dict[str, dict[str, float]] = {
    "gru": {
        "approx_kl": 1.0542571828864311e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "entropy": 1.838720977306366,
        "explained_variance": 0.003742992877960205,
        "learning_rate": 0.0003,
        "loss": 4.097128927707672,
        "old_approx_kl": 0.0002160656016105822,
        "policy_loss": -7.153053775255103e-05,
        "value_loss": 8.194400906562805,
    },
    "gru-adaptive_kl": {
        "approx_kl": 2.579142730496642e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "entropy": 1.8390075862407684,
        "explained_variance": 0.003742992877960205,
        "learning_rate": 0.0010125,
        "loss": 4.096173405647278,
        "old_approx_kl": 0.000204246493770599,
        "policy_loss": -8.94169024832081e-05,
        "value_loss": 8.192525625228882,
    },
    "gru-components": {
        "approx_kl": 6.7749579102382995e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "decisions_per_minibatch_mean": 17.0,
        "decisions_per_minibatch_min": 16.0,
        "entropy": 1.5378482937812805,
        "explained_variance": -0.0013949871063232422,
        "learning_rate": 0.0003,
        "loss": 5.309546709060669,
        "old_approx_kl": 5.872175097465515e-05,
        "policy_loss": -0.0008293738873277334,
        "value_loss": 10.620752334594727,
    },
    "gru-components-adaptive_kl": {
        "approx_kl": 1.4378327250597067e-05,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "decisions_per_minibatch_mean": 17.0,
        "decisions_per_minibatch_min": 16.0,
        "entropy": 1.5378399789333344,
        "explained_variance": -0.0013949871063232422,
        "learning_rate": 0.0010125,
        "loss": 5.307170808315277,
        "old_approx_kl": 8.584698662161827e-05,
        "policy_loss": -0.0011566880210835961,
        "value_loss": 10.616654992103577,
    },
    "gtrxl": {
        "approx_kl": 3.4707287710489254e-07,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "entropy": 1.837279349565506,
        "explained_variance": -0.0004922151565551758,
        "learning_rate": 0.0003,
        "loss": 4.256917595863342,
        "old_approx_kl": -0.00018701826532685484,
        "policy_loss": 2.6079510765386615e-05,
        "value_loss": 8.51378309726715,
    },
    "ppo": {
        "approx_kl": 3.5023936959532875e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "entropy": 1.8378453254699707,
        "explained_variance": 0.017375409603118896,
        "learning_rate": 0.0003,
        "loss": 4.575691878795624,
        "old_approx_kl": 0.0001916848183706743,
        "policy_loss": -0.0004286629361255345,
        "value_loss": 9.152240872383118,
    },
    "ppo-adaptive_kl": {
        "approx_kl": 6.289531796710435e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "entropy": 1.8378240764141083,
        "explained_variance": 0.017375409603118896,
        "learning_rate": 0.0010125,
        "loss": 4.573736608028412,
        "old_approx_kl": 0.00027258507422328604,
        "policy_loss": -0.0005273508528867987,
        "value_loss": 9.148527979850769,
    },
    "ppo-components": {
        "approx_kl": 5.335857451882475e-06,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "decisions_per_minibatch_mean": 17.0,
        "decisions_per_minibatch_min": 16.0,
        "entropy": 1.5441223680973053,
        "explained_variance": 0.011366784572601318,
        "learning_rate": 0.0003,
        "loss": 5.425827145576477,
        "old_approx_kl": 0.0002932331246686126,
        "policy_loss": -0.0005668830094185395,
        "value_loss": 10.852787971496582,
    },
    "ppo-components-adaptive_kl": {
        "approx_kl": 1.1571217328310013e-05,
        "clip_coef": 0.2,
        "clipfrac": 0.0,
        "decisions_per_minibatch_mean": 17.0,
        "decisions_per_minibatch_min": 16.0,
        "entropy": 1.544016033411026,
        "explained_variance": 0.011366784572601318,
        "learning_rate": 0.0010125,
        "loss": 5.424291610717773,
        "old_approx_kl": 0.00042525197042980745,
        "policy_loss": -0.00084662623380094,
        "value_loss": 10.850276470184326,
    },
}


def train_metrics(case: str) -> dict[str, float]:
    """The metrics dict ``train()`` returns at the end of one seeded window."""
    agent_name, lr_schedule, components = CASES[case]
    agent_cls, extra_kwargs = AGENTS[agent_name]
    env = _ComponentMaskVecEnv() if components else _FixedHoldVecEnv(emit_decisions=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = agent_cls(
            env,
            device="cpu",
            num_steps=NUM_STEPS,
            num_minibatches=2,
            update_epochs=2,
            eval_freq=0,
            log_freq=0,
            std_log=False,
            target_kl=None,
            net_arch=[8],
            seed=0,
            lr_schedule=lr_schedule,
            decision_level=components,
            **extra_kwargs,
        )
        train = agent.train
        recorded: list[dict[str, float]] = []

        def recording_train() -> dict[str, float]:
            recorded.append(train())
            return recorded[-1]

        agent.train = recording_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=agent.num_envs * agent.num_steps)
    finally:
        torch.set_num_threads(threads)
    assert len(recorded) == 1
    return recorded[0]


@pytest.mark.parametrize("case", sorted(CASES))
def test_seeded_train_metrics_match_the_recorded_snapshot(case):
    metrics = train_metrics(case)
    expected = SNAPSHOT[case]
    assert metrics.keys() == expected.keys()
    for key, value in expected.items():
        if math.isnan(value):
            assert math.isnan(metrics[key]), key
            continue
        torch.testing.assert_close(
            torch.tensor(metrics[key], dtype=torch.float64),
            torch.tensor(value, dtype=torch.float64),
            rtol=0,
            atol=0,
            msg=f"{case}: {key}",
        )
