"""Seeded SAC-family parameter values pinned to a recording made before the
optimizer-ownership fixes (sequence trunk on ``q_optimizer``; shared extractor
also on ``actor_optimizer`` under ``encoder_sharing="shared"``).

The values were recorded on CPU (torch 2.6.0+cu124, one thread). They pin what
those fixes must leave untouched:

- Sequence SAC (GRU, LSTM, GTrXL) with its sequence trunk frozen reproduces the
  recorded run exactly: before the fix the trunk was stepped by no optimizer,
  so freezing it now isolates the trunk's own trajectory as the only change.
- SAC under ``"shared_critic_grad"`` and ``"separate"`` reproduces the recorded
  run exactly.

Each module is summarized by the float64 sum and sum of squares of its
parameters. A different torch build or BLAS may round differently; re-record
the values there instead of loosening the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC, RecurrentSAC, TransformerSAC
from rl_garden.encoders.config import EncoderConfig

#: The build the values below were recorded on (CPU only).
SNAPSHOT_TORCH_VERSION = "2.6.0+cu124"
requires_snapshot_build = pytest.mark.skipif(
    torch.__version__ != SNAPSHOT_TORCH_VERSION,
    reason=(
        f"snapshot recorded on torch {SNAPSHOT_TORCH_VERSION} CPU; re-record on other "
        "builds, see module docstring"
    ),
)

# "gap" pooling tolerates the tiny 64x64 test image (see test_recurrent_sac.py).
ENCODER_CONFIG = EncoderConfig(features_dim=16, plain_conv_pooling="gap", proprio_latent_dim=4)

SEQUENCE_CASES = {
    "recurrent_gru": (RecurrentSAC, {"rnn_type": "gru", "rnn_hidden_size": 8}),
    "recurrent_lstm": (RecurrentSAC, {"rnn_type": "lstm", "rnn_hidden_size": 8}),
    "transformer": (
        TransformerSAC,
        {
            "embed_dim": 8,
            "head_dim": 4,
            "num_heads": 2,
            "num_transformer_layers": 1,
            "memory_len": 2,
        },
    ),
}


class DictObsVecEnv:
    """Two-env image+state env whose observations come from the global torch RNG."""

    num_envs = 2

    def __init__(self) -> None:
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (self.num_envs, 2)),
            high=np.broadcast_to(self.single_action_space.high, (self.num_envs, 2)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return self.observation(), {}

    def step(self, actions):
        del actions
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return self.observation(), rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def observation(self) -> dict[str, torch.Tensor]:
        return {
            "rgb_cam": torch.randint(0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(self.num_envs, 4),
        }


def module_digest(policy: torch.nn.Module) -> dict[str, tuple[float, float]]:
    """``{child_name: (sum, sum_of_squares)}`` over each parameterized child module."""
    digest: dict[str, tuple[float, float]] = {}
    for name, module in policy.named_children():
        params = [p.detach().reshape(-1).double() for p in module.parameters()]
        if not params:
            continue
        flat = torch.cat(params)
        digest[name] = (float(flat.sum()), float(flat.square().sum()))
    return digest


def run_single_threaded(fn):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return fn()
    finally:
        torch.set_num_threads(threads)


def build_sequence_sac(case: str):
    algorithm_cls, extra_kwargs = SEQUENCE_CASES[case]
    return algorithm_cls(
        env=DictObsVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=10,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        std_log=False,
        net_arch=[16],
        burn_in_len=2,
        learning_len=2,
        forward_len=1,
        encoder_config=ENCODER_CONFIG,
        seed=0,
        **extra_kwargs,
    )


def sequence_sac_digest(case: str, *, freeze_trunk: bool) -> dict[str, tuple[float, float]]:
    """Policy digest after a seeded 40-step ``learn()`` (several updates)."""

    def run():
        agent = build_sequence_sac(case)
        if freeze_trunk:
            for param in agent.policy.recurrent_encoder.parameters():
                param.requires_grad_(False)
        agent.learn(total_timesteps=40)
        return module_digest(agent.policy)

    return run_single_threaded(run)


def build_dict_sac(encoder_sharing: str) -> SAC:
    agent = SAC(
        env=DictObsVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        log_freq=0,
        std_log=False,
        net_arch={"pi": [16], "qf": [16]},
        encoder_config=ENCODER_CONFIG,
        encoder_sharing=encoder_sharing,
        seed=0,
    )
    for _ in range(8):
        obs = agent.env.observation()
        next_obs = agent.env.observation()
        actions = torch.rand(agent.num_envs, 2) * 2 - 1
        agent.replay_buffer.add(
            obs, next_obs, actions, torch.randn(agent.num_envs), torch.zeros(agent.num_envs)
        )
    return agent


def dict_sac_digests(encoder_sharing: str) -> dict[int, dict[str, tuple[float, float]]]:
    """Policy digests after 1 and 3 seeded ``train()`` updates."""

    def run():
        agent = build_dict_sac(encoder_sharing)
        agent.train(gradient_steps=1)
        after_one = module_digest(agent.policy)
        agent.train(gradient_steps=2)
        return {1: after_one, 3: module_digest(agent.policy)}

    return run_single_threaded(run)


# Recorded with ``freeze_trunk=False`` at the pre-fix commit (where freezing
# the trunk made no difference: no optimizer stepped it).
SEQUENCE_SNAPSHOT: dict[str, dict[str, tuple[float, float]]] = {
    "recurrent_gru": {
        "actor": (-0.381474374094978, 6.937358114305656),
        "actor_extractor": (29.502477833106013, 94.86034884975405),
        "critic": (3.0927228393265978, 13.059209539622607),
        "critic_target": (2.978650687262416, 13.083046108631939),
        "recurrent_encoder": (1.2423858494439628, 29.852167164422923),
    },
    "recurrent_lstm": {
        "actor": (-0.5244752033613622, 7.8313934843532005),
        "actor_extractor": (28.617532920313124, 94.82503905305259),
        "critic": (2.3432669585454278, 13.130921405893629),
        "critic_target": (2.2805274858837947, 13.107951786256383),
        "recurrent_encoder": (3.730789840890793, 40.360332885501215),
    },
    "transformer": {
        "actor": (-0.9532658503740095, 7.150439701147091),
        "actor_extractor": (77.07433391976451, 95.2117636947306),
        "critic": (1.409211655438412, 12.344771781587767),
        "critic_target": (1.2540900961103034, 12.338573574918083),
        "recurrent_encoder": (52.18729507320677, 137.12575441751443),
    },
}

# Keyed by the number of ``train()`` updates. At the pre-fix commit
# ``encoder_sharing="shared"`` reproduced the "shared_critic_grad" values
# exactly: the actor loss's extractor gradient was never applied.
SAC_SNAPSHOT: dict[str, dict[int, dict[str, tuple[float, float]]]] = {
    "shared_critic_grad": {
        1: {
            "actor": (-0.19102977510192432, 7.080043945635417),
            "actor_extractor": (2.060633542061396, 94.51744420638532),
            "critic": (1.0640464433963643, 11.977444795623843),
            "critic_target": (1.042067997783306, 11.975980044828882),
        },
        3: {
            "actor": (-0.20516118162777275, 7.077597938554527),
            "actor_extractor": (4.983785180581549, 94.55093052405776),
            "critic": (1.090967813739553, 11.979248812787844),
            "critic_target": (1.0429080404283013, 11.976031231495977),
        },
    },
    "separate": {
        1: {
            "actor": (-0.9686795542365871, 7.316179026162065),
            "actor_extractor": (3.53804343679235, 94.5215363260979),
            "critic": (2.779237331575132, 11.801445321995644),
            "critic_extractor": (23.78250620652068, 94.71595031038629),
            "critic_target": (2.792304418850108, 11.803746199145312),
        },
        3: {
            "actor": (-0.9816950829117559, 7.315311368975754),
            "actor_extractor": (5.942066211718384, 94.54861114817388),
            "critic": (2.7560118084802525, 11.799004454611545),
            "critic_extractor": (28.980258972247043, 94.74889006301689),
            "critic_target": (2.7916572860413, 11.803652450846442),
        },
    },
}


@requires_snapshot_build
@pytest.mark.parametrize("case", sorted(SEQUENCE_CASES))
def test_sequence_sac_with_frozen_trunk_matches_recorded_run(case):
    assert sequence_sac_digest(case, freeze_trunk=True) == SEQUENCE_SNAPSHOT[case]


@requires_snapshot_build
@pytest.mark.parametrize("case", sorted(SEQUENCE_CASES))
def test_sequence_sac_trunk_now_leaves_its_recorded_values(case):
    digest = sequence_sac_digest(case, freeze_trunk=False)
    assert digest["recurrent_encoder"] != SEQUENCE_SNAPSHOT[case]["recurrent_encoder"]


@requires_snapshot_build
@pytest.mark.parametrize("encoder_sharing", sorted(SAC_SNAPSHOT))
def test_dict_sac_matches_recorded_run(encoder_sharing):
    assert dict_sac_digests(encoder_sharing) == SAC_SNAPSHOT[encoder_sharing]
