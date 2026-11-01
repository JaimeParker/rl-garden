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

A second recording, made before the SAC step methods, ``_critic_features``,
``policy_class`` and ``_make_optimizers`` were extracted (a pure refactor),
pins the rest of the family: RLPD (bias excluded from weight decay, an LR
warmup and a prior-data buffer, high UTD), RLPDHybrid, CQL, CalQL (high UTD),
SACFlow, OfflineSAC, FlashSAC and SAC's high-UTD and fixed-alpha paths. Each
case runs two seeded ``train()`` calls, the first with ``compute_info=True``,
and pins the returned info as well as the parameters. RLPD's optimizer param
groups are pinned by parameter name.

Each module is summarized by the float64 sum and sum of squares of its
parameters. A different torch build or BLAS may round differently; re-record
the values there instead of loosening the tolerance.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import torch
from gymnasium import spaces
from test_cql_calql import _fill as fill_complete_trajectory
from test_cql_calql import _offline_env as cql_offline_env
from test_cql_calql import _offline_kwargs as cql_offline_kwargs
from test_flash_sac import _FakeEnv as FlashSACEnv
from test_flash_sac import _fill_buffer as fill_flash_sac_buffer
from test_rlpd import DummyVecEnv as StateVecEnv
from test_rlpd_hybrid import DummyVecEnv as HybridActionVecEnv
from test_sac_flow_core import DummyVecEnv as SpaceVecEnv
from test_sac_flow_core import _sac_flow_kwargs

from rl_garden.algorithms import (
    CQL,
    RLPD,
    SAC,
    CalQL,
    FlashSAC,
    OfflineEnvSpec,
    RecurrentSAC,
    SACFlow,
    TransformerSAC,
)
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
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


def build_dict_sac(encoder_sharing: str, **overrides: Any) -> SAC:
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
        **overrides,
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


SMALL_ONLINE_KWARGS: dict[str, Any] = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 64,
    "batch_size": 8,
    "learning_starts": 0,
    "training_freq": 1,
    "eval_freq": 0,
    "log_freq": 0,
    "std_log": False,
    "net_arch": [8],
    "seed": 0,
}


def fill_random_transitions(agent, buffer=None, transitions: int = 8) -> None:
    """Add seeded random transitions shaped by the agent's (Dict) observation
    space to ``buffer`` (the agent's replay buffer when omitted)."""
    buffer = agent.replay_buffer if buffer is None else buffer
    env = agent.env
    num_envs = buffer.num_envs

    def observation() -> dict[str, torch.Tensor]:
        return {
            key: (
                torch.randint(0, 256, (num_envs, *space.shape), dtype=torch.uint8)
                if space.dtype == np.uint8
                else torch.randn(num_envs, *space.shape)
            )
            for key, space in env.single_observation_space.spaces.items()
        }

    for _ in range(transitions):
        actions = torch.rand(num_envs, *env.single_action_space.shape) * 2 - 1
        buffer.add(
            observation(), observation(), actions, torch.randn(num_envs), torch.zeros(num_envs)
        )


def build_rlpd(**overrides: Any) -> RLPD:
    """RLPD with bias and norm parameters excluded from AdamW's weight decay."""
    kwargs = {
        **SMALL_ONLINE_KWARGS,
        "weight_decay": 0.1,
        "use_adamw": True,
        "exclude_bias_from_decay": True,
        **overrides,
    }
    return RLPD(env=StateVecEnv(), **kwargs)


def build_rlpd_with_prior_data() -> RLPD:
    agent = build_rlpd(lr_schedule="linear_warmup", lr_warmup_steps=4, utd=2)
    agent.offline_replay_buffer = agent._build_prior_data_buffer(32)
    agent.offline_data_ratio = 0.5
    fill_random_transitions(agent, agent.offline_replay_buffer)
    fill_random_transitions(agent)
    return agent


def build_rlpd_hybrid() -> RLPDHybrid:
    agent = RLPDHybrid(env=HybridActionVecEnv(), discrete_hidden_dim=8, **SMALL_ONLINE_KWARGS)
    fill_random_transitions(agent)
    return agent


def build_cql() -> CQL:
    agent = CQL(env=cql_offline_env(), seed=0, **cql_offline_kwargs())
    fill_complete_trajectory(agent)
    return agent


def build_calql_high_utd() -> CalQL:
    agent = CalQL(env=cql_offline_env(), seed=0, utd=2, **cql_offline_kwargs())
    fill_complete_trajectory(agent)
    return agent


def build_sac_flow() -> SACFlow:
    env = SpaceVecEnv(
        spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )
    agent = SACFlow(env=env, **{**_sac_flow_kwargs(), "learning_starts": 0, "seed": 0})
    fill_random_transitions(agent)
    return agent


def build_offline_sac() -> OfflineSAC:
    env = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        num_envs=2,
    )
    agent = OfflineSAC(
        env=env,
        buffer_size=64,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        net_arch=[16],
        seed=0,
    )
    fill_random_transitions(agent)
    return agent


def build_flash_sac() -> FlashSAC:
    agent = FlashSAC(
        env=FlashSACEnv(),
        eval_env=None,
        buffer_size=400,
        buffer_device="cpu",
        learning_starts=0,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=10,
        utd=1.0,
        n_step=3,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        num_bins=11,
        min_v=-2.0,
        max_v=2.0,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        seed=0,
        device="cpu",
        std_log=False,
        log_freq=0,
        eval_freq=0,
        save_final_checkpoint=False,
    )
    fill_flash_sac_buffer(agent, n_transitions=20)
    return agent


#: ``{case: (builder, gradient steps of the first and second train() call)}``.
#: With ``utd=2`` the first call is one high-UTD group and the second adds a
#: one-step remainder at ``utd=1``.
FAMILY_CASES: dict[str, tuple[Callable[[], Any], tuple[int, int]]] = {
    "rlpd_exclude_bias_prior_data_high_utd": (build_rlpd_with_prior_data, (2, 3)),
    "rlpd_hybrid": (build_rlpd_hybrid, (1, 2)),
    "cql": (build_cql, (1, 2)),
    "calql_high_utd": (build_calql_high_utd, (2, 3)),
    "sac_flow": (build_sac_flow, (1, 2)),
    "offline_sac": (build_offline_sac, (1, 2)),
    "flash_sac": (build_flash_sac, (1, 2)),
    "sac_shared_high_utd": (lambda: build_dict_sac("shared", utd=2), (2, 3)),
    "sac_fixed_alpha_delayed_actor": (
        lambda: build_dict_sac("shared_critic_grad", ent_coef=0.2, policy_frequency=2),
        (1, 3),
    ),
}


def family_digests(case: str) -> dict[str, Any]:
    """The info of a first ``train(compute_info=True)`` call and the policy
    digest after it and after a second ``train()`` call."""
    builder, (first_steps, second_steps) = FAMILY_CASES[case]

    def run():
        agent = builder()
        info = agent.train(first_steps, compute_info=True)
        first = module_digest(agent.policy)
        agent.train(second_steps)
        return {"info": info, "first": first, "second": module_digest(agent.policy)}

    return run_single_threaded(run)


def optimizer_group_layout(
    policy: torch.nn.Module, q_optimizer, actor_optimizer
) -> dict[str, list[tuple[str, float, float, tuple[str, ...]]]]:
    """``{optimizer: [(class, weight_decay, lr, parameter names), ...]}``, one
    entry per param group, parameters named as in ``policy``."""
    names = {id(param): name for name, param in policy.named_parameters()}
    return {
        optimizer_name: [
            (
                type(optimizer).__name__,
                group["weight_decay"],
                group["lr"],
                tuple(names[id(p)] for p in group["params"]),
            )
            for group in optimizer.param_groups
        ]
        for optimizer_name, optimizer in (
            ("q_optimizer", q_optimizer),
            ("actor_optimizer", actor_optimizer),
        )
    }


def rlpd_optimizer_group_layout(**overrides: Any):
    agent = build_rlpd(**overrides)
    return optimizer_group_layout(agent.policy, agent.q_optimizer, agent.actor_optimizer)


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


# Recorded before the SAC step methods, _critic_features, policy_class and
# _make_optimizers were extracted.
FAMILY_SNAPSHOT: dict[str, dict[str, dict[str, Any]]] = {
    "calql_high_utd": {
        "first": {
            "actor": (11.80224750321213, 24.206770715924524),
            "critic": (62.14777698589023, 90.56162804391494),
            "critic_target": (62.18462135500624, 90.58118981618557),
        },
        "info": {
            "action_saturation": 0.522148609161377,
            "actor_loss": -0.017709527164697647,
            "alpha": 0.9999367594718933,
            "alpha_loss": 2.6215357780456543,
            "calql_bound_rate": 0.0833333358168602,
            "cql_alpha": 1.0,
            "cql_loss": 3.2104129791259766,
            "cql_ood_values": 3.546128749847412,
            "cql_q_diff": 3.2104129791259766,
            "critic_loss": 5.09021520614624,
            "entropy": 0.6215355396270752,
            "entropy_gaussian": 2.2819201946258545,
            "policy_std": 0.5698824524879456,
            "predicted_q": 0.05474846065044403,
            "qf_loss": 5.09021520614624,
            "tanh_correction": -1.2356963157653809,
            "target_entropy": -2.0,
            "target_q": -0.8435779213905334,
            "td_loss": 1.8798022270202637,
            "utd_ratio": 2.0,
        },
        "second": {
            "actor": (11.809224746451946, 24.20337723787153),
            "critic": (62.104169058369735, 90.53793601633842),
            "critic_target": (62.18362943513165, 90.58065077738523),
        },
    },
    "cql": {
        "first": {
            "actor": (11.801847578113666, 24.206167488948612),
            "critic": (62.170508214621805, 90.57728995340136),
            "critic_target": (62.184836641496645, 90.58132785182808),
        },
        "info": {
            "action_saturation": 0.5466126799583435,
            "actor_loss": -0.06510099768638611,
            "alpha": 0.9999367594718933,
            "alpha_loss": 2.5568161010742188,
            "cql_alpha": 1.0,
            "cql_loss": 3.281921625137329,
            "cql_ood_values": 2.892681121826172,
            "cql_q_diff": 3.281921625137329,
            "critic_loss": 5.00361442565918,
            "entropy": 0.5568159818649292,
            "entropy_gaussian": 2.3201324939727783,
            "policy_std": 0.8030503988265991,
            "predicted_q": 0.0935913473367691,
            "qf_loss": 5.00361442565918,
            "tanh_correction": -1.9388846158981323,
            "target_entropy": -2.0,
            "target_q": -0.777753472328186,
            "td_loss": 1.7216929197311401,
        },
        "second": {
            "actor": (11.807087105280516, 24.20175795448548),
            "critic": (62.13353450050636, 90.56516378124587),
            "critic_target": (62.1844149284359, 90.58119699048177),
        },
    },
    "flash_sac": {
        "first": {
            "actor": (220.8810133574334, 418.00000029519504),
            "critic": (796.4385362340022, 1612.000155077198),
            "critic_target": (796.4922082870007, 1611.9996877551566),
        },
        "info": {
            "actor/loss": 0.7056969404220581,
            "critic/loss": 2.7738237380981445,
            "temperature/value": 0.009999999776482582,
        },
        "second": {
            "actor": (220.70795599549638, 418.0000513888681),
            "critic": (795.9991216673848, 1612.000303452342),
            "critic_target": (796.4886857122381, 1611.998577704991),
        },
    },
    "offline_sac": {
        "first": {
            "actor": (1.9525333776837215, 8.427742427015607),
            "critic": (-2.683345486962935, 12.051004836445355),
            "critic_target": (-2.6740918605937622, 12.052947198647724),
        },
        "info": {
            "action_saturation": 0.1386348307132721,
            "actor_loss": -0.30982324481010437,
            "alpha": 0.9997000694274902,
            "alpha_loss": 3.283658742904663,
            "critic_loss": 30.425310134887695,
            "entropy": 0.2836589813232422,
            "entropy_gaussian": 0.03783988952636719,
            "policy_std": 0.4532300531864166,
            "predicted_q": 0.15816576778888702,
            "qf_loss": 30.425310134887695,
            "tanh_correction": -0.4108767807483673,
            "target_entropy": -3.0,
            "target_q": -2.385256290435791,
            "td_loss": 30.425310134887695,
        },
        "second": {
            "actor": (1.9624684613081627, 8.423250564636335),
            "critic": (-2.701624539520708, 12.04666579204227),
            "critic_target": (-2.674318644509185, 12.052895891319071),
        },
    },
    "rlpd_exclude_bias_prior_data_high_utd": {
        "first": {
            "actor": (0.9416147631127387, 4.674164667315874),
            "critic": (77.54406762262965, 116.12541799201878),
            "critic_target": (77.54051369670718, 116.12780880465994),
        },
        "info": {
            "action_saturation": 0.14508363604545593,
            "actor_loss": 0.09225122630596161,
            "alpha": 0.9997000694274902,
            "alpha_loss": 2.6428003311157227,
            "critic_loss": 21.89041519165039,
            "entropy": 0.6428004503250122,
            "entropy_gaussian": 2.8371362686157227,
            "policy_std": 1.070538878440857,
            "predicted_q": -0.015511012636125088,
            "qf_loss": 21.89041519165039,
            "tanh_correction": -1.716605544090271,
            "target_entropy": -2.0,
            "target_q": 0.59965980052948,
            "td_loss": 21.89041519165039,
            "utd_ratio": 2.0,
        },
        "second": {
            "actor": (0.9428417915478349, 4.672542816474007),
            "critic": (77.57520684886458, 116.10274363034539),
            "critic_target": (77.54113491912568, 116.12731567609123),
        },
    },
    "rlpd_hybrid": {
        "first": {
            "actor": (0.9422146442811936, 4.670139483098242),
            "critic": (77.50987790231011, 116.09754548230099),
            "critic_target": (77.54017122772939, 116.12752820806503),
            "discrete_critic": (8.660569957777625, 12.400288000722597),
            "discrete_target_critic": (8.658490649001806, 12.399952677844176),
        },
        "info": {
            "action_saturation": 0.24989044666290283,
            "actor_loss": -0.4359511137008667,
            "alpha": 0.9997000694274902,
            "alpha_loss": 4.168850898742676,
            "critic_loss": 6.514853000640869,
            "entropy": 1.1688504219055176,
            "entropy_gaussian": 2.630980968475342,
            "policy_std": 1.0260313749313354,
            "predicted_q": -0.02451116405427456,
            "qf_loss": 6.514853000640869,
            "tanh_correction": -1.7672852277755737,
            "target_entropy": -3.0,
            "target_q": -0.1710195243358612,
            "td_loss": 6.514853000640869,
        },
        "second": {
            "actor": (0.9342912703286856, 4.662339772472391),
            "critic": (77.52796027017757, 116.07959849129021),
            "critic_target": (77.53986756301993, 116.12666914661683),
            "discrete_critic": (8.662491323710128, 12.395004971277466),
            "discrete_target_critic": (8.658562745222753, 12.399879270771645),
        },
    },
    "sac_fixed_alpha_delayed_actor": {
        "first": {
            "actor": (-0.18383009164244868, 7.081365273876708),
            "actor_extractor": (3.0238033493967578, 94.51895011560671),
            "critic": (1.0424453987070592, 11.975596316267575),
            "critic_target": (1.0418518759979634, 11.975961578774148),
        },
        "info": {
            "actor_loss": 0.0,
            "alpha": 0.20000000298023224,
            "critic_loss": 0.41372716426849365,
            "predicted_q": 0.05881960690021515,
            "qf_loss": 0.41372716426849365,
            "target_q": 0.05541457235813141,
            "td_loss": 0.41372716426849365,
        },
        "second": {
            "actor": (-0.18598265197942965, 7.079388163499562),
            "actor_extractor": (4.679566459375792, 94.54021894034611),
            "critic": (1.0330029281612951, 11.971487096768042),
            "critic_target": (1.0417449087108253, 11.975869610139453),
        },
    },
    "sac_flow": {
        "first": {
            "actor": (1.544425372616388, 12.703588446740255),
            "critic": (-4.357990785036236, 13.023318156199277),
            "critic_target": (-4.369869412155822, 13.026701852971614),
        },
        "info": {
            "action_norm": 0.8812612295150757,
            "actor_loss": 1.6391586065292358,
            "alpha": 0.9997000694274902,
            "alpha_loss": 0.6158270835876465,
            "critic_loss": 31.411060333251953,
            "entropy": -1.3841729164123535,
            "log_prob_mean": 1.167405366897583,
            "log_prob_std": 3.71924090385437,
            "predicted_q": -0.23195010423660278,
            "qf_loss": 31.411060333251953,
            "target_entropy": -2.0,
            "target_q": 3.108680248260498,
            "td_loss": 31.411060333251953,
        },
        "second": {
            "actor": (1.5461287736106897, 12.693987072951604),
            "critic": (-4.346588270913344, 13.019503343307857),
            "critic_target": (-4.369473624974489, 13.026572492265851),
        },
    },
    "sac_shared_high_utd": {
        "first": {
            "actor": (-0.1895299313182477, 7.079942003266544),
            "actor_extractor": (-1.434625088873645, 94.51957521049728),
            "critic": (1.0865835445729317, 11.978892320550564),
            "critic_target": (1.0425489166082116, 11.976010701796657),
        },
        "info": {
            "action_saturation": 0.16433022916316986,
            "actor_loss": -0.8340271711349487,
            "alpha": 0.9997000694274902,
            "alpha_loss": 3.0005197525024414,
            "critic_loss": 1.6394767761230469,
            "entropy": 1.0005196332931519,
            "entropy_gaussian": 2.6713242530822754,
            "policy_std": 1.1898561716079712,
            "predicted_q": 0.05939166992902756,
            "qf_loss": 1.6394767761230469,
            "tanh_correction": -0.9045696258544922,
            "target_entropy": -2.0,
            "target_q": 0.9275203943252563,
            "td_loss": 1.6394767761230469,
            "utd_ratio": 2.0,
        },
        "second": {
            "actor": (-0.20152669106028043, 7.077323171088531),
            "actor_extractor": (-0.8698755297280059, 94.5646437874848),
            "critic": (1.1217508403933607, 11.982345107863125),
            "critic_target": (1.0445774842082756, 11.976149889884503),
        },
    },
}

# RLPD's optimizers under exclude_bias_from_decay=True and weight_decay > 0,
# recorded at the same commit. The ensemble critic stacks its biases along the
# member axis, so none is 1-D and its no-decay group is empty.
RLPD_EXCLUDE_BIAS_OPTIMIZER_GROUPS: dict[str, list[tuple[str, float, float, tuple[str, ...]]]] = {
    "q_optimizer": [
        (
            "AdamW",
            0.1,
            0.0003,
            (
                "critic.ens_p_trunk__0__weight",
                "critic.ens_p_trunk__0__bias",
                "critic.ens_p_trunk__1__weight",
                "critic.ens_p_trunk__1__bias",
                "critic.ens_p_head__weight",
                "critic.ens_p_head__bias",
            ),
        ),
        ("AdamW", 0.0, 0.0003, ()),
    ],
    "actor_optimizer": [
        (
            "AdamW",
            0.1,
            0.0003,
            ("actor.trunk.0.weight", "actor.fc_mean.weight", "actor.fc_logstd.weight"),
        ),
        (
            "AdamW",
            0.0,
            0.0003,
            ("actor.trunk.0.bias", "actor.fc_mean.bias", "actor.fc_logstd.bias"),
        ),
    ],
}


@requires_snapshot_build
@pytest.mark.parametrize("case", sorted(FAMILY_CASES))
def test_sac_family_matches_recorded_run(case):
    assert family_digests(case) == FAMILY_SNAPSHOT[case]


def test_rlpd_exclude_bias_optimizer_groups_match_recording():
    assert run_single_threaded(rlpd_optimizer_group_layout) == RLPD_EXCLUDE_BIAS_OPTIMIZER_GROUPS
