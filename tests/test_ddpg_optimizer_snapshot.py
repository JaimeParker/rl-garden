"""Seeded DDPG / TD3 / TD3+BC-family parameter values pinned to a recording
made before ``encoder_sharing="shared"`` put the shared extractor on the
actor optimizer too.

The values were recorded on CPU (torch 2.6.0+cu124, one thread). They pin what
that fix must leave untouched: every family member (DDPG, TD3, TD3BC, ReBRAC,
SPOT, Off2OnSPOT) under ``"shared_critic_grad"`` and ``"separate"`` reproduces
the recorded run exactly. The pre-fix ``"shared"`` run is recorded as well, as
the reference ``tests/test_ddpg_encoder_sharing.py`` compares the fixed
``"shared"`` run against.

Each case trains on image + state observations (so the extractor has
parameters to share) in two seeded ``train()`` calls: the first ends with the
first actor update (``policy_freq`` gradient steps), the second adds two more
gradient steps. Each module is summarized as in
``tests/test_sac_optimizer_snapshot.py``; a different torch build or BLAS may
round differently, so re-record the values there instead of loosening the
tolerance.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from test_sac_optimizer_snapshot import (
    module_digest,
    requires_snapshot_build,
    run_single_threaded,
)

from rl_garden.algorithms import (
    DDPG,
    SPOT,
    TD3,
    TD3BC,
    Off2OnSPOT,
    OfflineEnvSpec,
    ReBRAC,
)
from rl_garden.encoders.config import EncoderConfig

STATE_DIM = 4
ACTION_DIM = 2
NUM_ENVS = 2
DDPG_IMAGE_SHAPE = (32, 40, 3)
TD3BC_IMAGE_SHAPE = (64, 64, 3)
DDPG_ENCODER_CONFIG = EncoderConfig(
    backbone="drqv2_conv", proprio_latent_dim=4, image_augmentation="none"
)
# "gap" pooling tolerates the tiny 64x64 test image (see test_recurrent_sac.py).
TD3BC_ENCODER_CONFIG = EncoderConfig(
    features_dim=16, plain_conv_pooling="gap", proprio_latent_dim=4
)


def image_state_space(image_shape: tuple[int, int, int]) -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb_cam": spaces.Box(0, 255, image_shape, np.uint8),
            "state": spaces.Box(-1.0, 1.0, (STATE_DIM,), np.float32),
        }
    )


class ImageStateVecEnv:
    """Spaces-only vector env for the online/off2on members; training reads the
    replay buffer only, so no ``reset``/``step`` is needed."""

    num_envs = NUM_ENVS

    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        self.single_observation_space = image_state_space(image_shape)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)


def offline_image_env() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        image_state_space(TD3BC_IMAGE_SHAPE),
        spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
        num_envs=NUM_ENVS,
    )


def random_observation(space: spaces.Dict) -> dict[str, torch.Tensor]:
    return {
        "rgb_cam": torch.randint(
            0, 256, (NUM_ENVS, *space["rgb_cam"].shape), dtype=torch.uint8
        ),
        "state": torch.randn(NUM_ENVS, STATE_DIM),
    }


def fill_transitions(agent, transitions: int = 8, **step_kwargs: torch.Tensor) -> None:
    space = agent.env.single_observation_space
    for _ in range(transitions):
        agent.replay_buffer.add(
            random_observation(space),
            random_observation(space),
            torch.rand(NUM_ENVS, ACTION_DIM) * 2 - 1,
            torch.randn(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            **step_kwargs,
        )


ONLINE_KWARGS: dict[str, Any] = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 32,
    "batch_size": 4,
    "learning_starts": 0,
    "training_freq": 1,
    "eval_freq": 0,
    "log_freq": 0,
    "std_log": False,
    "seed": 0,
}
OFFLINE_KWARGS: dict[str, Any] = {
    "device": "cpu",
    "buffer_device": "cpu",
    "buffer_size": 32,
    "batch_size": 4,
    "std_log": False,
    "log_freq": 0,
    "net_arch": [16],
    "encoder_config": TD3BC_ENCODER_CONFIG,
    "seed": 0,
}


def build_ddpg_like(algorithm_cls: type, encoder_sharing: str):
    agent = algorithm_cls(
        env=ImageStateVecEnv(DDPG_IMAGE_SHAPE),
        feature_dim=8,
        hidden_dim=16,
        encoder_config=DDPG_ENCODER_CONFIG,
        encoder_sharing=encoder_sharing,
        **ONLINE_KWARGS,
    )
    fill_transitions(agent, episode_end=torch.zeros(NUM_ENVS, dtype=torch.bool))
    return agent


def build_offline(algorithm_cls: type, encoder_sharing: str, **extra: Any):
    agent = algorithm_cls(
        env=offline_image_env(), encoder_sharing=encoder_sharing, **OFFLINE_KWARGS, **extra
    )
    fill_transitions(agent)
    return agent


def build_off2on_spot(encoder_sharing: str) -> Off2OnSPOT:
    agent = Off2OnSPOT(
        env=ImageStateVecEnv(TD3BC_IMAGE_SHAPE),
        encoder_sharing=encoder_sharing,
        vae_hidden_dim=16,
        vae_iterations=0,
        **{**ONLINE_KWARGS, **OFFLINE_KWARGS},
    )
    fill_transitions(agent)
    return agent


#: ``{case: (builder(encoder_sharing), gradient steps of the first train() call)}``.
#: The first call ends with the first actor update: DDPG updates the actor on
#: every gradient step, the rest every ``policy_freq=2`` steps.
FAMILY_CASES: dict[str, tuple[Callable[[str], Any], int]] = {
    "ddpg": (lambda sharing: build_ddpg_like(DDPG, sharing), 1),
    "td3": (lambda sharing: build_ddpg_like(TD3, sharing), 2),
    "td3_bc": (lambda sharing: build_offline(TD3BC, sharing), 2),
    "rebrac": (lambda sharing: build_offline(ReBRAC, sharing), 2),
    "spot": (lambda sharing: build_offline(SPOT, sharing, vae_hidden_dim=16), 2),
    "off2on_spot": (build_off2on_spot, 2),
}
SECOND_CALL_STEPS = 2


def family_digests(case: str, encoder_sharing: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Policy digests after the first ``train()`` call (ending with the first
    actor update) and after a second one."""
    builder, first_steps = FAMILY_CASES[case]

    def run():
        agent = builder(encoder_sharing)
        agent.train(first_steps)
        first = module_digest(agent.policy)
        agent.train(SECOND_CALL_STEPS)
        return {"first": first, "second": module_digest(agent.policy)}

    return run_single_threaded(run)


#: ``{case: {encoder_sharing: {"first" | "second": digest}}}``, recorded before
#: the fix; "shared" is the pre-fix run.
SNAPSHOT: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]] = {
    "ddpg": {
        "separate": {
            "first": {
                "actor": (2.357398054090069, 41.99939332049341),
                "actor_extractor": (-6.751139422867709, 252.2621448433191),
                "critic": (-10.929139945671295, 69.99983050097052),
                "critic_extractor": (-19.518183217202022, 253.3240910231481),
                "critic_target": (-9.953311377517366, 69.99999820635175),
            },
            "second": {
                "actor": (2.6199712868547493, 41.9998569868026),
                "actor_extractor": (-6.473756180837881, 252.263414986301),
                "critic": (-11.683705093368872, 70.00091739508684),
                "critic_extractor": (-19.66291322725192, 253.32294734804154),
                "critic_target": (-9.985041587025364, 69.99999769449026),
            },
        },
        "shared": {
            "first": {
                "actor": (18.343336424554536, 42.000169636755736),
                "actor_extractor": (-6.701845281681017, 252.26812527172683),
                "critic": (1.2485545157099693, 69.99619712558975),
                "critic_target": (1.6071403747214106, 69.9999635614258),
            },
            "second": {
                "actor": (17.9260930988969, 41.99993965101544),
                "actor_extractor": (-6.910333258522769, 252.26868389155572),
                "critic": (1.3893248375005385, 69.99139837758223),
                "critic_target": (1.6021433543544674, 69.99979273676631),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (18.44943795339418, 42.00044171197179),
                "actor_extractor": (-6.701845281681017, 252.26812527172683),
                "critic": (1.2485545157099693, 69.99619712558975),
                "critic_target": (1.6071403747214106, 69.9999635614258),
            },
            "second": {
                "actor": (18.662177466382445, 42.00046063668381),
                "actor_extractor": (-6.9106322794323205, 252.2686854232108),
                "critic": (1.3902389193184206, 69.99137606959279),
                "critic_target": (1.6021517250262889, 69.99979269757789),
            },
        },
    },
    "off2on_spot": {
        "separate": {
            "first": {
                "actor": (1.5237359622115036, 6.229650192306934),
                "actor_extractor": (2.6830465668660963, 94.51748560329698),
                "actor_target": (1.523139134325902, 6.22915676822542),
                "critic": (-0.6019846231210977, 11.087450753105138),
                "critic_extractor": (20.809810119544977, 94.68641546127812),
                "critic_target": (-0.5813934036414139, 11.091740240368162),
                "vae": (-10.567673536224902, 26.514026189503515),
            },
            "second": {
                "actor": (1.5239817481924547, 6.230090549237918),
                "actor_extractor": (3.2032015892966683, 94.51949839992179),
                "actor_target": (1.523143268583226, 6.229161481048667),
                "critic": (-0.6161619538324885, 11.085400200715343),
                "critic_extractor": (22.572687163914793, 94.70240080847486),
                "critic_target": (-0.58156713610515, 11.091707166380154),
                "vae": (-10.567673536224902, 26.514026189503515),
            },
        },
        "shared": {
            "first": {
                "actor": (1.2445580273633823, 6.234005510922179),
                "actor_extractor": (-1.488917196551256, 94.5001105399142),
                "actor_target": (1.2410750779672526, 6.233276605003897),
                "critic": (4.3497684703907, 12.353166254562373),
                "critic_target": (4.319292075873818, 12.353865464470633),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
            "second": {
                "actor": (1.2458130983868614, 6.23427301521928),
                "actor_extractor": (-4.863078758073016, 94.49894553736901),
                "actor_target": (1.2410987259354442, 6.233281583150686),
                "critic": (4.351555016197381, 12.352250505390845),
                "critic_target": (4.319453400210477, 12.353856319910774),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (1.2445580273633823, 6.234005510922179),
                "actor_extractor": (-1.488917196551256, 94.5001105399142),
                "actor_target": (1.2410750779672526, 6.233276605003897),
                "critic": (4.3497684703907, 12.353166254562373),
                "critic_target": (4.319292075873818, 12.353865464470633),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
            "second": {
                "actor": (1.2458130983868614, 6.23427301521928),
                "actor_extractor": (-4.863078758073016, 94.49894553736901),
                "actor_target": (1.2410987259354442, 6.233281583150686),
                "critic": (4.351555016197381, 12.352250505390845),
                "critic_target": (4.319453400210477, 12.353856319910774),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
        },
    },
    "rebrac": {
        "separate": {
            "first": {
                "actor": (1.4991364726884058, 6.223003992904282),
                "actor_extractor": (7.431522984298454, 94.5752835342584),
                "actor_target": (1.5230160861829063, 6.229122490424442),
                "critic": (31.282573492848314, 43.08302363890298),
                "critic_extractor": (-1.2815454900688792, 94.74984230979385),
                "critic_target": (31.418029648922357, 43.091708029846885),
            },
            "second": {
                "actor": (1.452972874569241, 6.2144976937667735),
                "actor_extractor": (9.218914810207707, 94.66521062145003),
                "actor_target": (1.522665922602755, 6.229046029256608),
                "critic": (31.166107561453828, 43.09109398395317),
                "critic_extractor": (-13.898369258476691, 94.89897574203945),
                "critic_target": (31.416770306442004, 43.0916765864055),
            },
        },
        "shared": {
            "first": {
                "actor": (1.2370593313826248, 6.236341953876156),
                "actor_extractor": (-6.156679535388093, 94.60900990368889),
                "actor_target": (1.2410377034102567, 6.233287441285757),
                "critic": (36.17682777091977, 44.31806334082144),
                "critic_target": (36.31842789803295, 44.35368337791005),
            },
            "second": {
                "actor": (1.2639334315317683, 6.242121423877094),
                "actor_extractor": (-6.3785383017036565, 94.80475710342952),
                "actor_target": (1.2411521321628243, 6.2333295789081005),
                "critic": (36.07654598975205, 44.30804927543156),
                "critic_target": (36.317218709393444, 44.353434570830814),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (1.2370593313826248, 6.236341953876156),
                "actor_extractor": (-6.156679535388093, 94.60900990368889),
                "actor_target": (1.2410377034102567, 6.233287441285757),
                "critic": (36.17682777091977, 44.31806334082144),
                "critic_target": (36.31842789803295, 44.35368337791005),
            },
            "second": {
                "actor": (1.2639334315317683, 6.242121423877094),
                "actor_extractor": (-6.3785383017036565, 94.80475710342952),
                "actor_target": (1.2411521321628243, 6.2333295789081005),
                "critic": (36.07654598975205, 44.30804927543156),
                "critic_target": (36.317218709393444, 44.353434570830814),
            },
        },
    },
    "spot": {
        "separate": {
            "first": {
                "actor": (1.5237359622115036, 6.229650192306934),
                "actor_extractor": (2.6830465668660963, 94.51748560329698),
                "actor_target": (1.523139134325902, 6.22915676822542),
                "critic": (-0.6019846231210977, 11.087450753105138),
                "critic_extractor": (20.809810119544977, 94.68641546127812),
                "critic_target": (-0.5813934036414139, 11.091740240368162),
                "vae": (-10.567673536224902, 26.514026189503515),
            },
            "second": {
                "actor": (1.5239817481924547, 6.230090549237918),
                "actor_extractor": (3.2032015892966683, 94.51949839992179),
                "actor_target": (1.523143268583226, 6.229161481048667),
                "critic": (-0.6161619538324885, 11.085400200715343),
                "critic_extractor": (22.572687163914793, 94.70240080847486),
                "critic_target": (-0.58156713610515, 11.091707166380154),
                "vae": (-10.567673536224902, 26.514026189503515),
            },
        },
        "shared": {
            "first": {
                "actor": (1.2445580273633823, 6.234005510922179),
                "actor_extractor": (-1.488917196551256, 94.5001105399142),
                "actor_target": (1.2410750779672526, 6.233276605003897),
                "critic": (4.3497684703907, 12.353166254562373),
                "critic_target": (4.319292075873818, 12.353865464470633),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
            "second": {
                "actor": (1.2458130983868614, 6.23427301521928),
                "actor_extractor": (-4.863078758073016, 94.49894553736901),
                "actor_target": (1.2410987259354442, 6.233281583150686),
                "critic": (4.351555016197381, 12.352250505390845),
                "critic_target": (4.319453400210477, 12.353856319910774),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (1.2445580273633823, 6.234005510922179),
                "actor_extractor": (-1.488917196551256, 94.5001105399142),
                "actor_target": (1.2410750779672526, 6.233276605003897),
                "critic": (4.3497684703907, 12.353166254562373),
                "critic_target": (4.319292075873818, 12.353865464470633),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
            "second": {
                "actor": (1.2458130983868614, 6.23427301521928),
                "actor_extractor": (-4.863078758073016, 94.49894553736901),
                "actor_target": (1.2410987259354442, 6.233281583150686),
                "critic": (4.351555016197381, 12.352250505390845),
                "critic_target": (4.319453400210477, 12.353856319910774),
                "vae": (-9.285035636945395, 26.29174830460068),
            },
        },
    },
    "td3": {
        "separate": {
            "first": {
                "actor": (2.8516391344110588, 41.9987545835816),
                "actor_extractor": (-6.72777366531227, 252.26613658894843),
                "critic": (-11.49384431571923, 70.00061091078676),
                "critic_extractor": (-19.481771682574617, 253.32513999951283),
                "critic_target": (-9.958957839268152, 70.00000318740041),
            },
            "second": {
                "actor": (3.3091070958490008, 41.99803128984354),
                "actor_extractor": (-6.636782152488195, 252.265928296616),
                "critic": (-12.205852934652626, 70.00136714816526),
                "critic_extractor": (-19.430075321201002, 253.32633329062918),
                "critic_target": (-9.981426907974928, 70.00000157217156),
            },
        },
        "shared": {
            "first": {
                "actor": (18.2536595717786, 41.99840485785819),
                "actor_extractor": (-6.791537198023434, 252.2675911781909),
                "critic": (1.126125398251446, 69.99231585763917),
                "critic_target": (1.6059163935932759, 69.99992116316011),
            },
            "second": {
                "actor": (18.342459642107542, 41.9976091423354),
                "actor_extractor": (-6.84516923176534, 252.26741422604118),
                "critic": (0.9242826924485996, 69.99013410982533),
                "critic_target": (1.5990999462516697, 69.99980900320314),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (17.915260546341642, 41.99887806671639),
                "actor_extractor": (-6.791537198023434, 252.2675911781909),
                "critic": (1.126125398251446, 69.99231585763917),
                "critic_target": (1.6059163935932759, 69.99992116316011),
            },
            "second": {
                "actor": (17.808330720686854, 41.99789597610606),
                "actor_extractor": (-6.844973196822139, 252.26740172630952),
                "critic": (0.9228103973522082, 69.99015442531572),
                "critic_target": (1.5990850478765637, 69.99980921868158),
            },
        },
    },
    "td3_bc": {
        "separate": {
            "first": {
                "actor": (1.5111358772119274, 6.229607809863329),
                "actor_extractor": (3.2335254828074085, 94.52048697446466),
                "actor_target": (1.523076101540937, 6.229156484244729),
                "critic": (-0.6226685173169244, 11.089204535371653),
                "critic_extractor": (14.606485906591274, 94.68306537877544),
                "critic_target": (-0.5814965985482559, 11.091748972076893),
            },
            "second": {
                "actor": (1.5005658899171976, 6.228882802541763),
                "actor_extractor": (5.23700647362989, 94.53509173124635),
                "actor_target": (1.5229636110452702, 6.229154891746834),
                "critic": (-0.645550474699121, 11.088004516565295),
                "critic_extractor": (11.00630005972414, 94.68785849751441),
                "critic_target": (-0.5818167784600519, 11.091728376519997),
            },
        },
        "shared": {
            "first": {
                "actor": (1.251255735871382, 6.233566633263287),
                "actor_extractor": (2.3803998293153654, 94.52285090814595),
                "actor_target": (1.2411086685606278, 6.233274312319748),
                "critic": (4.296105093235383, 12.350077634758183),
                "critic_target": (4.319023881049361, 12.353850061253134),
            },
            "second": {
                "actor": (1.2593526121927425, 6.234570260075453),
                "actor_extractor": (-0.4396219038333129, 94.51912508667115),
                "actor_target": (1.2411999667529017, 6.23328051479143),
                "critic": (4.289737286948366, 12.348029748390575),
                "critic_target": (4.318877428886481, 12.35381998631734),
            },
        },
        "shared_critic_grad": {
            "first": {
                "actor": (1.2530006725573912, 6.233727874765412),
                "actor_extractor": (2.3803998293153654, 94.52285090814595),
                "actor_target": (1.2411173987784423, 6.23327512698233),
                "critic": (4.296105093235383, 12.350077634758183),
                "critic_target": (4.319023881049361, 12.353850061253134),
            },
            "second": {
                "actor": (1.2621604048181325, 6.234852381773751),
                "actor_extractor": (-0.4396219038333129, 94.51912508667115),
                "actor_target": (1.2412226684391499, 6.233282727145886),
                "critic": (4.289737286948366, 12.348029748390575),
                "critic_target": (4.318877428886481, 12.35381998631734),
            },
        },
    },
}


@requires_snapshot_build
@pytest.mark.parametrize("encoder_sharing", ["shared_critic_grad", "separate"])
@pytest.mark.parametrize("case", sorted(FAMILY_CASES))
def test_family_matches_recorded_run(case, encoder_sharing):
    assert family_digests(case, encoder_sharing) == SNAPSHOT[case][encoder_sharing]
