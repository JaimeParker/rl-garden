"""Flat Args, keyword constructors and checkpoint metadata share parameter fields."""
from __future__ import annotations

from functools import lru_cache
import dataclasses
import importlib
import inspect
import subprocess
import sys
from typing import Literal, get_args, get_origin, get_type_hints

import pytest
from test_args_builder_drift import capture_builder, registered_entries
from test_awac import _state_env

from rl_garden.algorithms.dreamer_v3 import DreamerV3
from rl_garden.algorithms.ncrl import NCRL
from rl_garden.algorithms.rssm_pretrain import RSSMPretrain
from rl_garden.algorithms.awac import AWAC, _AWACRolloutTrainingShell
from rl_garden.algorithms.base_algorithm import BaseAlgorithm
from rl_garden.algorithms.off2on_awac import Off2OnAWAC
from rl_garden.algorithms.iql import IQL, _IQLRolloutTrainingShell
from rl_garden.algorithms.off2on_iql import Off2OnIQL
from rl_garden.algorithms.td3_bc import TD3BC
from rl_garden.algorithms.rebrac import ReBRAC
from rl_garden.algorithms.spot import SPOT, _SPOTRolloutTrainingShell
from rl_garden.algorithms.off2on_spot import Off2OnSPOT
from rl_garden.algorithms.fql import FQL
from rl_garden.algorithms.off2on_fql import Off2OnFQL
from rl_garden.algorithms.floq import FloQ, _FloQRolloutTrainingShell, Off2OnFloQ
from rl_garden.algorithms.fino import FINO, _FINORolloutTrainingShell, Off2OnFINO
from rl_garden.algorithms.fac import FAC, _FACRolloutTrainingShell, Off2OnFAC
from rl_garden.algorithms.value_flows import (
    ValueFlows, _ValueFlowsRolloutTrainingShell, Off2OnValueFlows,
)
from rl_garden.algorithms.acfql import ACFQL, _ACFQLRolloutTrainingShell
from rl_garden.algorithms.chunked_rollout import ChunkedRolloutMixin
from rl_garden.hyperparameters.spot import SPOTParams
from rl_garden.algorithms.qam import QAM
from rl_garden.algorithms.qgf import QGF
from rl_garden.algorithms.bcq import BCQ
from rl_garden.algorithms.plas import PLAS
from rl_garden.algorithms.idql import IDQL
from rl_garden.algorithms.bc import BC
from rl_garden.algorithms.dagger import DAgger
from rl_garden.algorithms.flow_bc import FlowBC
from rl_garden.algorithms.mean_flow_bc import MeanFlowBC
from rl_garden.algorithms.diffusion_bc import DiffusionBC
from rl_garden.algorithms.a2a_bc import A2ABC
from rl_garden.algorithms.consistency_distill_bc import ConsistencyDistillBC
from rl_garden.algorithms.bppo import BPPO
from rl_garden.algorithms.unio4 import UniO4
from rl_garden.algorithms.unio4_ope import UniO4OPE
from test_bppo import _state_env as _bppo_env
from rl_garden.algorithms.hilp import HILP
from rl_garden.algorithms.opal import OPAL
from rl_garden.algorithms.ddpg import DDPG
from rl_garden.algorithms.td3 import TD3
from test_td3 import DummyDictVecEnv
from test_dagger import _FakeEnv, _ConstantExpert
from test_a2a_bc import _env_spec as _a2a_env
from rl_garden.algorithms import (
    PPO, RecurrentPPO, TransformerPPO, GAIL, DPPO, FlowPPO,
    DiffusionCMDistillOnline, PolicyDistillation,
)
from test_ppo_adaptive_kl import _FakeBoxEnv as _ppo_env
from test_dppo_smoke import _FakeEnv as _chain_env
from test_policy_distillation import _FakeDualObsEnv as _distill_env

from rl_garden.algorithms.tdmpc2 import TDMPC2
from rl_garden.algorithms.tdmpc2_multitask import TDMPC2Multitask
from rl_garden.algorithms.pwm import PWM
from rl_garden.algorithms.supermimic import SuperMimic
from test_tdmpc2_smoke import DummyVecEnv as _tdmpc_env
from test_tdmpc2_multitask_agent import _make_env_spec as _multitask_env
from test_pwm_smoke import env_spec as _pwm_env
from _supermimic_fakes import PointMassTrackingEnv
from rl_garden.algorithms import SAC, SACFlow, RecurrentSAC, TransformerSAC, RLPD, RLPDHybrid, ExPLORe, SUPE, ACRLPD
from rl_garden.algorithms.sac_ddp import SACDDP
from test_rlpd_hybrid import DummyVecEnv as _sac_env
from rl_garden.algorithms.flash_sac import FlashSAC
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.algorithms.edac import EDAC
from rl_garden.algorithms.cql import CQL, _CQLRolloutTrainingShell
from rl_garden.algorithms.calql import CalQL, _CalQLRolloutTrainingShell
from rl_garden.algorithms.wsrl import WSRL
from rl_garden.algorithms.off2on_calql import Off2OnCalQL
from test_flash_sac import _FakeEnv as _flash_env
from test_offline_sac import _state_env as _offline_sac_env





class _ACFQLShellHost(ChunkedRolloutMixin, _ACFQLRolloutTrainingShell):
    """The internal shell requires the host's action-chunking methods."""


@dataclasses.dataclass(frozen=True)
class Family:
    classes: tuple
    env: object
    entries: tuple
    small: dict = dataclasses.field(default_factory=lambda: {"net_arch": (32, 32, 32)})
    values: dict = dataclasses.field(default_factory=dict)
    builder_net_arch_type: type = tuple


SPOT_SMALL = {"net_arch": (32, 32), "vae_hidden_dim": 16, "vae_iterations": 0}
SPOT_VALUES = {"vae_latent_dim": 4, "vae_hidden_dim": 24, "vae_iterations": 2}


IQL_VALUES = {
    "expectile": 0.8,
    "kernel_init": "orthogonal_near_zero_output",
    "critic_subsample_size": 1,
    "actor_lr_warmup_steps": 2,
    "actor_lr_decay_steps": 20,
    "actor_lr_min_ratio": 0.1,
}


FQL_VALUES = {"net_arch": (32, 32), "kernel_init": "orthogonal", "activation_fn": "relu"}


CHUNK_BC_CLASSES = (DiffusionBC, A2ABC, ConsistencyDistillBC)
BC_CLASSES = (BC, DAgger, FlowBC, MeanFlowBC, *CHUNK_BC_CLASSES)


def _skill_env():
    from gymnasium import spaces
    from rl_garden.algorithms.offline import OfflineEnvSpec

    return OfflineEnvSpec(spaces.Box(-1.0, 1.0, (6,)), spaces.Box(-1.0, 1.0, (2,)))


SKILL_CLASSES = (HILP, OPAL)
IMAGE_ONLINE_CLASSES = (DDPG, TD3)


BPPO_CLASSES = (BPPO, UniO4, UniO4OPE)
BPPO_SMALL = {"value_hidden_dims": (32, 32), "q_hidden_dims": (32, 32),
              "actor_hidden_dims": (32, 32)}
BPPO_VALUES = {"value_hidden_dims": (24, 24), "q_hidden_dims": (24, 24),
               "actor_hidden_dims": (24, 24), "omega": 0.6,
               "use_layer_norm": True, "use_group_norm": True, "dropout_rate": 0.2}


ON_POLICY_CLASSES = (PPO, RecurrentPPO, TransformerPPO, GAIL, DPPO, FlowPPO,
                     DiffusionCMDistillOnline, PolicyDistillation)
PPO_SMALL = {"net_arch": (32, 32, 32), "num_minibatches": 2,
             "lr_warmup_steps": 2, "lr_decay_steps": 20}
PPO_VALUES = {"num_minibatches": 1, "log_std_init_per_dim": (-0.2, -0.3),
              "action_mean_bias_init": (0.1, 0.2), "net_arch": (32, 32, 32)}
CHAIN_SMALL = {"actor_mlp_dims": (32, 32, 32), "critic_mlp_dims": (32, 32, 32),
               "lr_warmup_steps": 2, "lr_decay_steps": 20}
CHAIN_VALUES = {"actor_mlp_dims": (16, 16, 16), "critic_mlp_dims": (16, 16, 16),
                "clip_advantage_lower_quantile": 0.1,
                "clip_advantage_upper_quantile": 0.9, "reward_horizon": 2,
                "time_dim": 24, "gamma_denoising": 0.9, "sigma_safe_max": 0.8}


CQL_CLASSES = (CQL, CalQL, _CQLRolloutTrainingShell, _CalQLRolloutTrainingShell,
               WSRL, Off2OnCalQL)
CQL_VALUES = {"net_arch": (32, 32, 32), "sarsa_hidden_dims": (32, 32),
              "kernel_init": "orthogonal", "ent_coef": 0.2,
              "target_entropy": -2.0, "critic_subsample_size": 1,
              "cql_action_sample_method": "normal", "cql_clip_diff_min": -5.0,
              "cql_clip_diff_max": 5.0}
CQL_SMALL = {"net_arch": (32, 32, 32), "n_critics": 3, "cql_n_actions": 2}
CQL_ENTRIES = (("offline", "cql"), ("offline", "calql"), ("offline", "wsrl"),
               ("off2on", "calql"), ("off2on", "wsrl"))


FAMILIES = [
    Family((CQL, _CQLRolloutTrainingShell), _offline_sac_env, (("offline", "cql"),),
           small=CQL_SMALL, values=CQL_VALUES),
    Family((CalQL,), _offline_sac_env, (("offline", "calql"),),
           small=CQL_SMALL, values=CQL_VALUES),
    Family((WSRL, _CalQLRolloutTrainingShell), _offline_sac_env,
           (("offline", "wsrl"), ("off2on", "wsrl")), small=CQL_SMALL, values=CQL_VALUES),
    Family((Off2OnCalQL,), _offline_sac_env, (("off2on", "calql"),),
           small=CQL_SMALL, values=CQL_VALUES),
    Family((FlashSAC,), _flash_env, (("online", "flash_sac"),),
           small={"actor_hidden_dim": 16, "critic_hidden_dim": 16,
                  "actor_num_blocks": 1, "critic_num_blocks": 1, "num_bins": 11},
           values={"target_entropy": -2.0}),
    Family((OfflineSAC,), _offline_sac_env, (),
           values={"ent_coef": 0.2, "target_entropy": -2.0,
                   "critic_subsample_size": 1}, builder_net_arch_type=list),
    Family((EDAC,), _offline_sac_env, (("offline", "edac"),),
           small={"net_arch": (16, 16), "n_critics": 3},
           # The edac entry's default widths are a tuple, not None: name a value here.
           values={"ent_coef": 0.2, "target_entropy": -2.0, "net_arch": (32, 32),
                   "critic_subsample_size": 1}, builder_net_arch_type=list),
    Family((PPO,), _ppo_env, (("online", "ppo"),), small=PPO_SMALL, values=PPO_VALUES),
    Family((RecurrentPPO,), _ppo_env, (("online", "recurrent_ppo"),),
           small={**PPO_SMALL, "rnn_hidden_size": 32}, values=PPO_VALUES),
    Family((TransformerPPO,), _ppo_env, (("online", "transformer_ppo"),),
           small={**PPO_SMALL, "embed_dim": 32, "head_dim": 8,
                  "num_transformer_layers": 1, "memory_len": 4},
           values={**PPO_VALUES, "embed_dim": 16}),
    Family((GAIL,), _ppo_env, (("online", "gail"),), small=PPO_SMALL,
           values={**PPO_VALUES, "disc_net_arch": (16, 16)}),
    Family((DPPO,), _chain_env, (("online", "dppo"),), small=CHAIN_SMALL,
           values={**CHAIN_VALUES, "act_steps": 2, "ft_denoising_steps": 2}),
    Family((FlowPPO,), _chain_env, (("online", "flow_ppo"),), small=CHAIN_SMALL,
           values=CHAIN_VALUES),
    Family((DiffusionCMDistillOnline,), _chain_env, (("online", "diffusion_cm_distill_online"),),
           small=CHAIN_SMALL,
           values={**CHAIN_VALUES, "act_steps": 2, "ft_denoising_steps": 2,
                   "cm_mlp_dims": (16, 16, 16), "cm_ema_decay": 0.9}),
    Family((PolicyDistillation,), _distill_env, (("online", "policy_distillation"),),
           small={"net_arch": (32, 32, 32)}, values={"net_arch": (32, 32, 32)},
           builder_net_arch_type=list),
    Family((BPPO,), _bppo_env, (("offline", "bppo"),),
           small=BPPO_SMALL, values=BPPO_VALUES),
    Family((UniO4,), _bppo_env, (("offline", "unio4"),),
           small={**BPPO_SMALL, "num_policies": 2}, values=BPPO_VALUES),
    Family((UniO4OPE,), _bppo_env, (("offline", "unio4_ope"),),
           small={**BPPO_SMALL, "num_policies": 2, "dynamics_hidden_dims": (8, 8),
                  "dynamics_weight_decay": (1e-5, 1e-5, 1e-5),
                  "dynamics_n_ensemble": 2, "dynamics_n_elites": 1},
           values={**BPPO_VALUES, "dynamics_hidden_dims": (16, 16),
                   "dynamics_weight_decay": (2e-5, 2e-5, 2e-5),
                   "dynamics_n_ensemble": 3, "dynamics_n_elites": 2,
                   "dynamics_max_epochs": 2}),
    Family((HILP,), _skill_env, (("offline", "hilp"),),
           small={"skill_dim": 4, "value_hidden_dims": (16, 16),
                  "actor_hidden_dims": (16, 16)},
           values={"value_hidden_dims": (24, 24), "actor_hidden_dims": (24, 24),
                   "expectile": 0.8, "skill_expectile": 0.7,
                   "discount": 0.9, "skill_discount": 0.9,
                   "p_currgoal": 0.1, "p_trajgoal": 0.5, "p_randomgoal": 0.4}),
    Family((OPAL,), _skill_env, (("offline", "opal"),),
           small={"skill_dim": 4, "chunk_size": 2, "hidden_size": 8,
                  "vae_hidden_dims": (8, 8)},
           values={"vae_hidden_dims": (16, 16)}),
    Family((DDPG,), DummyDictVecEnv, (("online", "drqv2"),),
           small={"hidden_dim": 16, "feature_dim": 8},
           values={"stddev_schedule": "linear(0.5,0.1,100)", "lr_min_ratio": 0.1}),
    Family((TD3,), DummyDictVecEnv, (("online", "td3"),),
           small={"hidden_dim": 16, "feature_dim": 8},
           values={"stddev_schedule": "linear(0.5,0.1,100)", "lr_min_ratio": 0.1}),
    Family((BC,), _state_env, (("offline", "bc"),)),
    Family((DAgger,), _FakeEnv, (("online", "dagger"),),
           values={"net_arch": (32, 32)}, builder_net_arch_type=list),
    Family((FlowBC,), _state_env, (("offline", "flow_bc"),),
           values={"net_arch": (32, 32)}),
    Family((MeanFlowBC,), _state_env, (("offline", "mean_flow_bc"),),
           values={"net_arch": (32, 32)}),
    Family((DiffusionBC,), _state_env, (("offline", "diffusion_bc"),),
           small={"mlp_dims": (16, 16, 16), "horizon_steps": 2, "denoising_steps": 4},
           values={"ema_decay": 0.9, "ema_start_step": 2, "time_dim": 24,
                   "mlp_dims": (24, 24, 24), "min_sampling_denoising_std": 0.2}),
    Family((ConsistencyDistillBC,), _state_env, (("offline", "consistency_distill_bc"),),
           small={"mlp_dims": (16, 16, 16), "horizon_steps": 2, "denoising_steps": 4},
           values={"cm_ema_decay": 0.9, "time_dim": 24, "mlp_dims": (24, 24, 24)}),
    Family((A2ABC,), lambda: _a2a_env(4, 2), (("offline", "a2a_bc"),),
           small={"horizon_steps": 2, "cond_steps": 2, "latent_dim": 16,
                  "cnn_num_layers": 1, "cnn_hidden_channels": 8,
                  "decoder_net_arch": (16, 16), "flow_hidden_dims": (16, 16)},
           values={"cnn_kernel_size": 3, "cnn_activation_fn": "gelu",
                   "decoder_net_arch": (24, 24), "flow_hidden_dims": (24, 24)}),
    Family((QAM,), _state_env, (("offline", "qam"),),
           small={"net_arch": (32, 32), "flow_steps": 2},
           values={"expectile": 0.8, "kernel_init": "orthogonal", "activation_fn": "relu"}),
    Family((QGF,), _state_env, (("offline", "qgf"),),
           small={"net_arch": (32, 32), "denoise_steps": 2},
           values={"expectile": 0.8, "robust_critic_t_emb_size": 24,
                   "kernel_init": "orthogonal", "activation_fn": "relu"}),
    Family((BCQ,), _state_env, (("offline", "bcq"),),
           small={"net_arch": (32, 32), "vae_hidden_dim": 16},
           values={"vae_latent_dim": 4, "vae_hidden_dim": 24, "soft_q_lambda": 0.6,
                   "net_arch": (32, 32)},
           builder_net_arch_type=list),
    Family((PLAS,), _state_env, (("offline", "plas"),),
           small={"net_arch": (32, 32), "vae_hidden_dim": 16, "vae_iterations": 0},
           values={"vae_latent_dim": 4, "vae_hidden_dim": 24, "soft_q_lambda": 0.6,
                   "net_arch": (32, 32)},
           builder_net_arch_type=list),
    Family((IDQL,), _state_env, (("offline", "idql"),),
           small={"critic_hidden_dims": (16, 16), "value_hidden_dims": (16, 16),
                  "diffusion_mlp_dims": (16, 16), "denoising_steps": 2, "n_action_samples": 4},
           values={"expectile": 0.8, "critic_subsample_size": 1,
                   "critic_hidden_dims": (24, 24), "value_hidden_dims": (24, 24),
                   "diffusion_mlp_dims": (24, 24)}),
    Family((FQL, Off2OnFQL), _state_env,
           (("offline", "fql"), ("off2on", "fql")), values=FQL_VALUES, builder_net_arch_type=list),
    Family((FloQ, _FloQRolloutTrainingShell, Off2OnFloQ), _state_env,
           (("offline", "floq"), ("off2on", "floq")),
           values={**FQL_VALUES, "time_embed_dim": 32}, builder_net_arch_type=list),
    Family((FINO, _FINORolloutTrainingShell, Off2OnFINO), _state_env,
           (("offline", "fino"), ("off2on", "fino")), values={**FQL_VALUES, "num_samples": 3}, builder_net_arch_type=list),
    Family((FAC, _FACRolloutTrainingShell, Off2OnFAC), _state_env,
           (("offline", "fac"), ("off2on", "fac")),
           values={**FQL_VALUES, "bc_batch_size": 4, "bc_pretrain_steps": 0}, builder_net_arch_type=list),
    Family((ValueFlows, _ValueFlowsRolloutTrainingShell, Off2OnValueFlows), _state_env,
           (("offline", "value_flows"), ("off2on", "value_flows")), values=FQL_VALUES, builder_net_arch_type=list),
    Family((_ACFQLShellHost, ACFQL), _state_env, (("off2on", "acfql"),), values=FQL_VALUES, builder_net_arch_type=list),

    Family((IQL, _IQLRolloutTrainingShell, Off2OnIQL), _state_env,
           (("offline", "iql"), ("off2on", "iql")), values=IQL_VALUES),
    Family((AWAC, _AWACRolloutTrainingShell, Off2OnAWAC), _state_env,
           (("offline", "awac"), ("off2on", "awac"))),
    Family((TD3BC,), _state_env, (("offline", "td3_bc"),)),
    Family((ReBRAC,), _state_env, (("offline", "rebrac"),),
           values={"net_arch": [32, 32]}, builder_net_arch_type=list),
    Family((SPOT, _SPOTRolloutTrainingShell, Off2OnSPOT), _state_env,
           (("offline", "spot"), ("off2on", "spot")), SPOT_SMALL, SPOT_VALUES),
]

MODEL_BASED_CLASSES = (TDMPC2, TDMPC2Multitask, PWM, SuperMimic)
MODEL_BASED_SMALL = {"horizon": 2, "latent_dim": 16, "mlp_dim": 16,
                     "num_q": 2, "num_bins": 11}
MODEL_BASED_VALUES = {"latent_dim": 24, "simnorm_dim": 4, "num_samples": 12,
                      "num_elites": 3, "num_pi_trajs": 1, "horizon": 4,
                      "discount_min": 0.9, "discount_max": 0.999,
                      "discount_denom": 6.0, "tau": 0.02,
                      "log_std_min": -9.0, "log_std_max": 1.0}
FAMILIES += [
    Family((TDMPC2,), _tdmpc_env, (("online", "tdmpc2"),),
           small={**MODEL_BASED_SMALL, "num_samples": 8, "num_elites": 4,
                  "num_pi_trajs": 2, "iterations": 1}, values=MODEL_BASED_VALUES),
    Family((TDMPC2Multitask,), _multitask_env, (("offline", "tdmpc2_multitask"),),
           small={**MODEL_BASED_SMALL, "task_dim": 8, "enc_dim": 8}, values=MODEL_BASED_VALUES),
    Family((PWM,), _pwm_env, (("offline", "pwm"),),
           small={"horizon": 2, "actor_units": (16,), "critic_units": (16,),
                  "critic_iterations": 1, "critic_batches": 1},
           values={"actor_units": (24,), "critic_units": (24,), "horizon": 4}),
    Family((SuperMimic,), PointMassTrackingEnv, (("online", "supermimic"),),
           small={"wm_hidden_dims": (16,), "head_hidden_dims": (16,),
                  "goal_encoder_dims": (16,), "decoder_dims": (16,),
                  "critic_dims": (16,), "horizon": 2, "wm_window": 2,
                  "wm_batches": 1, "wm_batch_size": 4, "imagination_batch": 4,
                  "critic_epochs": 1, "critic_minibatches": 1},
           values={"wm_hidden_dims": (24,), "head_hidden_dims": (24,),
                   "goal_encoder_dims": (24,), "decoder_dims": (24,),
                   "critic_dims": (24,), "actor_betas": (0.8, 0.9),
                   "lr_final_ratio": 0.5, "horizon": 4, "wm_window": 3, "replay_ratio": 8.0}),
]


SAC_CLASSES = (SAC, SACDDP, SACFlow, RecurrentSAC, TransformerSAC, RLPD, RLPDHybrid, ExPLORe, SUPE, ACRLPD)
SAC_SMALL = {"net_arch": (16, 16, 16), "n_critics": 3,
             "lr_warmup_steps": 2, "lr_decay_steps": 20}
SAC_VALUES = {"net_arch": (24, 24, 24), "ent_coef": 0.1, "target_entropy": -3.0,
              "critic_subsample_size": 1, "lr_min_ratio": 0.1, "tau": 0.02,
              "flow_hidden_dims": (24, 24), "flow_use_layer_norm": True,
              "relabeler_hidden_dims": (24, 24), "rnd_hidden_dims": (24, 24),
              "actor_log_std_min": -4.0, "embed_dim": 24,
              "dropout_rate": 0.1, "prio_exponent": 0.8,
              "importance_sampling_exponent": 0.5, "burn_in_len": 8,
              "learning_len": 3, "forward_len": 3}
SAC_SEQUENCE_SMALL = {**SAC_SMALL, "burn_in_len": 4, "learning_len": 2, "forward_len": 2}
FAMILIES += [
    Family((SAC, SACDDP), _sac_env, (("online", "sac"),), small=SAC_SMALL, values=SAC_VALUES),
    Family((SACFlow,), _sac_env, (("online", "sac_flow"),),
           small={**SAC_SMALL, "flow_hidden_dims": (16, 16)}, values=SAC_VALUES),
    Family((RecurrentSAC,), _sac_env, (("online", "recurrent_sac"),),
           small={**SAC_SEQUENCE_SMALL, "rnn_hidden_size": 16}, values=SAC_VALUES),
    Family((TransformerSAC,), _sac_env, (("online", "transformer_sac"),),
           small={**SAC_SEQUENCE_SMALL, "embed_dim": 16, "head_dim": 4, "num_heads": 4,
                  "memory_len": 2, "num_transformer_layers": 1},
           values={**SAC_VALUES, "memory_len": 3}),
    Family((RLPD,), _sac_env, (("online", "rlpd"),), small=SAC_SMALL, values=SAC_VALUES),
    Family((RLPDHybrid,), _sac_env, (("online", "rlpd_hybrid"),),
           small={**SAC_SMALL, "discrete_hidden_dim": 16}, values=SAC_VALUES),
    Family((ExPLORe,), _sac_env, (("online", "explore"),),
           small={**SAC_SMALL, "offline_relabel_type": "gt", "relabeler_hidden_dims": (16, 16),
                  "rnd_hidden_dims": (16, 16), "rnd_feature_dim": 16}, values=SAC_VALUES),
    Family((SUPE,), _sac_env, (("online", "supe"),),
           small={**SAC_SMALL, "offline_relabel_type": "gt", "relabeler_hidden_dims": (16, 16),
                  "rnd_hidden_dims": (16, 16), "rnd_feature_dim": 16}, values=SAC_VALUES),
    Family((ACRLPD,), _sac_env, (("online", "acrlpd"),),
           small={**SAC_SMALL, "horizon_length": 2}, values=SAC_VALUES),
]


DREAMER_CLASSES = (DreamerV3, NCRL, RSSMPretrain)
DREAMER_VALUES = {
    "size": "25M", "blocks": 4, "batch_length": 4, "lam": 0.9,
    # The dreamer_v3 entry's default is None (use --horizon): name a value here.
    "discount_horizon": 100.0,
    "discount": 0.98, "online_ratio": 0.5,
    "guidance_schedule": ("linear", 0.8, 0.2, 20),
    "deter": 64, "hidden": 32, "classes": 8, "cnn_depth": 8,
}
FAMILIES += [
    Family((DreamerV3,), None, (("online", "dreamer_v3"),),
           small={"stoch": 4, "batch_length": 6, "imag_horizon": 3}, values=DREAMER_VALUES),
    Family((NCRL,), None, (("online", "ncrl"),),
           small={"batch_length": 6, "imag_horizon": 3, "ac_units": 32,
                  "reward_units": 32, "bc_train_steps": 2}, values=DREAMER_VALUES),
    Family((RSSMPretrain,), None, (("offline", "rssm_pretrain"),),
           small={"stoch": 4, "batch_length": 6},
           values={**DREAMER_VALUES, "size": "size12M"}),
]



CLASSES = [(cls, family.env) for family in FAMILIES for cls in family.classes]
CLASS_FAMILY = {cls: family for family in FAMILIES for cls in family.classes}
ENTRIES = registered_entries()
ENTRY_KEYS = [key for family in FAMILIES for key in family.entries]
FIELDS = [(cls, env, field.name) for cls, env in CLASSES
          for field in dataclasses.fields(cls.params_class)]


# Dataset and expert inputs retain their original leading positional slots.
POSITIONAL_INPUTS = {
    CQL: ("env",),
    CalQL: ("env",),
    FlashSAC: ("env", "eval_env"),
    OfflineSAC: ("env",),
    EDAC: ("env",),
    SUPE: ("env", "opal_checkpoint", "eval_env"),
    DPPO: ("env", "bc_checkpoint", "eval_env"),
    PolicyDistillation: ("env", "teacher_policy", "teacher_obs_keys", "student_obs_keys", "eval_env"),
    A2ABC: ("env", "dataset_path"),
    DiffusionBC: ("env", "dataset_path"),
    ConsistencyDistillBC: ("env", "dataset_path", "bc_checkpoint"),
    DAgger: ("env", "expert", "eval_env"),
    HILP: ("env", "dataset_path"),
    OPAL: ("env", "dataset_path"),
}


@pytest.mark.parametrize("cls", [cls for cls, _ in CLASSES], ids=lambda cls: cls.__name__)
def test_constructor_positionals_are_only_leading_inputs(cls):
    allowed = POSITIONAL_INPUTS.get(cls, ("env", "eval_env"))
    # Forwarding wrappers such as ACFQL reach a concrete signature in the MRO.
    for owner in cls.__mro__:
        if "__init__" not in vars(owner):
            continue
        parameters = tuple(inspect.signature(owner.__init__).parameters.values())[1:]
        positional = tuple(p.name for p in parameters if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ))
        assert positional == allowed[:len(positional)], (cls, owner, positional)
        if not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters):
            break
    else:
        raise AssertionError(f"No bounded constructor signature for {cls.__name__}")


def test_base_metadata_without_params_is_unchanged():
    from types import SimpleNamespace

    assert BaseAlgorithm.params_class is None
    agent = SimpleNamespace(params_class=BaseAlgorithm.params_class, seed=17, device="cpu")
    assert BaseAlgorithm._checkpoint_metadata(agent) == {"seed": 17, "device": "cpu"}


def field_value(family, field, default, hint):
    return family.values[field] if field in family.values else nondefault(default, hint)


def nondefault(value, hint):
    if value is not None and type(None) in get_args(hint):
        return nondefault(value, next(t for t in get_args(hint) if t is not type(None)))
    if get_origin(hint) is Literal:
        return next(item for item in get_args(hint) if item != value)
    if value is None:
        types = [t for t in get_args(hint) if t is not type(None)]
        hint = types[0]
        if get_origin(hint) is Literal:
            return get_args(hint)[0]
        return 0.1 if hint is float else (32, 32)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return 16 if value == 32 else value + 1
    if isinstance(value, float):
        return value * 2 if value else 0.01
    raise AssertionError(f"No non-default value for {hint}: {value!r}")


@lru_cache(maxsize=1)
def _dreamer_contract_artifacts():
    import tempfile
    from pathlib import Path
    from _ncrl_online_synthetic import make_pretrain_artifacts

    directory = tempfile.TemporaryDirectory()
    artifacts = make_pretrain_artifacts(Path(directory.name))
    artifacts["temporary_directory"] = directory
    return artifacts


def construct(cls, env, **kwargs):
    if issubclass(cls, DREAMER_CLASSES):
        from unittest.mock import patch
        from _ncrl_online_synthetic import PixelVecEnv, TINY_SIZE
        from rl_garden.algorithms.rssm_pretrain import sequence_dataset_env_spec
        from rl_garden.world_models.rssm import RSSM_SIZES

        artifacts = _dreamer_contract_artifacts()
        common = dict(batch_size=4, buffer_device="cpu", device="cpu", std_log=False,
                      compute_dtype="float32", log_freq=0)
        if issubclass(cls, RSSMPretrain):
            with patch.dict(RSSM_SIZES, {name: TINY_SIZE for name in RSSM_SIZES}):
                return cls(sequence_dataset_env_spec(artifacts["dataset_dir"]),
                           dataset_dir=artifacts["dataset_dir"], **common, **kwargs)
        common.update(buffer_size=32, learning_starts=20, eval_freq=0)
        if issubclass(cls, NCRL):
            common.update(pretrained_world_model=str(artifacts["checkpoint"]),
                          retrieval_db=str(artifacts["retrieval_db"]),
                          episode_frames=10, action_repeat=1)
        else:
            common["rssm_size"] = TINY_SIZE
        return cls(PixelVecEnv(), **common, **kwargs)

    if cls in SAC_CLASSES:
        instance = env()
        if cls is SUPE:
            from rl_garden.networks.opal_vae import OPALVAE
            kwargs.update(opal_checkpoint="", opal_vae=OPALVAE(4, instance.single_action_space, 3, 2,
                          hidden_size=8, prior_hidden_dims=(8, 8), decoder_hidden_dims=(8, 8)))
        return cls(instance, buffer_size=32, batch_size=4, buffer_device="cpu",
                   device="cpu", std_log=False, **kwargs)

    if cls in MODEL_BASED_CLASSES:
        import tempfile
        from pathlib import Path

        common = dict(buffer_size=32, batch_size=4, buffer_device="cpu",
                      device="cpu", std_log=False)
        if cls is TDMPC2:
            return cls(env(), episode_length=8, seed_steps=16, **common, **kwargs)
        if cls is SuperMimic:
            common.pop("batch_size")
            return cls(env(4, extra_keys=True), learning_starts=16, **common, **kwargs)
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name)
        common.pop("buffer_device")
        if cls is TDMPC2Multitask:
            from test_tdmpc2_multitask_agent import _TASKS, _OBS_DIMS, _ACTION_DIMS, _EPISODE_LENGTHS

            agent = cls(env(), tasks=_TASKS, obs_dims=_OBS_DIMS, action_dims=_ACTION_DIMS,
                        episode_lengths=_EPISODE_LENGTHS, mmap_dir=path / "buffer",
                        **common, **kwargs)
        else:
            from test_pwm_smoke import write_converted_world_model

            checkpoint = write_converted_world_model(path / "teacher")
            agent = cls(env(), world_model_checkpoint=str(checkpoint), task_idx=0,
                        mmap_dir=path / "buffer", **common, **kwargs)
        agent._contract_directory = directory
        return agent
    if cls in ON_POLICY_CLASSES:
        from rl_garden.envs.wrappers import ActionChunkWrapper

        instance = env()
        if cls in (DPPO, DiffusionCMDistillOnline, FlowPPO):
            # Read only the chunk length the env wrapper needs; the inputs reach the
            # constructor unvalidated, so the rejection tests exercise the constructor.
            chunk_field = "horizon_length" if cls is FlowPPO else "act_steps"
            default = cls.params_class.__dataclass_fields__[chunk_field].default
            steps = kwargs.get(chunk_field, getattr(kwargs.get("params"), chunk_field, default))
            instance = ActionChunkWrapper(instance, act_steps=steps)
        if cls is GAIL:
            kwargs["demo_env_id"] = None
        if cls is PolicyDistillation:
            kwargs.update(teacher_policy=None, teacher_obs_keys=("privileged",),
                          student_obs_keys=("student_vec",))
        return cls(instance, num_steps=4, device="cpu", std_log=False, **kwargs)
    if cls is UniO4OPE:
        kwargs["task"] = "halfcheetah-medium-v2"
    if cls in SKILL_CLASSES:
        import tempfile
        from pathlib import Path
        from test_hilp import _write_h5

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            _write_h5(path, num_traj=1, length=64)
            return cls(env(), dataset_path=str(path), batch_size=4,
                       device="cpu", std_log=False, **kwargs)
    if cls is DAgger:
        return cls(env(), expert=_ConstantExpert(), buffer_size=32,
                   demo_buffer_size=64, batch_size=4, buffer_device="cpu",
                   device="cpu", std_log=False, **kwargs)
    if cls in CHUNK_BC_CLASSES:
        import tempfile
        from pathlib import Path

        from rl_garden.hyperparameters import params_kwargs
        from test_diffusion_bc_smoke import _write_h5_dataset
        from test_a2a_bc import _write_vision_h5_dataset, _test_encoder_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            if cls is A2ABC:
                _write_vision_h5_dataset(path, num_traj=1, steps_per_traj=64,
                                         state_dim=4, action_dim=2)
                kwargs["encoder_config"] = _test_encoder_config
            else:
                _write_h5_dataset(path, num_traj=1, steps_per_traj=64,
                                  obs_dim=6, action_dim=3)
            if cls is ConsistencyDistillBC:
                # The teacher gets the settings it shares with the student; the student's
                # inputs reach its constructor unvalidated, so rejection tests exercise it.
                teacher_fields = {f.name for f in dataclasses.fields(DiffusionBC.params_class)}
                given = kwargs.get("params")
                shared = params_kwargs(
                    given if isinstance(given, cls.params_class) else cls.params_class(),
                    cls.params_class)
                shared.update(kwargs)
                teacher_kwargs = {k: v for k, v in shared.items() if k in teacher_fields}
                teacher = DiffusionBC(env(), str(path), batch_size=4, device="cpu",
                                      std_log=False, **teacher_kwargs)
                kwargs["bc_checkpoint"] = str(teacher.save(Path(directory) / "teacher.pt"))
            return cls(env(), dataset_path=str(path), batch_size=4, device="cpu",
                       std_log=False, **kwargs)
    return cls(env(), buffer_size=32, batch_size=4, buffer_device="cpu",
               device="cpu", std_log=False, **kwargs)


@pytest.mark.parametrize("cls,env,field", FIELDS, ids=[f"{c.__name__}-{f}" for c, _, f in FIELDS])
def test_parameter_construction_paths(cls, env, field):
    defaults = cls.params_class()
    hint = get_type_hints(cls.params_class)[field]
    family = CLASS_FAMILY[cls]
    value = field_value(family, field, getattr(defaults, field), hint)
    small = dict(family.small)
    params_fields = {f.name for f in dataclasses.fields(cls.params_class)}
    for mode, other in (("group", "layer"), ("layer", "group")):
        if field.endswith(f"_use_{mode}_norm") and value:
            partner = field.replace(f"_use_{mode}_norm", f"_use_{other}_norm")
            if partner in params_fields:
                small[partner] = False
    if cls is HILP and field in {"p_currgoal", "p_trajgoal", "p_randomgoal"}:
        probabilities = {name: getattr(defaults, name)
                         for name in ("p_currgoal", "p_trajgoal", "p_randomgoal")}
        probabilities[field] = value
        partner = "p_trajgoal" if field == "p_randomgoal" else "p_randomgoal"
        probabilities[partner] = 1.0 - sum(v for k, v in probabilities.items() if k != partner)
        small.update({name: probability for name, probability in probabilities.items()
                      if name != field})
    if cls in BPPO_CLASSES and field in {"use_layer_norm", "use_group_norm"} and value:
        small["use_group_norm" if field == "use_layer_norm" else "use_layer_norm"] = False
    values = {**small, field: value}
    params = cls.params_class(**values)
    if (cls in (RecurrentSAC, TransformerSAC) and field == "nstep") or (cls is ACRLPD and field == "actor_q_agg"):
        message = "uses forward_len" if field == "nstep" else "does not take actor_q_agg"
        for keywords in (values, {"params": params}, {"params": cls.params_class(**small), field: value}):
            with pytest.raises(ValueError, match=message):
                construct(cls, env, **keywords)
        return
    agents = (
        construct(cls, env, **values),
        construct(cls, env, params=params),
        construct(cls, env, params=cls.params_class(**small), **{field: value}),
    )
    for agent in agents:
        assert getattr(agent.params, field) == value
        expected = list(value) if field in {"net_arch", "critic_flow_net_arch", "mlp_dims", "decoder_net_arch", "flow_hidden_dims", "dynamics_hidden_dims", "dynamics_weight_decay", "actor_mlp_dims", "critic_mlp_dims", "cm_mlp_dims", "log_std_init_per_dim", "action_mean_bias_init"} else value
        assert getattr(agent, field) == expected
        metadata = agent._checkpoint_metadata()
        if cls is GAIL and field == "disc_net_arch":
            expected = list(value)
        assert {f.name for f in dataclasses.fields(cls.params_class)} <= metadata.keys()
        if cls in (FAC, _FACRolloutTrainingShell, Off2OnFAC) and field == "bc_pretrain_steps":
            expected = None  # Metadata records the lazily resolved prelude budget.
        if cls is DiffusionBC and field == "ema_start_step" and value is None:
            expected = 20 * max(1, agent._dataset_size // agent.batch_size)
        if (cls is PWM and field in {"actor_units", "critic_units"}) or (cls is SuperMimic and field == "actor_betas"):
            expected = list(expected)
        if cls is NCRL and field == "guidance_schedule":
            expected = list(value)
        assert metadata[field] == expected
    assert getattr(params, field) == value


@pytest.mark.parametrize("cls,env", CLASSES, ids=[c.__name__ for c, _ in CLASSES])
def test_unknown_parameter_rejected(cls, env):
    with pytest.raises(TypeError, match="unknown_parameter"):
        construct(cls, env, unknown_parameter=1)


@pytest.mark.parametrize("key", ENTRY_KEYS, ids=["/".join(k) for k in ENTRY_KEYS])
def test_parameter_fields_reach_builder_and_defaults_are_explicit(key):
    entry = ENTRIES[key]
    params_class = entry.algorithm_cls().params_class
    args_fields = {f.name for f in dataclasses.fields(entry.args_cls)}
    defaults = params_class()
    hints = get_type_hints(params_class)
    for field in dataclasses.fields(params_class):
        assert field.name in args_fields
        args = entry.args_cls()
        default = getattr(args, field.name)
        assert (default == getattr(defaults, field.name)
                or field.name in vars(entry.args_cls).get("__annotations__", {})
                or (key == ("online", "dreamer_v3")
                    and field.name == "discount_horizon" and default is None))
        family = CLASS_FAMILY[entry.algorithm_cls()]
        value = field_value(family, field.name, default, hints[field.name])
        setattr(args, field.name, value)
        if key == ("online", "acrlpd") and field.name == "nstep":
            with pytest.raises(ValueError, match="does not take nstep"):
                capture_builder(*key, entry, args=args)
            continue
        if key == ("offline", "pwm"):
            from unittest.mock import patch
            with patch("rl_garden.training.offline.entrypoints.pwm.pwm_task_layout") as layout:
                layout.return_value.task_idx = 0
                layout.return_value.buffer_size = 32
                layout.return_value.num_eval_steps = 8
                _, _, captured = capture_builder(*key, entry, args=args)
        else:
            _, _, captured = capture_builder(*key, entry, args=args)
        if key == ("online", "sac_flow") and field.name in {
            "actor_use_layer_norm", "actor_log_std_min", "actor_log_std_mode",
        }:
            assert field.name not in captured["kwargs"]
            continue
        expected = (family.builder_net_arch_type(value)
                    if field.name == "net_arch" and value is not None else value)
        if key == ("online", "sac_flow") and field.name == "flow_hidden_dims":
            expected = list(value)
        assert captured["kwargs"][field.name] == expected
        if field.name == "net_arch":
            assert type(captured["kwargs"][field.name]) is type(expected)


@pytest.mark.parametrize("key", ENTRY_KEYS, ids=["/".join(k) for k in ENTRY_KEYS])
def test_entrypoint_import_keeps_algorithms_lazy(key):
    module = ENTRIES[key].args_cls.__module__
    code = (f"import {module}; import sys; "
            "assert 'rl_garden.algorithms' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("key", [("offline", "awac"), ("off2on", "awac")])
def test_awac_optional_architecture_and_constructor_choices_parse(key):
    import importlib

    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    args = registry.parse_args([key[1], "--net-arch", "None", "--grad-clip-norm", "None",
                                "--kernel-init", "orthogonal_near_zero_output"])
    assert args.net_arch is None
    assert args.grad_clip_norm is None
    assert args.kernel_init == "orthogonal_near_zero_output"
    args = registry.parse_args([key[1], "--net-arch", "32", "64",
                                "--std-parameterization", "uniform"])
    assert args.net_arch == (32, 64)
    assert args.std_parameterization == "uniform"


@pytest.mark.parametrize("key", ENTRY_KEYS, ids=["/".join(k) for k in ENTRY_KEYS])
def test_args_as_params_does_not_override_operational_keywords(key):
    entry = ENTRIES[key]
    params = entry.args_cls()
    params.buffer_size = 1000
    params.buffer_device = "cuda"
    family = next(family for family in FAMILIES if key in family.entries)
    for name, value in family.small.items():
        setattr(params, name, value)
    if hasattr(params, "net_arch"):
        params.net_arch = (32, 32)
    env = family.env
    agent = construct(entry.algorithm_cls(), env, params=params)
    if entry.algorithm_cls() is RSSMPretrain:
        assert agent.buffer_size == agent.replay_buffer.num_steps
        assert str(agent.buffer_device) == "cpu"
    elif entry.algorithm_cls() in ON_POLICY_CLASSES:
        assert agent.num_steps == 4
    else:
        assert agent.buffer_size == (1 if entry.algorithm_cls() in (*CHUNK_BC_CLASSES, *SKILL_CLASSES) else 32)
        assert str(agent.buffer_device) == "cpu"
    assert agent.device.type == "cpu"
    assert type(agent.params) is entry.algorithm_cls().params_class
    if hasattr(params, "net_arch"):
        assert tuple(agent.params.net_arch) == (32, 32)


@pytest.mark.parametrize("cls,env", CLASSES, ids=[c.__name__ for c, _ in CLASSES])
def test_params_of_another_class_rejected(cls, env):
    @dataclasses.dataclass
    class Unrelated:
        tau: float = 0.5

    with pytest.raises(TypeError, match="params must be a"):
        construct(cls, env, params=Unrelated())


@pytest.mark.parametrize("cls,env", [(c, e) for c, e in CLASSES if c is not IDQL and c not in (*BC_CLASSES, *SKILL_CLASSES, *IMAGE_ONLINE_CLASSES, *BPPO_CLASSES, *ON_POLICY_CLASSES, *MODEL_BASED_CLASSES, *SAC_CLASSES, *DREAMER_CLASSES, *CQL_CLASSES, FlashSAC, OfflineSAC, EDAC)],
                         ids=[c.__name__ for c, _ in CLASSES if c is not IDQL and c not in (*BC_CLASSES, *SKILL_CLASSES, *IMAGE_ONLINE_CLASSES, *BPPO_CLASSES, *ON_POLICY_CLASSES, *MODEL_BASED_CLASSES, *SAC_CLASSES, *DREAMER_CLASSES, *CQL_CLASSES, FlashSAC, OfflineSAC, EDAC)])
@pytest.mark.parametrize("field,value,message", [
    ("tau", 0.0, "tau must be in (0, 1], got 0.0."),
    ("grad_clip_norm", 0.0, "grad_clip_norm must be positive or None, got 0.0."),
])
def test_parameter_validation_stays_at_construction(cls, env, field, value, message):
    params = cls.params_class(**{**CLASS_FAMILY[cls].small, field: value})
    with pytest.raises(ValueError) as error:
        construct(cls, env, params=params)
    assert str(error.value) == message


@pytest.mark.parametrize("cls", [SPOT, _SPOTRolloutTrainingShell, Off2OnSPOT])
def test_spot_metadata_reads_live_values_without_mutating_params(cls):
    agent = construct(cls, _state_env, **SPOT_SMALL)
    snapshot = dataclasses.asdict(agent.params)
    agent.lambd = 0.37
    assert agent._checkpoint_metadata()["lambd"] == 0.37
    if cls is not SPOT:
        agent.switch_to_online_mode()
        agent.max_online_updates = 7
        agent.lambd_cool = True
        agent._global_update += 3
        assert agent.gamma == agent.online_discount
        assert agent._checkpoint_metadata()["gamma"] == agent.gamma
        assert agent._current_lambd() < agent.lambd
        if cls is Off2OnSPOT:
            state = agent._extra_checkpoint_state()
            assert state["spot_gamma"] == agent.gamma
            assert state["spot_max_online_updates"] == 7
    assert dataclasses.asdict(agent.params) == snapshot


def test_td3bc_parameter_subclasses_state_their_defaults():
    from rl_garden.hyperparameters.rebrac import ReBRACParams
    from rl_garden.hyperparameters.spot import Off2OnSPOTParams
    from rl_garden.hyperparameters.td3_bc import TD3BCParams, TD3CoreParams

    for child, parent in ((TD3BCParams, TD3CoreParams),
                          (ReBRACParams, TD3CoreParams), (SPOTParams, TD3CoreParams),
                          (Off2OnSPOTParams, SPOTParams)):
        assert issubclass(child, parent)
        for field in dataclasses.fields(parent):
            if getattr(child(), field.name) != getattr(parent(), field.name):
                assert field.name in child.__annotations__


@pytest.mark.parametrize("cls", [ReBRAC, SPOT, _SPOTRolloutTrainingShell, Off2OnSPOT])
def test_td3bc_only_alpha_is_absent_from_other_family_members(cls):
    assert "alpha" not in {field.name for field in dataclasses.fields(cls.params_class)}
    with pytest.raises(TypeError, match="unexpected keyword argument.*alpha"):
        cls(None, alpha=2.5)
    agent = construct(cls, _state_env, **CLASS_FAMILY[cls].small)
    assert not hasattr(agent, "alpha")
    assert "alpha" not in agent._checkpoint_metadata()


@pytest.mark.parametrize("key", [("offline", "rebrac"), ("offline", "spot"),
                                 ("off2on", "spot")])
def test_unused_alpha_is_absent_from_args_and_builder(key):
    import importlib

    entry = ENTRIES[key]
    assert "alpha" not in {field.name for field in dataclasses.fields(entry.args_cls)}
    _, _, captured = capture_builder(*key, entry)
    assert "alpha" not in captured["kwargs"]
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    with pytest.raises(SystemExit):
        registry.parse_args([key[1], "--alpha", "2.5"])


@pytest.mark.parametrize("key", [("offline", "td3_bc"), ("offline", "rebrac"),
                                 ("offline", "spot"), ("off2on", "spot")])
def test_td3bc_family_optional_architecture_and_initializer_parse(key):
    import importlib

    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    args = registry.parse_args([key[1], "--net-arch", "None", "--grad-clip-norm", "None",
                                "--kernel-init", "orthogonal_near_zero_output"])
    assert args.net_arch is None
    assert args.grad_clip_norm is None
    assert args.kernel_init == "orthogonal_near_zero_output"
    args = registry.parse_args([key[1], "--net-arch", "32", "64"])
    assert args.net_arch == (32, 64)


@pytest.mark.parametrize("key", ENTRY_KEYS, ids=["/".join(k) for k in ENTRY_KEYS])
@pytest.mark.parametrize("flags,code", [(["--help"], 0),
                                       (["--definitely-not-a-flag", "1"], 2)])
def test_migrated_entry_cli_exit_codes(key, flags, code, capsys):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    with pytest.raises(SystemExit) as error:
        registry.parse_args([key[1], *flags])
    assert error.value.code == code
    output = capsys.readouterr()
    assert output.out or output.err


@pytest.mark.parametrize("phase", ["online", "offline", "off2on"])
def test_registry_unknown_flag_exits_two(phase, capsys):
    registry = importlib.import_module(f"rl_garden.training.{phase}.registry").registry
    registry.discover()
    name = next(iter(registry.entries()))
    with pytest.raises(SystemExit) as error:
        registry.parse_args([name, "--definitely-not-a-flag", "1"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out or output.err


@pytest.mark.parametrize("key", [("offline", "iql"), ("off2on", "iql")])
@pytest.mark.parametrize("field", ["net_arch"])
def test_iql_new_architecture_flags_parse(key, field):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    flag = "--" + field.replace("_", "-")
    args = registry.parse_args([key[1], flag, "32", "64"])
    assert getattr(args, field) == (32, 64)
    args = registry.parse_args([key[1], flag, "None"])
    assert getattr(args, field) is None


@pytest.mark.parametrize("cls", [IQL, _IQLRolloutTrainingShell, Off2OnIQL])
def test_iql_metadata_reads_live_values_without_mutating_params(cls):
    agent = construct(cls, _state_env, net_arch=(32, 32))
    snapshot = dataclasses.asdict(agent.params)
    agent.temperature = 1.7
    agent.expectile = 0.6
    if cls is not IQL:
        agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
        state = agent._extra_checkpoint_state()
        agent._load_extra_checkpoint_state(state)
    assert agent._checkpoint_metadata()["temperature"] == 1.7
    assert agent._checkpoint_metadata()["expectile"] == 0.6
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("cls", [IQL, _IQLRolloutTrainingShell, Off2OnIQL])
@pytest.mark.parametrize("overrides,message", [
    ({"expectile": 1.0}, "expectile must be in (0, 1), got 1.0."),
    ({"temperature": 0.0}, "temperature must be positive, got 0.0."),
    ({"adv_clip_max": 0.0}, "adv_clip_max must be positive, got 0.0."),
    ({"actor_lr_schedule": "warmup_cosine"},
     "warmup_cosine requires a positive decay step count for the actor "
     "schedule; set actor_lr_decay_steps (or lr_decay_steps, which the "
     "actor schedule falls back to), got 0."),
])
def test_iql_parameter_validation(cls, overrides, message):
    with pytest.raises(ValueError) as error:
        construct(cls, _state_env, params=cls.params_class(**overrides))
    assert str(error.value) == message


@pytest.mark.parametrize("key", [("offline", "iql"), ("off2on", "iql")])
@pytest.mark.parametrize("field", ["actor_hidden_dims", "critic_hidden_dims", "value_hidden_dims"])
def test_iql_deprecated_architecture_flags_rejected(key, field):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    assert field not in {f.name for f in dataclasses.fields(ENTRIES[key].args_cls)}
    with pytest.raises(SystemExit) as error:
        registry.parse_args([key[1], "--" + field.replace("_", "-"), "32", "64"])
    assert error.value.code == 2


@pytest.mark.parametrize("cls", [IQL, _IQLRolloutTrainingShell, Off2OnIQL])
@pytest.mark.parametrize("net_arch", [None, (32, 32)])
def test_iql_deprecated_architecture_keywords_keep_resolution_and_precedence(cls, net_arch):
    aliases = {"actor_hidden_dims": (16, 16), "critic_hidden_dims": (24, 24),
               "value_hidden_dims": (32, 32)}
    with pytest.warns(DeprecationWarning, match="actor_hidden_dims"):
        agent = construct(cls, _state_env, net_arch=net_arch, **aliases)
    expected = ({"pi": [16, 16], "qf": [24, 24], "vf": [32, 32]}
                if net_arch is None else [32, 32])
    assert agent.net_arch == expected
    for field in aliases:
        assert field not in {f.name for f in dataclasses.fields(cls.params_class)}
        assert not hasattr(agent, field)
        assert field not in agent._checkpoint_metadata()


@pytest.mark.parametrize("key", [("offline", "iql"), ("off2on", "iql")])
def test_iql_leaf_overrides_only_defaults_and_uses_supported_initializer_type(key):
    from rl_garden.hyperparameters.iql import IQLParams
    from rl_garden.networks.mlp import KernelInit

    assert get_type_hints(IQLParams)["kernel_init"] == KernelInit | None
    leaf = ENTRIES[key].args_cls
    fields = {f.name for f in dataclasses.fields(IQLParams)}
    declared = fields & leaf.__annotations__.keys()
    assert declared == {"n_critics", "critic_subsample_size", "actor_use_layer_norm",
                        "critic_use_layer_norm"}
    assert all(getattr(leaf(), name) != getattr(IQLParams(), name) for name in declared)
    assert get_type_hints(leaf)["critic_subsample_size"] == get_type_hints(IQLParams)["critic_subsample_size"]


@pytest.mark.parametrize("cls", [FQL, Off2OnFQL, FloQ, Off2OnFloQ, FINO, Off2OnFINO,
                                 FAC, Off2OnFAC, ValueFlows, Off2OnValueFlows, ACFQL])
def test_fql_family_params_exclude_ineffective_compatibility_keywords(cls):
    fields = {field.name for field in dataclasses.fields(cls.params_class)}
    assert "actor_use_group_norm" not in fields
    if cls in (FAC, Off2OnFAC):
        assert "alpha" not in fields
    if cls in (ValueFlows, Off2OnValueFlows):
        assert not fields & {"critic_use_group_norm", "num_groups", "critic_dropout_rate"}
    agent = construct(cls, _state_env, **CLASS_FAMILY[cls].small)
    assert agent.actor_use_group_norm is False
    assert agent._checkpoint_metadata()["alpha"] == agent.alpha


@pytest.mark.parametrize("cls", [FINO, Off2OnFINO, FAC, Off2OnFAC,
                                 ValueFlows, Off2OnValueFlows, FQL, Off2OnFQL])
def test_fql_family_metadata_reads_live_values(cls):
    agent = construct(cls, _state_env, **CLASS_FAMILY[cls].small)
    snapshot = dataclasses.asdict(agent.params)
    if cls in (FINO, Off2OnFINO):
        assert agent.params.num_samples is None
        agent.policy.beta = 2.5
        assert agent._checkpoint_metadata()["beta"] == 2.5
        assert isinstance(agent.num_samples, int)
    elif cls in (FAC, Off2OnFAC):
        agent.fac_lambda = 0.37
        assert agent._checkpoint_metadata()["fac_lambda"] == 0.37
        if cls is Off2OnFAC:
            agent.switch_to_online_mode()
            assert agent._online_finetuning
            agent._load_extra_checkpoint_state(agent._extra_checkpoint_state())
    elif cls in (ValueFlows, Off2OnValueFlows):
        if cls is Off2OnValueFlows:
            agent.switch_to_online_mode()
            assert agent.policy_extraction == "rpg"
        assert agent._checkpoint_metadata()["policy_extraction"] == agent.policy_extraction
    else:
        agent.reference_compatible = True
        assert agent._checkpoint_metadata()["reference_compatible"] is True
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("key", [(phase, name) for phase in ("offline", "off2on")
                                  for name in ("fql", "floq", "fino", "fac", "value_flows", "acfql")
                                  if (phase, name) in ENTRIES])
def test_fql_family_optional_architecture_and_initializer_flags_parse(key):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    args = registry.parse_args([key[1], "--net-arch", "32", "64",
                                "--kernel-init", "orthogonal_near_zero_output"])
    assert args.net_arch == (32, 64)
    assert args.kernel_init == "orthogonal_near_zero_output"
    args = registry.parse_args([key[1], "--net-arch", "None", "--grad-clip-norm", "None"])
    assert args.net_arch is None
    assert args.grad_clip_norm is None
    if key[1] == "floq":
        args = registry.parse_args([key[1], "--critic-flow-net-arch", "32", "64"])
        assert args.critic_flow_net_arch == (32, 64)
        assert registry.parse_args([key[1], "--critic-flow-net-arch", "None"]).critic_flow_net_arch is None


def test_fql_parameter_hierarchy_keeps_omitted_fields_out_of_children():
    from rl_garden.hyperparameters.fql import (
        ACFQLParams, FACParams, FINOParams, FQLCriticParams, FQLDistillationParams,
        FQLParams, FQLReferenceParams, FloQParams, ValueFlowsParams,
    )

    for child in (FQLCriticParams, FQLDistillationParams, FQLReferenceParams,
                  FloQParams, FINOParams, FACParams, ValueFlowsParams, ACFQLParams):
        assert issubclass(child, FQLParams)
        for parent in child.__bases__:
            for field in dataclasses.fields(parent):
                if getattr(child(), field.name) != getattr(parent(), field.name):
                    assert field.name in child.__annotations__
    assert "alpha" not in {field.name for field in dataclasses.fields(FQLParams)}
    assert "alpha" not in {field.name for field in dataclasses.fields(FACParams)}
    assert "reference_compatible" not in {field.name for field in dataclasses.fields(FloQParams)}


@pytest.mark.parametrize("name", ["qam", "qgf", "bcq", "plas", "idql"])
def test_independent_offline_new_and_widened_flags_parse(name):
    registry = importlib.import_module("rl_garden.training.offline.registry").registry
    if name == "idql":
        args = registry.parse_args([name, "--critic-hidden-dims", "16", "32",
                                    "--value-hidden-dims", "24", "32",
                                    "--critic-subsample-size", "None"])
        assert args.critic_hidden_dims == (16, 32)
        assert args.value_hidden_dims == (24, 32)
        assert args.critic_subsample_size is None
    else:
        args = registry.parse_args([name, "--net-arch", "None",
                                    "--kernel-init", "orthogonal_near_zero_output"])
        assert args.net_arch is None
        assert args.kernel_init == "orthogonal_near_zero_output"
        assert registry.parse_args([name, "--net-arch", "16", "32"]).net_arch == (16, 32)
    if name in ("qam", "qgf"):
        assert registry.parse_args([name, "--activation-fn", "mish"]).activation_fn == "mish"


@pytest.mark.parametrize("cls", [QAM, QGF, BCQ, PLAS, IDQL])
def test_independent_offline_metadata_reads_live_values(cls):
    agent = construct(cls, _state_env, **CLASS_FAMILY[cls].small)
    snapshot = dataclasses.asdict(agent.params)
    agent.actor_lr = 0.00017
    assert agent._checkpoint_metadata()["actor_lr"] == 0.00017
    assert dataclasses.asdict(agent.params) == snapshot
    assert "horizon_length" not in snapshot
    if cls is BCQ:
        assert agent._checkpoint_metadata()["num_target_candidates"] == 10
        assert "num_target_candidates" not in snapshot


@pytest.mark.parametrize("cls", BC_CLASSES)
def test_bc_metadata_stays_live(cls):
    family = CLASS_FAMILY[cls]
    agent = construct(cls, family.env, **family.small)
    snapshot = dataclasses.asdict(agent.params)
    field = "cm_lr" if cls is ConsistencyDistillBC else "actor_lr"
    setattr(agent, field, 0.00017)
    assert agent._checkpoint_metadata()[field] == 0.00017
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("cls", BC_CLASSES)
def test_bc_clip_validation_is_unchanged(cls):
    family = CLASS_FAMILY[cls]
    field = "cm_grad_clip_norm" if cls is ConsistencyDistillBC else "grad_clip_norm"
    with pytest.raises(ValueError, match=f"{field} must be positive or None, got 0.0."):
        construct(cls, family.env, **{**family.small, field: 0.0})


BC_NEW_FLAGS = [
    ((phase, name), field)
    for phase, name, fields in (
        ("offline", "bc", ("net_arch",)),
        ("online", "dagger", (
            "actor_dropout_rate", "actor_use_group_norm", "actor_use_layer_norm",
            "backbone_type", "grad_clip_norm", "kernel_init", "lr_decay_steps",
            "lr_min_ratio", "lr_schedule", "lr_warmup_steps", "num_groups",
            "std_parameterization", "use_adamw")),
        ("offline", "diffusion_bc", ("mlp_dims",)),
        ("offline", "consistency_distill_bc", ("mlp_dims",)),
        ("offline", "a2a_bc", (
            "cnn_activation_fn", "decoder_activation_fn", "decoder_kernel_init",
            "flow_activation_fn", "flow_kernel_init", "flow_use_layer_norm")),
    )
    for field in fields
]


ON_POLICY_NEW_FLAGS = [
    (("online", name), field)
    for name, fields in (
        ("ppo", ("net_arch",)),
        ("recurrent_ppo", ("net_arch",)),
        ("transformer_ppo", ("net_arch",)),
        ("gail", ("net_arch", "disc_use_adamw")),
        ("dppo", ("actor_mlp_dims", "critic_mlp_dims")),
        ("flow_ppo", ("actor_mlp_dims", "critic_mlp_dims")),
        ("diffusion_cm_distill_online", ("actor_mlp_dims", "critic_mlp_dims")),
        ("policy_distillation", ("use_adamw", "actor_use_layer_norm", "actor_use_group_norm",
                                 "num_groups", "actor_dropout_rate", "kernel_init",
                                 "backbone_type", "std_parameterization")),
    )
    for field in fields
]


@pytest.mark.parametrize("key,field", BC_NEW_FLAGS + ON_POLICY_NEW_FLAGS)
def test_bc_new_flags_parse(key, field):
    entry = ENTRIES[key]
    params_class = entry.algorithm_cls().params_class
    hint = get_type_hints(params_class)[field]
    family = CLASS_FAMILY[entry.algorithm_cls()]
    value = field_value(family, field, getattr(params_class(), field), hint)
    flag = "--" + field.replace("_", "-")
    if isinstance(value, bool):
        flags = [flag if value else "--no-" + field.replace("_", "-")]
    else:
        flags = [flag, *(str(v) for v in value)] if isinstance(value, tuple) else [flag, str(value)]
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    assert getattr(registry.parse_args([key[1], *flags]), field) == value
    if type(None) in get_args(hint):
        assert getattr(registry.parse_args([key[1], flag, "None"]), field) is None


@pytest.mark.parametrize("cnn_activation,expected", [("relu", "relu"), ("mish", "mish"), (None, "gelu")])
def test_a2a_legacy_activation_mapping_stays_visible(cnn_activation, expected):
    entry = ENTRIES["offline", "a2a_bc"]
    args = entry.args_cls(activation_fn="gelu", cnn_activation_fn=cnn_activation)
    _, _, captured = capture_builder("offline", "a2a_bc", entry, args=args)
    assert captured["kwargs"]["cnn_activation_fn"] == expected


@pytest.mark.parametrize("cls", (*IMAGE_ONLINE_CLASSES, *BPPO_CLASSES))
def test_ddpg_params_gradient_clip_validation(cls):
    with pytest.raises(ValueError, match="grad_clip_norm must be positive or None, got 0.0."):
        construct(cls, CLASS_FAMILY[cls].env, **CLASS_FAMILY[cls].small, grad_clip_norm=0.0)


@pytest.mark.parametrize("key", [("online", "drqv2"), ("online", "td3")])
def test_ddpg_new_schedule_flags_parse(key):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    args = registry.parse_args([key[1], "--lr-schedule", "warmup_cosine",
                                "--lr-warmup-steps", "2", "--lr-decay-steps", "20",
                                "--lr-min-ratio", "0.1"])
    assert (args.lr_schedule, args.lr_warmup_steps, args.lr_decay_steps, args.lr_min_ratio) == (
        "warmup_cosine", 2, 20, 0.1)


@pytest.mark.parametrize("cls,env", [(HILP, _skill_env), (OPAL, _skill_env),
                                    (DDPG, DummyDictVecEnv), (TD3, DummyDictVecEnv)])
def test_skill_and_ddpg_metadata_uses_live_values(cls, env):
    agent = construct(cls, env, **CLASS_FAMILY[cls].small)
    snapshot = dataclasses.asdict(agent.params)
    field = "lr" if cls in SKILL_CLASSES else "policy_lr"
    setattr(agent, field, 0.002)
    assert agent._checkpoint_metadata()[field] == 0.002
    assert dataclasses.asdict(agent.params) == snapshot


def test_td3_params_inherit_ddpg_settings():
    from rl_garden.hyperparameters.ddpg import DDPGParams
    from rl_garden.hyperparameters.td3 import TD3Params

    assert issubclass(TD3Params, DDPGParams)
    assert set(TD3Params.__annotations__) == {"policy_freq", "target_noise_std", "target_noise_clip"}
    assert all(getattr(TD3Params(), f.name) == getattr(DDPGParams(), f.name)
               for f in dataclasses.fields(DDPGParams))


@pytest.mark.parametrize("key", [("offline", name) for name in ("bppo", "unio4", "unio4_ope")])
@pytest.mark.parametrize("canonical,legacy,default,value", [
    ("use_layer_norm", "actor_use_layer_norm", False, True),
    ("use_group_norm", "actor_use_group_norm", False, True),
    ("dropout_rate", "actor_dropout_rate", None, 0.2),
])
def test_bppo_actor_cli_alias_precedence(key, canonical, legacy, default, value):
    entry = ENTRIES[key]
    args = entry.args_cls()
    setattr(args, canonical, default)
    setattr(args, legacy, value)
    _, _, captured = capture_builder(*key, entry, args=args)
    assert captured["kwargs"][canonical] == (value if default is None else default)
    setattr(args, canonical, value)
    setattr(args, legacy, default)
    _, _, captured = capture_builder(*key, entry, args=args)
    assert captured["kwargs"][canonical] == value


def test_bppo_params_follow_the_shared_critic_hierarchy():
    from rl_garden.hyperparameters.bppo import BPPOCriticParams, BPPOParams
    from rl_garden.hyperparameters.unio4 import UniO4Params
    from rl_garden.hyperparameters.unio4_ope import UniO4OPEParams

    assert issubclass(BPPOParams, BPPOCriticParams)
    assert issubclass(UniO4Params, BPPOCriticParams)
    assert issubclass(UniO4OPEParams, UniO4Params)
    for params in (BPPOParams, UniO4Params):
        assert not set(params.__annotations__) & set(BPPOCriticParams.__annotations__)


@pytest.mark.parametrize("name", ["bppo", "unio4", "unio4_ope"])
def test_bppo_new_actor_flags_parse(name):
    registry = importlib.import_module("rl_garden.training.offline.registry").registry
    args = registry.parse_args([name, "--use-layer-norm", "True", "--use-group-norm", "True",
                                "--dropout-rate", "0.2",
                                "--kernel-init", "orthogonal_near_zero_output"])
    assert args.use_layer_norm is True and args.use_group_norm is True
    assert args.dropout_rate == 0.2
    assert args.kernel_init == "orthogonal_near_zero_output"
    assert registry.parse_args([name, "--dropout-rate", "None"]).dropout_rate is None


@pytest.mark.parametrize("cls", ON_POLICY_CLASSES)
def test_on_policy_metadata_reads_live_values(cls):
    family = CLASS_FAMILY[cls]
    agent = construct(cls, family.env, **family.small)
    snapshot = dataclasses.asdict(agent.params)
    name = "learning_rate" if issubclass(cls, PPO) else "actor_lr"
    setattr(agent, name, 0.00017)
    assert agent._checkpoint_metadata()[name] == 0.00017
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("key", [("online", name) for name in (
    "ppo", "recurrent_ppo", "transformer_ppo", "gail", "dppo", "flow_ppo",
    "diffusion_cm_distill_online", "policy_distillation")])
def test_on_policy_optional_architecture_and_initializer_flags_parse(key):
    registry = importlib.import_module("rl_garden.training.online.registry").registry
    field = "actor_mlp_dims" if key[1] in ("dppo", "flow_ppo", "diffusion_cm_distill_online") else "net_arch"
    flag = "--" + field.replace("_", "-")
    args = registry.parse_args([key[1], flag, "32", "32", "32",
                                "--kernel-init", "orthogonal_near_zero_output"])
    assert getattr(args, field) == (32, 32, 32)
    assert args.kernel_init == "orthogonal_near_zero_output"
    assert getattr(registry.parse_args([key[1], flag, "None"]), field) is None


@pytest.mark.parametrize("cls", MODEL_BASED_CLASSES)
def test_model_based_metadata_uses_live_values(cls):
    family = CLASS_FAMILY[cls]
    agent = construct(cls, family.env, **family.small)
    snapshot = dataclasses.asdict(agent.params)
    agent.horizon = 7
    assert agent._checkpoint_metadata()["horizon"] == 7
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("cls", SAC_CLASSES)
def test_sac_metadata_collisions_and_live_values(cls):
    family = CLASS_FAMILY[cls]
    agent = construct(cls, family.env, **family.small)
    snapshot = dataclasses.asdict(agent.params)
    assert agent.alpha_lr == agent.q_lr
    metadata = agent._checkpoint_metadata()
    assert metadata["ent_coef"] == agent.ent_coef_init
    assert metadata["target_entropy"] == agent.target_entropy_arg
    assert metadata["target_entropy_value"] == agent.target_entropy
    agent.policy_lr = 0.0007
    agent.utd = 0.25
    assert agent._checkpoint_metadata()["policy_lr"] == 0.0007
    assert agent._checkpoint_metadata()["utd"] == 0.25
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("cls", (RLPD, RLPDHybrid, ExPLORe, SUPE, ACRLPD))
def test_sac_child_defaults_do_not_replace_caller_overrides(cls):
    family = CLASS_FAMILY[cls]
    values = {**family.small, "n_critics": 4, "critic_subsample_size": 1,
              "critic_use_layer_norm": False, "backup_entropy": True}
    for kwargs in (values, {"params": cls.params_class(**values)}):
        agent = construct(cls, family.env, **kwargs)
        for name in ("n_critics", "critic_subsample_size", "critic_use_layer_norm", "backup_entropy"):
            assert getattr(agent, name) == values[name]
            assert getattr(agent.params, name) == values[name]


SAC_NEW_FLAGS = [
    ('acrlpd', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('acrlpd', 'net_arch', ['--net-arch', 'None'], None),
    ('acrlpd', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('acrlpd', 'std_parameterization', ['--std-parameterization', 'uniform'], 'uniform'),
    ('acrlpd', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('rlpd', 'backup_entropy', ['--no-backup-entropy'], False),
    ('rlpd', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('rlpd', 'net_arch', ['--net-arch', 'None'], None),
    ('rlpd', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('rlpd', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('explore', 'backup_entropy', ['--no-backup-entropy'], False),
    ('explore', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('explore', 'net_arch', ['--net-arch', 'None'], None),
    ('explore', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('explore', 'relabeler_activation_fn', ['--relabeler-activation-fn', 'relu'], 'relu'),
    ('explore', 'relabeler_activation_fn', ['--relabeler-activation-fn', 'None'], None),
    ('explore', 'relabeler_kernel_init', ['--relabeler-kernel-init', 'xavier_uniform'], 'xavier_uniform'),
    ('explore', 'relabeler_kernel_init', ['--relabeler-kernel-init', 'None'], None),
    ('explore', 'rnd_activation_fn', ['--rnd-activation-fn', 'relu'], 'relu'),
    ('explore', 'rnd_activation_fn', ['--rnd-activation-fn', 'None'], None),
    ('explore', 'rnd_kernel_init', ['--rnd-kernel-init', 'xavier_uniform'], 'xavier_uniform'),
    ('explore', 'rnd_kernel_init', ['--rnd-kernel-init', 'None'], None),
    ('explore', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('sac', 'backup_entropy', ['--no-backup-entropy'], False),
    ('sac', 'grad_clip_norm', ['--grad-clip-norm', '0.1'], 0.1),
    ('sac', 'grad_clip_norm', ['--grad-clip-norm', 'None'], None),
    ('sac', 'lr_decay_steps', ['--lr-decay-steps', '1'], 1),
    ('sac', 'lr_min_ratio', ['--lr-min-ratio', '0.1'], 0.1),
    ('sac', 'lr_schedule', ['--lr-schedule', 'linear_warmup'], 'linear_warmup'),
    ('sac', 'lr_warmup_steps', ['--lr-warmup-steps', '1'], 1),
    ('sac', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('sac', 'net_arch', ['--net-arch', 'None'], None),
    ('sac', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('sac', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('sac', 'use_adamw', ['--use-adamw'], True),
    ('sac', 'weight_decay', ['--weight-decay', '0.01'], 0.01),
    ('recurrent_sac', 'backup_entropy', ['--no-backup-entropy'], False),
    ('recurrent_sac', 'grad_clip_norm', ['--grad-clip-norm', '0.1'], 0.1),
    ('recurrent_sac', 'grad_clip_norm', ['--grad-clip-norm', 'None'], None),
    ('recurrent_sac', 'lr_decay_steps', ['--lr-decay-steps', '1'], 1),
    ('recurrent_sac', 'lr_min_ratio', ['--lr-min-ratio', '0.1'], 0.1),
    ('recurrent_sac', 'lr_schedule', ['--lr-schedule', 'linear_warmup'], 'linear_warmup'),
    ('recurrent_sac', 'lr_warmup_steps', ['--lr-warmup-steps', '1'], 1),
    ('recurrent_sac', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('recurrent_sac', 'net_arch', ['--net-arch', 'None'], None),
    ('recurrent_sac', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('recurrent_sac', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('recurrent_sac', 'use_adamw', ['--use-adamw'], True),
    ('recurrent_sac', 'weight_decay', ['--weight-decay', '0.01'], 0.01),
    ('rlpd_hybrid', 'discrete_tau', ['--discrete-tau', '0.1'], 0.1),
    ('rlpd_hybrid', 'discrete_tau', ['--discrete-tau', 'None'], None),
    ('rlpd_hybrid', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('rlpd_hybrid', 'net_arch', ['--net-arch', 'None'], None),
    ('rlpd_hybrid', 'num_grippers', ['--num-grippers', '2'], 2),
    ('rlpd_hybrid', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('rlpd_hybrid', 'std_parameterization', ['--std-parameterization', 'uniform'], 'uniform'),
    ('rlpd_hybrid', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('rlpd_hybrid', 'use_grasp_penalty', ['--use-grasp-penalty'], True),
    ('sac_flow', 'backup_entropy', ['--no-backup-entropy'], False),
    ('sac_flow', 'flow_hidden_dims', ['--flow-hidden-dims', '24', '24'], (24, 24)),
    ('sac_flow', 'flow_hidden_dims', ['--flow-hidden-dims', 'None'], None),
    ('sac_flow', 'grad_clip_norm', ['--grad-clip-norm', '0.1'], 0.1),
    ('sac_flow', 'grad_clip_norm', ['--grad-clip-norm', 'None'], None),
    ('sac_flow', 'lr_decay_steps', ['--lr-decay-steps', '1'], 1),
    ('sac_flow', 'lr_min_ratio', ['--lr-min-ratio', '0.1'], 0.1),
    ('sac_flow', 'lr_schedule', ['--lr-schedule', 'linear_warmup'], 'linear_warmup'),
    ('sac_flow', 'lr_warmup_steps', ['--lr-warmup-steps', '1'], 1),
    ('sac_flow', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('sac_flow', 'net_arch', ['--net-arch', 'None'], None),
    ('sac_flow', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('sac_flow', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('sac_flow', 'use_adamw', ['--use-adamw'], True),
    ('sac_flow', 'weight_decay', ['--weight-decay', '0.01'], 0.01),
    ('supe', 'backup_entropy', ['--no-backup-entropy'], False),
    ('supe', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('supe', 'net_arch', ['--net-arch', 'None'], None),
    ('supe', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('supe', 'relabeler_activation_fn', ['--relabeler-activation-fn', 'relu'], 'relu'),
    ('supe', 'relabeler_activation_fn', ['--relabeler-activation-fn', 'None'], None),
    ('supe', 'relabeler_kernel_init', ['--relabeler-kernel-init', 'xavier_uniform'], 'xavier_uniform'),
    ('supe', 'relabeler_kernel_init', ['--relabeler-kernel-init', 'None'], None),
    ('supe', 'rnd_activation_fn', ['--rnd-activation-fn', 'relu'], 'relu'),
    ('supe', 'rnd_activation_fn', ['--rnd-activation-fn', 'None'], None),
    ('supe', 'rnd_kernel_init', ['--rnd-kernel-init', 'xavier_uniform'], 'xavier_uniform'),
    ('supe', 'rnd_kernel_init', ['--rnd-kernel-init', 'None'], None),
    ('supe', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('transformer_sac', 'backup_entropy', ['--no-backup-entropy'], False),
    ('transformer_sac', 'grad_clip_norm', ['--grad-clip-norm', '0.1'], 0.1),
    ('transformer_sac', 'grad_clip_norm', ['--grad-clip-norm', 'None'], None),
    ('transformer_sac', 'lr_decay_steps', ['--lr-decay-steps', '1'], 1),
    ('transformer_sac', 'lr_min_ratio', ['--lr-min-ratio', '0.1'], 0.1),
    ('transformer_sac', 'lr_schedule', ['--lr-schedule', 'linear_warmup'], 'linear_warmup'),
    ('transformer_sac', 'lr_warmup_steps', ['--lr-warmup-steps', '1'], 1),
    ('transformer_sac', 'net_arch', ['--net-arch', '24', '24', '24'], (24, 24, 24)),
    ('transformer_sac', 'net_arch', ['--net-arch', 'None'], None),
    ('transformer_sac', 'policy_frequency', ['--policy-frequency', '2'], 2),
    ('transformer_sac', 'target_network_frequency', ['--target-network-frequency', '2'], 2),
    ('transformer_sac', 'use_adamw', ['--use-adamw'], True),
    ('transformer_sac', 'weight_decay', ['--weight-decay', '0.01'], 0.01),
]

@pytest.mark.parametrize("name,field,flags,expected", SAC_NEW_FLAGS)
def test_sac_new_flags_parse(name, field, flags, expected):
    from rl_garden.training.online.registry import registry

    args = registry.parse_args([name, *flags])
    assert getattr(args, field) == expected


@pytest.mark.parametrize("flags,expected", [
    ([], [256, 256, 256]),
    (["--flow-hidden-dim", "64", "--flow-hidden-layers", "2"], [64, 64]),
    (["--flow-hidden-dims", "24", "24"], [24, 24]),
    # A given width list wins over the two older flags, also when it equals the
    # class default.
    (["--flow-hidden-dim", "64", "--flow-hidden-layers", "2",
      "--flow-hidden-dims", "256", "256", "256"], [256, 256, 256]),
])
def test_sac_flow_width_flags(flags, expected):
    from rl_garden.training.online.registry import registry

    key = ("online", "sac_flow")
    args = registry.parse_args(["sac_flow", *flags])
    _, _, captured = capture_builder(*key, ENTRIES[key], args=args)
    assert captured["kwargs"]["flow_hidden_dims"] == expected


def test_sac_flow_without_widths_keeps_three_layers_of_256():
    assert SACFlow.params_class().flow_hidden_dims is None
    assert SACFlow._flow_hidden_dims(None) == [256, 256, 256]
    assert SACFlow._flow_hidden_dims((8, 8)) == [8, 8]


@pytest.mark.parametrize("cls", DREAMER_CLASSES)
def test_dreamer_metadata_uses_live_values(cls):
    agent = construct(cls, None, **CLASS_FAMILY[cls].small)
    original = agent.params.kl_free
    agent.kl_free = 0.7
    assert agent._checkpoint_metadata()["kl_free"] == 0.7
    assert agent.params.kl_free == original


def test_ncrl_params_snapshot_excludes_parent_only_inputs():
    from rl_garden.hyperparameters.dreamer import DreamerV3Params, NCRLParams

    agent = construct(NCRL, None, **CLASS_FAMILY[NCRL].small, actor_ent=0.003, discount=0.98)
    assert type(agent.params) is NCRLParams
    assert not issubclass(NCRLParams, DreamerV3Params)
    assert agent.act_entropy == agent.params.actor_ent == 0.003
    assert agent.discount_horizon == 1 / (1 - 0.98)
    names = {field.name for field in dataclasses.fields(NCRLParams)}
    assert not names & {"size", "rssm_size", "stoch", "unimix", "blocks", "lr", "warmup",
                        "discount_horizon", "act_entropy", "contdisc", "rew_scale", "con_scale",
                        "policy_scale", "value_scale", "repval_scale"}
    for overrides, message in (
        ({"params": DreamerV3Params()}, "params must be a NCRLParams, got DreamerV3Params"),
        ({"lr": 0.001}, "unexpected keyword argument(s): lr"),
    ):
        with pytest.raises(TypeError) as error:
            construct(NCRL, None, **overrides)
        assert str(error.value) == message
    with pytest.raises(TypeError) as error:
        construct(DreamerV3, None, params=NCRLParams())
    assert str(error.value) == "params must be a DreamerV3Params, got NCRLParams"


@pytest.mark.parametrize("flags,expected", [
    ([], 333.0),
    (["--horizon", "25"], 25.0),
    (["--discount-horizon", "100"], 100.0),
    (["--horizon", "25", "--discount-horizon", "333"], 333.0),
    (["--horizon", "25", "--discount-horizon", "None"], 25.0),
], ids=["omitted", "legacy", "canonical", "explicit-default", "none"])
def test_dreamer_horizon_alias(flags, expected):
    from rl_garden.training.online.registry import registry

    key = ("online", "dreamer_v3")
    args = registry.parse_args(["dreamer_v3", *flags])
    _, _, captured = capture_builder(*key, ENTRIES[key], args=args)
    assert captured["kwargs"]["discount_horizon"] == expected


def test_rssm_builder_extra_parameter_wins():
    from rl_garden.training.offline.entrypoints.rssm_pretrain import build_rssm_pretrain

    class Record:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args = ENTRIES["offline", "rssm_pretrain"].args_cls()
    agent = build_rssm_pretrain(args, None, None, algorithm_cls=Record, kl_free=0.25)
    assert agent.kwargs["kl_free"] == 0.25


DREAMER_NEW_FLAGS = [
    ("online", "dreamer_v3", "discount_horizon", "100", 100.0),
    ("online", "dreamer_v3", "discount_horizon", "None", None),
    ("online", "ncrl", "reward_layers", "4", 4),
    ("online", "ncrl", "reward_units", "32", 32),
    ("offline", "rssm_pretrain", "unimix", "0.02", 0.02),
    ("offline", "rssm_pretrain", "obs_layers", "2", 2),
    ("offline", "rssm_pretrain", "img_layers", "3", 3),
    ("offline", "rssm_pretrain", "dyn_layers", "2", 2),
    ("offline", "rssm_pretrain", "decoder_layers", "4", 4),
]


@pytest.mark.parametrize("phase,name,field,text,expected", DREAMER_NEW_FLAGS)
def test_dreamer_new_flags_parse(phase, name, field, text, expected):
    registry = importlib.import_module(f"rl_garden.training.{phase}.registry").registry
    args = registry.parse_args([name, "--" + field.replace("_", "-"), text])
    assert getattr(args, field) == expected


@pytest.mark.parametrize("cls", DREAMER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("style", ["defaults", "params", "keyword"])
def test_dreamer_extended_params_construct(cls, style):
    @dataclasses.dataclass
    class Extended(cls.params_class):
        kl_free: float = 0.75
        extra: int = 7

    class ExtendedAlgorithm(cls):
        params_class = Extended

    small = CLASS_FAMILY[cls].small
    keywords = (dict(params=Extended(**small, extra=9)) if style == "params"
                else {**small, **({"extra": 9} if style == "keyword" else {})})
    agent = construct(ExtendedAlgorithm, None, **keywords)
    assert type(agent.params) is Extended
    assert agent.kl_free == agent.params.kl_free == 0.75
    assert agent.extra == agent.params.extra == (7 if style == "defaults" else 9)
    assert agent._checkpoint_metadata()["extra"] == agent.extra


@pytest.mark.parametrize("cls", DREAMER_CLASSES, ids=lambda cls: cls.__name__)
def test_dreamer_unknown_keyword_message(cls):
    with pytest.raises(TypeError) as error:
        construct(cls, None, unknown_parameter=1)
    assert str(error.value) == "unexpected keyword argument(s): unknown_parameter"


@pytest.mark.parametrize("extended", [False, True], ids=["ncrl", "subclass"])
def test_ncrl_parent_handover_preserves_snapshot(monkeypatch, extended):
    from rl_garden.hyperparameters.dreamer import DreamerV3Params, NCRLParams

    @dataclasses.dataclass
    class Extended(NCRLParams):
        extra: int = 7
        kl_free: float = 0.75

    class ExtendedNCRL(NCRL):
        params_class = Extended

    original = DreamerV3.__init__
    snapshots = []
    parent_attributes = {}

    def record(self, *args, **kwargs):
        snapshot = self.params
        assert type(snapshot) is (Extended if extended else NCRLParams)
        hyper = self._pretrained_checkpoint["metadata"]["hyperparameters"]
        expected = dataclasses.asdict(DreamerV3Params())
        expected.update({name: getattr(snapshot, name) for name in (
            "reward_bins", "kl_free", "batch_length", "train_ratio", "imag_horizon",
            "lam", "dyn_scale", "rep_scale", "recon_scale", "slow_target_fraction")})
        expected.update(stoch=int(hyper["stoch"]), unimix=float(hyper["unimix"]),
                        blocks=int(hyper["blocks"]), obs_layers=int(hyper["obs_layers"]),
                        img_layers=int(hyper["img_layers"]), dyn_layers=int(hyper["dyn_layers"]),
                        decoder_layers=int(hyper["decoder_layers"]), size="12M",
                        act_entropy=0.003, discount_horizon=1 / (1 - 0.98))
        original(self, *args, **kwargs)
        assert self.params is snapshot
        for name, value in expected.items():
            assert getattr(self, name) == value, name
        parent_attributes.update(expected)
        snapshots.append((snapshot, str(hyper["size"])))

    monkeypatch.setattr(DreamerV3, "__init__", record)
    cls = ExtendedNCRL if extended else NCRL
    agent = construct(cls, None, **CLASS_FAMILY[NCRL].small, actor_ent=0.003, discount=0.98)
    assert agent.params is snapshots[0][0]
    assert type(agent.params) is (Extended if extended else NCRLParams)
    for name, value in parent_attributes.items():
        if name != "size":
            assert getattr(agent, name) == value, name
    assert agent.size == snapshots[0][1]


@pytest.mark.parametrize("cls,env", [(FlashSAC, _flash_env),
                                     (OfflineSAC, _offline_sac_env),
                                     (EDAC, _offline_sac_env)])
@pytest.mark.parametrize("mode", ["defaults", "params", "keyword"])
def test_sac_independent_branch_extended_params(cls, env, mode):
    @dataclasses.dataclass
    class ExtendedParams(cls.params_class):
        tau: float = 0.02
        extra: int = 7

    class ExtendedAlgorithm(cls):
        params_class = ExtendedParams

    values = CLASS_FAMILY[cls].small
    if mode == "params":
        kwargs = {"params": ExtendedParams(**values, extra=9)}
    else:
        kwargs = dict(values)
        if mode == "keyword":
            kwargs["extra"] = 9
    agent = construct(ExtendedAlgorithm, env, **kwargs)
    assert type(agent.params) is ExtendedParams
    assert agent.tau == agent.params.tau == 0.02
    assert agent.extra == (7 if mode == "defaults" else 9)
    assert agent._checkpoint_metadata()["extra"] == agent.extra


@pytest.mark.parametrize("mode", ["keyword", "params", "override"])
def test_edac_critic_default_override_survives_parent(mode):
    from rl_garden.hyperparameters.offline_sac import EDACParams

    values = {"n_critics": 4, "net_arch": (16, 16)}
    kwargs = (values if mode == "keyword" else
              {"params": EDACParams(**values)} if mode == "params" else
              {"params": EDACParams(net_arch=(16, 16)), "n_critics": 4})
    agent = construct(EDAC, _offline_sac_env, **kwargs)
    assert agent.n_critics == agent.params.n_critics == 4
    assert agent._checkpoint_metadata()["n_critics"] == 4


@pytest.mark.parametrize("flags,field,value", [
    (["--policy-frequency", "2"], "policy_frequency", 2),
    (["--target-network-frequency", "3"], "target_network_frequency", 3),
    (["--no-backup-entropy"], "backup_entropy", False),
    (["--net-arch", "None"], "net_arch", None),
    (["--ent-coef", "0.2"], "ent_coef", 0.2),
    (["--target-entropy", "-2.0"], "target_entropy", -2.0),
])
def test_edac_new_cli_inputs(flags, field, value):
    from rl_garden.training.offline.registry import registry

    args = registry.parse_args(["edac", *flags])
    assert getattr(args, field) == value
    _, _, captured = capture_builder("offline", "edac", ENTRIES["offline", "edac"], args=args)
    assert captured["kwargs"][field] == value


@pytest.mark.parametrize("cls", [OfflineSAC, EDAC])
@pytest.mark.parametrize("style", ["keyword", "params"])
def test_offline_sac_gradient_clip_validation(cls, style):
    values = {"grad_clip_norm": 0.0, "net_arch": (16, 16)}
    kwargs = values if style == "keyword" else {"params": cls.params_class(**values)}
    with pytest.raises(ValueError, match="grad_clip_norm must be positive or None, got 0.0."):
        construct(cls, _offline_sac_env, **kwargs)


@pytest.mark.parametrize("style", ["keyword", "params"])
def test_edac_eta_validation(style):
    values = {"eta": -1.0}
    kwargs = values if style == "keyword" else {"params": EDAC.params_class(**values)}
    with pytest.raises(ValueError, match="eta must be >= 0, got -1.0."):
        construct(EDAC, _offline_sac_env, **kwargs)


@pytest.mark.parametrize("cls,env", [(FlashSAC, _flash_env),
                                     (OfflineSAC, _offline_sac_env),
                                     (EDAC, _offline_sac_env)])
def test_sac_independent_branch_defaults_and_live_metadata(cls, env):
    agent = construct(cls, env)
    assert agent.params == cls.params_class()
    snapshot = dataclasses.asdict(agent.params)
    agent.tau = 0.03
    metadata = agent._checkpoint_metadata()
    assert metadata["tau"] == 0.03
    if cls is FlashSAC:
        assert metadata["target_entropy"] == agent.target_entropy
        assert not hasattr(agent, "q_mc_diagnostics")
    else:
        assert agent.alpha_lr == agent.q_lr
        assert agent.net_arch == [256, 256, 256]
        assert agent.ent_coef_init == "auto"
        assert metadata["target_entropy"] == "auto"
        assert metadata["target_entropy_value"] == agent.target_entropy
        assert agent.utd == 1.0
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("net_arch", [None, (256, 256, 256)])
def test_offline_sac_deprecated_widths_keep_caller_and_precedence(net_arch):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        agent = OfflineSAC(
            _offline_sac_env(), buffer_size=32, buffer_device="cpu", batch_size=4,
            device="cpu", std_log=False, net_arch=net_arch,
            actor_hidden_dims=(16, 16), critic_hidden_dims=(24, 24),
        )
    assert caught and all(w.category is DeprecationWarning for w in caught)
    assert all(w.filename == __file__ for w in caught)
    assert agent.net_arch == ({"pi": [16, 16], "qf": [24, 24]} if net_arch is None
                              else [256, 256, 256])


def test_flash_sac_numeric_entropy_cli_input():
    from rl_garden.training.online.registry import registry

    args = registry.parse_args(["flash_sac", "--target-entropy", "-2.0"])
    assert args.target_entropy == -2.0
    _, _, captured = capture_builder("online", "flash_sac", ENTRIES["online", "flash_sac"], args=args)
    assert captured["kwargs"]["target_entropy"] == -2.0


# Every registered entry declares and inherits its learning schema.
PARAMS_REGISTRY_ALLOWANCES = {}


@pytest.mark.parametrize("key", sorted(ENTRIES))
def test_every_registered_entry_inherits_algorithm_params(key):
    entry = ENTRIES[key]
    schema = entry.algorithm_cls().params_class
    valid = dataclasses.is_dataclass(schema) and issubclass(entry.args_cls, schema)
    if key in PARAMS_REGISTRY_ALLOWANCES:
        assert PARAMS_REGISTRY_ALLOWANCES[key].strip()
        assert not valid, "stale registry Params allowance"
    else:
        assert valid, key


@pytest.mark.parametrize("cls", (CQL, CalQL, WSRL, Off2OnCalQL))
@pytest.mark.parametrize("style", ("default", "params", "keyword"))
def test_cql_extended_params_schema(cls, style):
    @dataclasses.dataclass
    class ExtendedParams(cls.params_class):
        tau: float = 0.02
        extra_weight: float = 7.0

    class Extended(cls):
        params_class = ExtendedParams

    kwargs = dict(CQL_SMALL)
    if style == "params":
        kwargs = {"params": ExtendedParams(**kwargs, extra_weight=9.0)}
    elif style == "keyword":
        kwargs["extra_weight"] = 9.0
    agent = construct(Extended, _offline_sac_env, **kwargs)
    assert agent.tau == agent.params.tau == 0.02
    expected = 7.0 if style == "default" else 9.0
    assert agent.extra_weight == agent.params.extra_weight == expected
    assert agent._checkpoint_metadata()["extra_weight"] == expected


@pytest.mark.parametrize("cls", (CQL, CalQL, WSRL, Off2OnCalQL))
@pytest.mark.parametrize("as_params", (False, True))
def test_cql_gradient_clip_validation_reaches_constructor(cls, as_params):
    values = {**CQL_SMALL, "grad_clip_norm": 0.0}
    kwargs = {"params": cls.params_class(**values)} if as_params else values
    with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
        construct(cls, _offline_sac_env, **kwargs)


def test_plain_cql_rejects_calql_keyword():
    with pytest.raises(TypeError, match="use_calql"):
        construct(CQL, _offline_sac_env, use_calql=True)


@pytest.mark.parametrize("cls", (CalQL, WSRL, Off2OnCalQL))
def test_calql_tuple_input_and_setup_use_calql_order(cls):
    class Observed(cls):
        def _setup_model(self):
            self.use_calql_at_setup = self.use_calql
            super()._setup_model()

    agent = construct(Observed, _offline_sac_env, **CQL_SMALL,
                      sarsa_hidden_dims=[32, 32], use_calql=True)
    assert agent.use_calql_at_setup is False
    assert agent.use_calql is True
    assert agent.sarsa_hidden_dims == (32, 32)
    assert agent.params.sarsa_hidden_dims == [32, 32]


@pytest.mark.parametrize("cls", (WSRL, Off2OnCalQL))
@pytest.mark.parametrize("as_params", (False, True))
def test_cql_online_override_and_live_metadata(cls, as_params):
    values = {**CQL_SMALL, "online_cql_alpha": 0.7, "online_use_cql_loss": False}
    kwargs = {"params": cls.params_class(**values)} if as_params else values
    agent = construct(cls, _offline_sac_env, **kwargs)
    snapshot = dataclasses.asdict(agent.params)
    agent.switch_to_online_mode("append")
    agent.use_td_loss = False
    assert agent.cql_alpha == 0.7 and agent.use_cql_loss is False
    metadata = agent._checkpoint_metadata()
    assert metadata["cql_alpha"] == 0.7
    assert metadata["use_cql_loss"] is False and metadata["use_td_loss"] is False
    assert dataclasses.asdict(agent.params) == snapshot


@pytest.mark.parametrize("key", CQL_ENTRIES)
def test_cql_explicit_default_widths_override_legacy_width_flags(key):
    entry = ENTRIES[key]
    args = entry.args_cls()
    args.hidden_dim = 17
    args.net_arch = (256, 256, 256)
    _, _, captured = capture_builder(*key, entry, args=args)
    assert captured["kwargs"]["net_arch"] == (256, 256, 256)
    args.net_arch = None
    _, _, captured = capture_builder(*key, entry, args=args)
    assert captured["kwargs"]["net_arch"] == {
        "pi": [17] * args.actor_hidden_layers, "qf": [17] * args.critic_hidden_layers,
    }


@pytest.mark.parametrize("cls", (CQL, CalQL, WSRL, Off2OnCalQL))
def test_cql_default_params_snapshot_and_entropy_metadata(cls):
    agent = construct(cls, _offline_sac_env)
    assert dataclasses.asdict(agent.params) == dataclasses.asdict(cls.params_class())
    assert agent.ent_coef_init == "auto" and agent.target_entropy_arg == "auto"
    metadata = agent._checkpoint_metadata()
    assert metadata["target_entropy"] == "auto"
    assert metadata["target_entropy_value"] == agent.target_entropy


@pytest.mark.parametrize("key", CQL_ENTRIES)
def test_cql_new_flags_parse(key):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    args = registry.parse_args([key[1], "--net-arch", "32", "32", "--ent-coef", "0.2",
                                "--target-entropy", "-2.0", "--critic-subsample-size", "None"])
    assert args.net_arch == (32, 32) and args.ent_coef == 0.2
    assert args.target_entropy == -2.0 and args.critic_subsample_size is None
    assert registry.parse_args([key[1], "--net-arch", "None"]).net_arch is None


@pytest.mark.parametrize("cls", (CQL, CalQL, WSRL, Off2OnCalQL))
@pytest.mark.parametrize("net_arch", (None, (256, 256)))
def test_cql_deprecated_widths_keep_precedence_and_warning_frame(cls, net_arch):
    from pathlib import Path
    import rl_garden.algorithms.cql as module

    with pytest.warns(DeprecationWarning) as warnings:
        agent = construct(cls, _offline_sac_env, net_arch=net_arch,
                          actor_hidden_dims=(16, 16), critic_hidden_dims=(32, 32),
                          n_critics=3)
    expected = {"pi": [16, 16], "qf": [32, 32]} if net_arch is None else [256, 256]
    assert agent.net_arch == expected
    assert agent._checkpoint_metadata()["net_arch"] == expected
    # stacklevel=3 points from the resolver through the core helper to its constructor.
    assert Path(warnings[0].filename).resolve() == Path(module.__file__).resolve()


@pytest.mark.parametrize("key", CQL_ENTRIES)
@pytest.mark.parametrize("flag", ["--cql-action-sample-method", "--kernel-init"])
def test_cql_choice_flags_reject_unsupported_values(key, flag, capsys):
    registry = importlib.import_module(f"rl_garden.training.{key[0]}.registry").registry
    with pytest.raises(SystemExit) as error:
        registry.parse_args([key[1], flag, "bogus"])
    assert error.value.code == 2
    capsys.readouterr()


@pytest.mark.parametrize("key", CQL_ENTRIES)
def test_cql_entropy_flags_reach_the_constructor(key):
    entry = ENTRIES[key]
    _, _, captured = capture_builder(*key, entry)
    assert captured["kwargs"]["ent_coef"] == "auto"
    assert captured["kwargs"]["target_entropy"] == "auto"
    args = entry.args_cls()
    args.ent_coef, args.target_entropy = 0.2, -2.0
    _, _, captured = capture_builder(*key, entry, args=args)
    assert captured["kwargs"]["ent_coef"] == 0.2
    assert captured["kwargs"]["target_entropy"] == -2.0


ALIAS_FIELDS = [
    (name, field, "actor_" + field, default, alternative)
    for name in ("bppo", "unio4", "unio4_ope")
    for field, default, alternative in (
        ("use_layer_norm", False, True), ("use_group_norm", False, True),
        ("dropout_rate", None, 0.2),
    )
] + [("a2a_bc", "cnn_activation_fn", "activation_fn", "relu", "gelu")]


@pytest.mark.parametrize("name,field,legacy,default,alternative", ALIAS_FIELDS)
@pytest.mark.parametrize("case", ["omitted", "legacy", "canonical", "explicit-default", "none"])
def test_offline_alias_precedence(name, field, legacy, default, alternative, case):
    entry = ENTRIES["offline", name]
    args = entry.args_cls()
    expected = getattr(args, legacy)
    if case in {"legacy", "explicit-default", "none"}:
        setattr(args, legacy, alternative)
        expected = alternative
    if case == "canonical":
        setattr(args, field, alternative)
        expected = alternative
    elif case == "explicit-default":
        setattr(args, field, default)
        expected = alternative if default is None else default
    elif case == "none":
        setattr(args, field, None)
    _, _, captured = capture_builder("offline", name, entry, args=args)
    assert captured["kwargs"][field] == expected


@pytest.mark.parametrize("name,field,legacy,default,alternative", ALIAS_FIELDS)
def test_offline_alias_cli_values(name, field, legacy, default, alternative):
    from rl_garden.training.offline.registry import registry

    flag = "--" + field.replace("_", "-")
    for value in dict.fromkeys((None, default, alternative)):
        args = registry.parse_args([name, flag, str(value)])
        assert getattr(args, field) == value
    if field == "dropout_rate":
        args = registry.parse_args([name, flag, "None", "--actor-dropout-rate", "None"])
        _, _, captured = capture_builder("offline", name, ENTRIES["offline", name], args=args)
        assert captured["kwargs"][field] is None
