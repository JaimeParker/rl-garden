"""Extension points of the SAC family: ``policy_class`` / ``_policy_kwargs``,
``_critic_features``, the step methods ``_critic_step`` / ``_actor_step`` /
``_alpha_step`` and ``_make_optimizers``."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from test_optimizer_ownership import (
    COMMON_KWARGS,
    IMAGE_ENCODER_CONFIG,
    SEQUENCE_KWARGS,
    TinyVecEnv,
    optimizer_parameters,
)
from test_sac_optimizer_snapshot import (
    RLPD_EXCLUDE_BIAS_OPTIMIZER_GROUPS,
    build_rlpd,
    fill_random_transitions,
    optimizer_group_layout,
)
from torch import nn

from rl_garden.algorithms import RLPD, SAC, RecurrentSAC, SACFlow
from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.policies.recurrent_sac_policy import RecurrentSACPolicy
from rl_garden.policies.rlpd_hybrid_policy import RLPDHybridPolicy
from rl_garden.policies.rlpd_policy import RLPDPolicy
from rl_garden.policies.sac_flow_policy import SACFlowPolicy
from rl_garden.policies.sac_policy import SACPolicy

SAC_POLICY_KWARGS = [
    "observation_space",
    "action_space",
    "net_arch",
    "n_critics",
    "critic_subsample_size",
    "critic_impl",
    "actor_use_layer_norm",
    "critic_use_layer_norm",
    "log_std_min",
    "log_std_mode",
    "actor_feature_dim",
    "critic_spatial_emb_dim",
    "critic_backbone_type",
    "actor_extractor",
    "critic_extractor",
    "encoder_sharing",
]


def image_sac(algorithm_cls: type = SAC, **overrides: Any):
    kwargs = {**COMMON_KWARGS, "encoder_config": IMAGE_ENCODER_CONFIG, "seed": 0, **overrides}
    agent = algorithm_cls(env=TinyVecEnv(with_image=True), **kwargs)
    fill_random_transitions(agent)
    return agent


# --- policy_class and _policy_kwargs ---------------------------------------------------------


class AuxHeadSACPolicy(SACPolicy):
    """A ``SACPolicy`` with a head on the critic features, trained by the
    critic loss."""

    def __init__(self, *args: Any, aux_hidden: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aux_head = nn.Sequential(
            nn.Linear(self.critic_features_dim, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, 1)
        )

    def critic_and_encoder_parameters(self):
        yield from super().critic_and_encoder_parameters()
        yield from self.aux_head.parameters()


class AuxHeadSAC(SAC):
    """Reads the critic features to train ``aux_head`` inside the critic step."""

    policy_class = AuxHeadSACPolicy

    def _policy_kwargs(self, observation_space) -> dict[str, Any]:
        return {**super()._policy_kwargs(observation_space), "aux_hidden": 4}

    def _critic_features(self, obs, *, target: bool) -> torch.Tensor:
        features = super()._critic_features(obs, target=target)
        self._latest_critic_features = features
        return features

    def _critic_loss(self, data):
        loss, info = super()._critic_loss(data)
        prediction = self.policy.aux_head(self._latest_critic_features).squeeze(-1)
        aux_loss = F.mse_loss(prediction, data.rewards.reshape(-1))
        info["aux_loss"] = aux_loss.detach()
        return loss + aux_loss, info


def test_policy_class_subclass_is_built_and_trained_by_the_critic_optimizer():
    agent = image_sac(AuxHeadSAC)
    assert type(agent.policy) is AuxHeadSACPolicy

    optimizers = optimizer_parameters(agent)
    aux_ids = {id(p) for p in agent.policy.aux_head.parameters()}
    assert aux_ids <= {id(p) for p in optimizers["q_optimizer"]}
    assert not aux_ids & {id(p) for p in optimizers["actor_optimizer"]}
    owned = {id(p) for params in optimizers.values() for p in params}
    assert all(id(p) in owned for p in agent.policy.parameters() if p.requires_grad)

    aux_before = [p.detach().clone() for p in agent.policy.aux_head.parameters()]
    info = agent.train(gradient_steps=1, compute_info=True)
    assert "aux_loss" in info
    assert any(
        not torch.equal(before, after)
        for before, after in zip(aux_before, agent.policy.aux_head.parameters())
    )


@pytest.mark.parametrize(
    ("algorithm_cls", "extra_kwargs", "policy_cls", "expected_keys"),
    [
        (SAC, {}, SACPolicy, SAC_POLICY_KWARGS),
        (
            RLPD,
            {},
            RLPDPolicy,
            [
                *SAC_POLICY_KWARGS,
                "actor_dropout_rate",
                "critic_dropout_rate",
                "kernel_init",
                "backbone_type",
                "use_pnorm",
                "std_parameterization",
            ],
        ),
        (
            RLPDHybrid,
            {"discrete_hidden_dim": 8},
            RLPDHybridPolicy,
            [
                *SAC_POLICY_KWARGS,
                "actor_dropout_rate",
                "critic_dropout_rate",
                "kernel_init",
                "backbone_type",
                "use_pnorm",
                "std_parameterization",
                "discrete_hidden_dims",
            ],
        ),
        (
            RecurrentSAC,
            {**SEQUENCE_KWARGS, "rnn_hidden_size": 8},
            RecurrentSACPolicy,
            [
                "observation_space",
                "action_space",
                "recurrent_encoder",
                "net_arch",
                "n_critics",
                "critic_subsample_size",
                "critic_impl",
                "actor_use_layer_norm",
                "critic_use_layer_norm",
                "log_std_min",
                "log_std_mode",
                "actor_extractor",
                "critic_extractor",
                "encoder_sharing",
            ],
        ),
        (
            SACFlow,
            {"flow_hidden_dims": [8], "denoising_steps": 2},
            SACFlowPolicy,
            [
                "observation_space",
                "action_space",
                "net_arch",
                "n_critics",
                "critic_subsample_size",
                "critic_use_layer_norm",
                "critic_impl",
                "flow_hidden_dims",
                "denoising_steps",
                "noise_std",
                "flow_use_layer_norm",
                "actor_extractor",
                "critic_extractor",
                "encoder_sharing",
            ],
        ),
    ],
)
def test_policy_kwargs_feed_the_policy_class(algorithm_cls, extra_kwargs, policy_cls, expected_keys):
    agent = algorithm_cls(env=TinyVecEnv(), **COMMON_KWARGS, **extra_kwargs)
    assert algorithm_cls.policy_class is policy_cls
    assert type(agent.policy) is policy_cls
    assert list(agent._policy_kwargs(agent.env.single_observation_space)) == expected_keys


# --- _critic_features -------------------------------------------------------------------------


class FeatureRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.recorded_critic_features: list[torch.Tensor] = []
        super().__init__(*args, **kwargs)

    def _critic_features(self, obs, *, target: bool) -> torch.Tensor:
        features = super()._critic_features(obs, target=target)
        self.recorded_critic_features.append(features)
        return features


@pytest.mark.parametrize("update_encoder", [True, False])
@pytest.mark.parametrize(
    ("utd", "gradient_steps", "rows_per_step"),
    [(0.5, 3, 4), (2, 4, 2)],
    ids=["one_batch_per_step", "high_utd_slices"],
)
def test_critic_features_seen_once_per_critic_step(update_encoder, utd, gradient_steps, rows_per_step):
    agent = image_sac(
        FeatureRecordingSAC,
        utd=utd,
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000,
            update_actor=True,
            update_critic=True,
            update_encoder=update_encoder,
        ),
    )
    agent._start_initial_training_phase()

    agent.train(gradient_steps=gradient_steps)

    recorded = agent.recorded_critic_features
    assert len(recorded) == gradient_steps
    assert all(features.shape[0] == rows_per_step for features in recorded)
    assert all(features.requires_grad is update_encoder for features in recorded)


# --- step methods -----------------------------------------------------------------------------


class StepRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.steps: list[str] = []
        super().__init__(*args, **kwargs)

    def _critic_step(self, data):
        self.steps.append("critic")
        return super()._critic_step(data)

    def _actor_step(self, data):
        self.steps.append("actor")
        return super()._actor_step(data)

    def _alpha_step(self, log_prob_detached):
        alpha_loss = super()._alpha_step(log_prob_detached)
        self.steps.append("alpha" if alpha_loss is not None else "fixed_alpha")
        return alpha_loss

    def _post_actor_update(self, data):
        self.steps.append("post_actor")
        return super()._post_actor_update(data)


CRITIC_ONLY = ["critic"]
FULL_STEP = ["critic", "actor", "alpha", "post_actor"]


@pytest.mark.parametrize(
    ("overrides", "gradient_steps", "expected"),
    [
        ({}, 2, FULL_STEP * 2),
        ({"policy_frequency": 2}, 4, (CRITIC_ONLY + FULL_STEP) * 2),
        ({"ent_coef": 0.2}, 1, ["critic", "actor", "fixed_alpha", "post_actor"]),
        # Two high-UTD groups: every critic slice, then one actor step.
        ({"utd": 2}, 4, ["critic", "critic", "actor", "alpha", "post_actor"] * 2),
        # One group and a one-step remainder at utd=1.
        ({"utd": 2}, 3, ["critic", "critic", "actor", "alpha", "post_actor", *FULL_STEP]),
    ],
    ids=["every_step", "delayed_actor", "fixed_alpha", "high_utd", "high_utd_remainder"],
)
def test_train_calls_the_step_methods_in_order(overrides, gradient_steps, expected):
    agent = StepRecordingSAC(env=TinyVecEnv(), **COMMON_KWARGS, **overrides)
    fill_random_transitions(agent)

    agent.train(gradient_steps=gradient_steps, compute_info=True)

    assert agent.steps == expected


def test_alpha_step_steps_nothing_with_a_fixed_temperature():
    agent = SAC(env=TinyVecEnv(), **COMMON_KWARGS, ent_coef=0.2)
    assert agent.alpha_optimizer is None
    assert agent._alpha_step(torch.zeros(4)) is None


# --- _make_optimizers -------------------------------------------------------------------------


def test_make_optimizers_excluding_bias_reproduces_rlpd_param_groups():
    agent = build_rlpd(exclude_bias_from_decay=False)
    assert [len(opt.param_groups) for opt in (agent.q_optimizer, agent.actor_optimizer)] == [1, 1]

    q_optimizer, actor_optimizer = agent._make_optimizers(exclude_bias_from_decay=True)

    layout = optimizer_group_layout(agent.policy, q_optimizer, actor_optimizer)
    assert layout == RLPD_EXCLUDE_BIAS_OPTIMIZER_GROUPS


def test_rlpd_schedulers_drive_the_optimizers_it_keeps():
    agent = build_rlpd(lr_schedule="linear_warmup", lr_warmup_steps=4)
    schedulers = agent._lr_schedulers
    assert [scheduler.optimizer for scheduler in schedulers] == [
        agent.q_optimizer,
        agent.actor_optimizer,
    ]


def test_rlpd_keeps_one_group_per_optimizer_without_weight_decay():
    agent = build_rlpd(weight_decay=0.0)
    assert [len(opt.param_groups) for opt in (agent.q_optimizer, agent.actor_optimizer)] == [1, 1]
