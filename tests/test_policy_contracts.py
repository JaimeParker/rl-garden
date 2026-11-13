"""The actor contracts of ``rl_garden.policies.contracts``, their validators,
``SACActorBase``'s defaults and the import-path helpers of
``rl_garden.common.effective_config``."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.common.effective_config import import_path_of, resolve_import_path
from rl_garden.networks.actor_critic import DiagGaussianActor, SquashedGaussianActor
from rl_garden.networks.sac_actor_base import SACActorBase
from rl_garden.policies.contracts import (
    PPOActor,
    PPOActorCapabilities,
    SACActor,
    SACActorCapabilities,
    validate_ppo_actor,
    validate_sac_actor,
)

FEATURES_DIM = 5
ACTION_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
ACT_DIM = 3


# --- toy actors ----------------------------------------------------------------------------


class ToyGaussianPPOActor(nn.Module):
    """A diagonal Gaussian PPO actor written against the contract alone (no
    rl-garden base class)."""

    def __init__(self, features_dim: int = FEATURES_DIM, act_dim: int = ACT_DIM) -> None:
        super().__init__()
        self.mean = nn.Linear(features_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(features)
        return torch.distributions.Normal(mean, self.log_std.expand_as(mean).exp())

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        dist = self(features)
        action = dist.mean if deterministic else dist.sample()
        log_prob, entropy = dist.log_prob(action), dist.entropy()
        if sum_dims:
            log_prob, entropy = log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        dist = self(features)
        log_prob, entropy = dist.log_prob(actions), dist.entropy()
        if sum_dims:
            log_prob, entropy = log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)
        return log_prob, entropy

    def deterministic_action(self, features):
        return self(features).mean

    def clamp_action(self, actions):
        return actions.clamp(-1.0, 1.0)


class ToySummedOnlyPPOActor(nn.Module):
    """A tanh-squashed PPO actor: no entropy, no per-dimension log-prob."""

    def __init__(self, features_dim: int = FEATURES_DIM, act_dim: int = ACT_DIM) -> None:
        super().__init__()
        self.mean = nn.Linear(features_dim, act_dim)

    def _dist(self, features):
        return torch.distributions.Normal(self.mean(features), 0.5)

    def _log_prob(self, dist, pre_tanh):
        correction = torch.log(1 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        return (dist.log_prob(pre_tanh) - correction).sum(-1, keepdim=True)

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        if not sum_dims:
            raise NotImplementedError("summed log-prob only")
        dist = self._dist(features)
        pre_tanh = dist.mean if deterministic else dist.sample()
        return torch.tanh(pre_tanh), self._log_prob(dist, pre_tanh), None

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        if not sum_dims:
            raise NotImplementedError("summed log-prob only")
        pre_tanh = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        return self._log_prob(self._dist(features), pre_tanh), None

    def deterministic_action(self, features):
        return torch.tanh(self.mean(features))

    def clamp_action(self, actions):
        return actions


class ToySACActor(SACActorBase):
    """Only the two required methods."""

    def __init__(self, features_dim: int = FEATURES_DIM, act_dim: int = ACT_DIM) -> None:
        super().__init__()
        self.mean = nn.Linear(features_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def action_log_prob(self, features):
        normal = torch.distributions.Normal(self.mean(features), self.log_std.exp())
        action = normal.rsample()
        return action, normal.log_prob(action).sum(-1, keepdim=True)

    def deterministic_action(self, features):
        return self.mean(features)


class ToyDeterministicSACActor(SACActorBase):
    stochastic = False

    def __init__(self, features_dim: int = FEATURES_DIM, act_dim: int = ACT_DIM) -> None:
        super().__init__()
        self.head = nn.Linear(features_dim, act_dim)

    def action_log_prob(self, features):
        action = self.deterministic_action(features)
        return action, action.new_zeros(features.shape[0], 1)

    def deterministic_action(self, features):
        return torch.tanh(self.head(features))


class ToyBatchNormSACActor(ToySACActor):
    """Has running statistics and dropout, which the dry pass must not touch."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(FEATURES_DIM)
        self.dropout = nn.Dropout(0.5)

    def action_log_prob(self, features):
        return super().action_log_prob(self.dropout(self.norm(features)))


def make_toy_sac_actor(features_dim: int, action_space: spaces.Box) -> nn.Module:
    """A module-level actor factory, for the import-path round trip."""
    return ToySACActor(features_dim, int(np.prod(action_space.shape)))


# --- protocols -----------------------------------------------------------------------------


def test_toy_actors_satisfy_the_runtime_checkable_protocols():
    assert isinstance(ToyGaussianPPOActor(), PPOActor)
    assert isinstance(ToySummedOnlyPPOActor(), PPOActor)
    assert isinstance(ToySACActor(), SACActor)
    assert isinstance(ToyDeterministicSACActor(), SACActor)
    assert not isinstance(nn.Linear(2, 2), PPOActor)
    assert not isinstance(nn.Linear(2, 2), SACActor)


# --- PPO validator -------------------------------------------------------------------------


def test_gaussian_ppo_actor_reports_every_capability():
    capabilities = validate_ppo_actor(ToyGaussianPPOActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    assert capabilities == PPOActorCapabilities(
        has_entropy=True, supports_per_dim_log_prob=True, is_gaussian=True
    )


def test_in_tree_diag_gaussian_actor_reports_every_capability():
    actor = DiagGaussianActor(FEATURES_DIM, ACTION_SPACE, hidden_dims=[8])
    assert validate_ppo_actor(actor, FEATURES_DIM, ACTION_SPACE, "cpu") == PPOActorCapabilities(
        has_entropy=True, supports_per_dim_log_prob=True, is_gaussian=True
    )


def test_ppo_actor_without_entropy_or_per_dim_log_prob():
    capabilities = validate_ppo_actor(ToySummedOnlyPPOActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    assert capabilities == PPOActorCapabilities(
        has_entropy=False, supports_per_dim_log_prob=False, is_gaussian=False
    )


def test_ppo_device_defaults_to_the_actors_parameters():
    capabilities = validate_ppo_actor(ToyGaussianPPOActor(), FEATURES_DIM, ACTION_SPACE)
    assert capabilities.is_gaussian


def test_ppo_actor_missing_methods_lists_all_of_them():
    class IncompletePPOActor(nn.Module):
        def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
            raise AssertionError("never called")

        def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
            raise AssertionError("never called")

    with pytest.raises(TypeError) as error:
        validate_ppo_actor(IncompletePPOActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "IncompletePPOActor" in message
    assert "deterministic_action" in message and "clamp_action" in message
    assert "action_log_prob," not in message


def test_ppo_actor_must_be_a_module():
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="torch.nn.Module"):
        validate_ppo_actor(NotAModule(), FEATURES_DIM, ACTION_SPACE, "cpu")


def test_ppo_wrong_log_prob_shape_names_the_method_and_shapes():
    class FlatLogProbActor(ToyGaussianPPOActor):
        def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
            action, log_prob, entropy = super().action_log_prob(
                features, deterministic, sum_dims=sum_dims
            )
            return action, log_prob.squeeze(-1), entropy

    with pytest.raises(ValueError) as error:
        validate_ppo_actor(FlatLogProbActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "action_log_prob" in message
    assert "(2,)" in message and "(2, 1)" in message


def test_ppo_entropy_must_be_returned_by_both_or_neither_method():
    class HalfEntropyActor(ToyGaussianPPOActor):
        def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
            log_prob, _ = super().evaluate_action_log_prob(features, actions, sum_dims=sum_dims)
            return log_prob, None

    with pytest.raises(ValueError, match="entropy"):
        validate_ppo_actor(HalfEntropyActor(), FEATURES_DIM, ACTION_SPACE, "cpu")


def test_ppo_per_dim_log_prob_must_be_supported_by_both_or_neither_method():
    class HalfPerDimActor(ToyGaussianPPOActor):
        def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
            if not sum_dims:
                raise NotImplementedError
            return super().evaluate_action_log_prob(features, actions)

    with pytest.raises(ValueError, match="sum_dims=False"):
        validate_ppo_actor(HalfPerDimActor(), FEATURES_DIM, ACTION_SPACE, "cpu")


def test_ppo_wrong_return_arity_is_a_type_error():
    class TwoTupleActor(ToyGaussianPPOActor):
        def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
            return super().action_log_prob(features, deterministic, sum_dims=sum_dims)[:2]

    with pytest.raises(TypeError, match="action_log_prob must return a tuple of 3"):
        validate_ppo_actor(TwoTupleActor(), FEATURES_DIM, ACTION_SPACE, "cpu")


def test_actor_with_non_normal_forward_is_not_gaussian():
    class MeanForwardActor(ToyGaussianPPOActor):
        def forward(self, features):
            return super().forward(features).mean

        def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
            dist = ToyGaussianPPOActor.forward(self, features)
            action = dist.mean if deterministic else dist.sample()
            log_prob, entropy = dist.log_prob(action), dist.entropy()
            if sum_dims:
                log_prob, entropy = log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)
            return action, log_prob, entropy

        def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
            dist = ToyGaussianPPOActor.forward(self, features)
            log_prob, entropy = dist.log_prob(actions), dist.entropy()
            if sum_dims:
                log_prob, entropy = log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)
            return log_prob, entropy

        def deterministic_action(self, features):
            return self(features)

    capabilities = validate_ppo_actor(MeanForwardActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    assert capabilities.has_entropy and not capabilities.is_gaussian


def test_tanh_squashed_actor_with_a_log_std_and_its_base_normal_is_not_gaussian():
    class TanhSquashedLogStdActor(ToySummedOnlyPPOActor):
        """Names its parameter ``log_std`` and returns the pre-squash Normal
        from ``forward``, which is not the distribution of its actions."""

        def __init__(self) -> None:
            super().__init__()
            self.log_std = nn.Parameter(torch.full((1, ACT_DIM), -0.7))

        def forward(self, features):
            return self._dist(features)

        def _dist(self, features):
            mean = self.mean(features)
            return torch.distributions.Normal(mean, self.log_std.expand_as(mean).exp())

    capabilities = validate_ppo_actor(TanhSquashedLogStdActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    assert capabilities == PPOActorCapabilities(
        has_entropy=False, supports_per_dim_log_prob=False, is_gaussian=False
    )


def test_ppo_evaluate_must_reproduce_the_sampled_log_prob():
    class MisScoringActor(ToyGaussianPPOActor):
        def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
            log_prob, entropy = super().evaluate_action_log_prob(
                features, actions, sum_dims=sum_dims
            )
            return log_prob - 1.0, entropy

    with pytest.raises(ValueError) as error:
        validate_ppo_actor(MisScoringActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "MisScoringActor" in message
    assert (
        "evaluate_action_log_prob does not reproduce action_log_prob's log-prob for its own sample"
        in message
    )


def test_validators_require_a_box_action_space():
    with pytest.raises(TypeError, match="Box"):
        validate_ppo_actor(ToyGaussianPPOActor(), FEATURES_DIM, spaces.Discrete(3), "cpu")


# --- SAC validator -------------------------------------------------------------------------


def test_sac_actor_with_only_the_required_methods_passes():
    capabilities = validate_sac_actor(ToySACActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    assert capabilities == SACActorCapabilities(
        stochastic=True, log_prob_is_density=True, supports_evaluate_log_prob=False
    )


def test_deterministic_sac_actor_reports_stochastic_false():
    capabilities = validate_sac_actor(ToyDeterministicSACActor(), FEATURES_DIM, ACTION_SPACE)
    assert capabilities.stochastic is False


def test_sac_actor_declaring_evaluate_log_prob_is_probed():
    class ScoringSACActor(ToySACActor):
        supports_evaluate_log_prob = True

        def evaluate_action_log_prob(self, features, actions):
            normal = torch.distributions.Normal(self.mean(features), self.log_std.exp())
            return normal.log_prob(actions).sum(-1)  # (B,), not (B, 1)

    with pytest.raises(ValueError) as error:
        validate_sac_actor(ScoringSACActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "evaluate_action_log_prob" in message
    assert "(2,)" in message and "(2, 1)" in message


@pytest.mark.parametrize("seed", range(10))
def test_cql_style_squashed_gaussian_actor_passes_the_self_score_check(seed):
    """The actor ``CQLCore`` builds (``log_std_mode="clamp"``,
    ``log_std_min=-20``) on a 4-dim state and a ``Box(-1, 1, (2,))``: its
    samples saturate tanh, where scoring them back through ``atanh`` is
    lossy in float32; saturated rows are left out of the comparison."""
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    torch.manual_seed(seed)
    actor = SquashedGaussianActor(4, action_space, [16], log_std_mode="clamp", log_std_min=-20.0)
    capabilities = validate_sac_actor(actor, 4, action_space, "cpu")
    assert capabilities.supports_evaluate_log_prob


def test_a_mis_scoring_squashed_gaussian_actor_still_fails_the_self_score_check():
    class MisScoringSquashedActor(SquashedGaussianActor):
        def evaluate_action_log_prob(self, features, actions):
            return super().evaluate_action_log_prob(features, actions) - 1.0

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    torch.manual_seed(0)
    actor = MisScoringSquashedActor(
        4, action_space, [16], log_std_mode="clamp", log_std_min=-20.0
    )
    with pytest.raises(ValueError, match="does not reproduce action_log_prob's log-prob"):
        validate_sac_actor(actor, 4, action_space, "cpu")


def test_sac_actor_declaring_evaluate_log_prob_must_reproduce_its_sampled_log_prob():
    class ConsistentSACActor(ToySACActor):
        supports_evaluate_log_prob = True

        def evaluate_action_log_prob(self, features, actions):
            normal = torch.distributions.Normal(self.mean(features), self.log_std.exp())
            return normal.log_prob(actions).sum(-1, keepdim=True)

    class MisScoringSACActor(ConsistentSACActor):
        def evaluate_action_log_prob(self, features, actions):
            return super().evaluate_action_log_prob(features, actions) - 1.0

    assert validate_sac_actor(
        ConsistentSACActor(), FEATURES_DIM, ACTION_SPACE, "cpu"
    ).supports_evaluate_log_prob
    with pytest.raises(ValueError) as error:
        validate_sac_actor(MisScoringSACActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "MisScoringSACActor" in message
    assert (
        "evaluate_action_log_prob does not reproduce action_log_prob's log-prob for its own sample"
        in message
    )


def test_sac_actor_missing_methods_and_flags_lists_all_of_them():
    class BareSACActor(nn.Module):
        def action_log_prob(self, features):
            raise AssertionError("never called")

    with pytest.raises(TypeError) as error:
        validate_sac_actor(BareSACActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    for name in (
        "deterministic_action",
        "sample_actions_with_log_prob",
        "evaluate_action_log_prob",
        "diagnostics",
        "stochastic",
        "log_prob_is_density",
        "supports_evaluate_log_prob",
    ):
        assert name in message


def test_sac_flags_must_be_bool():
    class StringFlagActor(ToySACActor):
        stochastic = "yes"

    with pytest.raises(TypeError, match="stochastic='yes'"):
        validate_sac_actor(StringFlagActor(), FEATURES_DIM, ACTION_SPACE, "cpu")


def test_sac_wrong_sample_shape_names_the_method_and_shapes():
    class WrongSamplesActor(ToySACActor):
        def sample_actions_with_log_prob(self, features, n):
            actions, log_prob = super().sample_actions_with_log_prob(features, n)
            return actions.reshape(-1, actions.shape[-1]), log_prob

    with pytest.raises(ValueError) as error:
        validate_sac_actor(WrongSamplesActor(), FEATURES_DIM, ACTION_SPACE, "cpu")
    message = str(error.value)
    assert "sample_actions_with_log_prob" in message
    assert "(4, 3)" in message and "(2, 2, 3)" in message


def test_sac_wrong_action_shape_is_a_value_error():
    class WideActionActor(ToySACActor):
        def __init__(self) -> None:
            super().__init__(act_dim=ACT_DIM + 1)

    with pytest.raises(ValueError, match=r"action_log_prob: action has shape \(2, 4\)"):
        validate_sac_actor(WideActionActor(), FEATURES_DIM, ACTION_SPACE, "cpu")


# --- dry pass isolation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("validate", "actor_cls"),
    [
        (validate_ppo_actor, ToyGaussianPPOActor),
        (validate_ppo_actor, ToySummedOnlyPPOActor),
        (validate_sac_actor, ToySACActor),
    ],
)
def test_validators_do_not_consume_the_global_rng(validate, actor_cls):
    actor = actor_cls()
    torch.manual_seed(1234)
    expected_next = torch.rand(4)

    torch.manual_seed(1234)
    state_before = torch.get_rng_state()
    validate(actor, FEATURES_DIM, ACTION_SPACE, "cpu")
    assert torch.equal(torch.get_rng_state(), state_before)
    assert torch.equal(torch.rand(4), expected_next)


def test_dry_pass_leaves_running_statistics_and_training_mode_alone():
    actor = ToyBatchNormSACActor()
    running_mean = actor.norm.running_mean.clone()
    tracked = actor.norm.num_batches_tracked.clone()
    validate_sac_actor(actor, FEATURES_DIM, ACTION_SPACE, "cpu")
    assert torch.equal(actor.norm.running_mean, running_mean)
    assert torch.equal(actor.norm.num_batches_tracked, tracked)
    assert all(module.training for module in actor.modules())

    actor.dropout.eval()
    validate_sac_actor(actor, FEATURES_DIM, ACTION_SPACE, "cpu")
    assert actor.training and actor.norm.training and not actor.dropout.training


# --- SACActorBase defaults -----------------------------------------------------------------


def test_default_sample_actions_with_log_prob_shapes_and_gradients():
    torch.manual_seed(0)
    actor = ToySACActor()
    features = torch.randn(4, FEATURES_DIM)
    actions, log_prob = actor.sample_actions_with_log_prob(features, 3)
    assert actions.shape == (4, 3, ACT_DIM)
    assert log_prob.shape == (4, 3)

    (actions.sum() + log_prob.sum()).backward()
    for name, param in actor.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().sum() > 0, name


def test_default_sample_actions_keep_each_row_with_its_features():
    actor = ToyDeterministicSACActor()
    features = torch.randn(4, FEATURES_DIM)
    actions, log_prob = actor.sample_actions_with_log_prob(features, 3)
    expected = actor.deterministic_action(features)
    for sample in range(3):
        torch.testing.assert_close(actions[:, sample], expected)
    assert torch.equal(log_prob, torch.zeros(4, 3))


def test_default_evaluate_action_log_prob_raises_with_class_and_caller():
    actor = ToySACActor()
    with pytest.raises(NotImplementedError) as error:
        actor.evaluate_action_log_prob(torch.zeros(2, FEATURES_DIM), torch.zeros(2, ACT_DIM))
    message = str(error.value)
    assert "ToySACActor cannot score external actions" in message
    assert "test_default_evaluate_action_log_prob_raises_with_class_and_caller" in message


def test_default_diagnostics_keys():
    actor = ToySACActor()
    diagnostics = actor.diagnostics(torch.randn(4, FEATURES_DIM))
    assert set(diagnostics) == {"log_prob_mean", "log_prob_std", "action_abs_mean"}
    assert all(value.ndim == 0 and not value.requires_grad for value in diagnostics.values())


def test_sac_actor_base_required_methods_raise_until_implemented():
    actor = SACActorBase()
    with pytest.raises(NotImplementedError, match="action_log_prob"):
        actor.action_log_prob(torch.zeros(1, 1))
    with pytest.raises(NotImplementedError, match="deterministic_action"):
        actor.deterministic_action(torch.zeros(1, 1))


# --- import-path helpers -------------------------------------------------------------------


def test_import_path_round_trip():
    assert import_path_of(SACActorBase) == "rl_garden.networks.sac_actor_base:SACActorBase"
    for obj in (
        SACActorBase,
        SACActorBase.sample_actions_with_log_prob,
        make_toy_sac_actor,
        ToySACActor,
    ):
        assert resolve_import_path(import_path_of(obj)) is obj
    assert resolve_import_path("rl_garden.networks.sac_actor_base.SACActorBase") is SACActorBase
    assert (
        resolve_import_path("rl_garden.networks.sac_actor_base.SACActorBase.diagnostics")
        is SACActorBase.diagnostics
    )
    assert resolve_import_path("rl_garden.networks:SACActorBase") is SACActorBase


def test_resolve_import_path_errors_name_the_spec():
    spec = "rl_garden.networks.sac_actor_base:NoSuchActor"
    with pytest.raises(AttributeError, match=spec):
        resolve_import_path(spec)
    for spec in ("rlg_no_such_package_xyz.module:factory", "rlg_no_such_package_xyz.factory"):
        with pytest.raises(ImportError, match=spec):
            resolve_import_path(spec)
    with pytest.raises(ImportError, match="pkg.mod:attr"):
        resolve_import_path("rl_garden:a:b")


def test_resolve_import_path_propagates_a_missing_dependency(tmp_path, monkeypatch):
    (tmp_path / "rlg_broken_module_xyz.py").write_text(
        "import rlg_missing_dependency_xyz\n\ndef factory():\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for spec in ("rlg_broken_module_xyz:factory", "rlg_broken_module_xyz.factory"):
        with pytest.raises(ModuleNotFoundError) as error:
            resolve_import_path(spec)
        assert error.value.name == "rlg_missing_dependency_xyz"
        assert "rlg_broken_module_xyz" not in sys.modules


def test_import_path_of_rejects_unimportable_objects():
    def local_factory():
        pass

    with pytest.raises(ValueError):
        import_path_of(lambda: None)
    with pytest.raises(ValueError, match="local_factory"):
        import_path_of(local_factory)
    with pytest.raises(ValueError):
        import_path_of(ToySACActor())
