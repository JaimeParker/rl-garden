import math

import pytest
import torch
import torch.nn as nn

from rl_garden.networks.actor_vector_field import ActorVectorField
from rl_garden.networks.flow_logprob import flow_log_prob


class _LinearVelocityField(nn.Module):
    """v(features, x, t) = x @ A.T, ignoring features and t. Divergence of a
    linear field is constant (= trace(A)) at every point, giving a closed
    form to check the reverse-Euler log-density against."""

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.matrix = matrix
        self.use_time_conditioning = True

    def forward(self, features: torch.Tensor, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del features, times
        return x @ self.matrix.T


def _base_log_prob(x: torch.Tensor) -> torch.Tensor:
    action_dim = x.shape[-1]
    return -0.5 * (x**2).sum(-1) - 0.5 * action_dim * math.log(2 * math.pi)


def _reverse_euler_x0(matrix: torch.Tensor, actions: torch.Tensor, num_steps: int) -> torch.Tensor:
    x = actions
    for _ in range(num_steps):
        x = x - (x @ matrix.T) / num_steps
    return x


def test_exact_matches_closed_form_for_diagonal_field():
    torch.manual_seed(0)
    batch_size, action_dim, num_steps = 5, 4, 7
    diag = torch.tensor([0.3, -0.2, 0.5, 0.1])
    matrix = torch.diag(diag)
    field = _LinearVelocityField(matrix)
    features = torch.zeros(batch_size, 3)
    actions = torch.randn(batch_size, action_dim)

    x0 = _reverse_euler_x0(matrix, actions, num_steps)
    expected = _base_log_prob(x0) - diag.sum()

    result = flow_log_prob(field, features, actions, num_steps=num_steps, method="exact")
    assert torch.allclose(result, expected, atol=1e-5)


def test_hutch_rade_matches_exact_for_diagonal_field():
    torch.manual_seed(1)
    batch_size, action_dim, num_steps = 6, 3, 5
    diag = torch.tensor([1.0, -0.5, 0.25])
    matrix = torch.diag(diag)
    field = _LinearVelocityField(matrix)
    features = torch.zeros(batch_size, 2)
    actions = torch.randn(batch_size, action_dim)

    exact = flow_log_prob(field, features, actions, num_steps=num_steps, method="exact")
    rade = flow_log_prob(
        field, features, actions, num_steps=num_steps, method="hutch-rade", num_probes=4
    )
    assert torch.allclose(rade, exact, atol=1e-6)


def test_hutch_estimators_close_to_exact_for_general_matrix():
    torch.manual_seed(2)
    batch_size, action_dim, num_steps = 4, 3, 4
    matrix = torch.randn(action_dim, action_dim) * 0.3
    field = _LinearVelocityField(matrix)
    features = torch.zeros(batch_size, 2)
    actions = torch.randn(batch_size, action_dim)

    exact = flow_log_prob(field, features, actions, num_steps=num_steps, method="exact")

    gaus = flow_log_prob(
        field, features, actions, num_steps=num_steps, method="hutch-gaus", num_probes=1024
    )
    assert torch.allclose(gaus, exact, atol=0.15)

    rade = flow_log_prob(
        field, features, actions, num_steps=num_steps, method="hutch-rade", num_probes=256
    )
    assert torch.allclose(rade, exact, atol=0.15)


@pytest.fixture
def real_vector_field() -> ActorVectorField:
    torch.manual_seed(3)
    return ActorVectorField(
        features_dim=6,
        action_dim=3,
        hidden_dims=[8],
        use_time_conditioning=True,
    )


@pytest.mark.parametrize("method", ["exact", "hutch-rade", "hutch-gaus"])
def test_real_actor_vector_field_output_contract(real_vector_field, method):
    batch_size = 5
    features = torch.randn(batch_size, 6)
    actions = torch.randn(batch_size, 3)

    with torch.no_grad():
        result = flow_log_prob(
            real_vector_field, features, actions, num_steps=4, method=method, num_probes=4
        )
    assert result.shape == (batch_size,)
    assert torch.isfinite(result).all()
    assert result.requires_grad is False

    with torch.enable_grad():
        result_grad_ctx = flow_log_prob(
            real_vector_field, features, actions, num_steps=4, method=method, num_probes=4
        )
    assert result_grad_ctx.requires_grad is False
    assert torch.isfinite(result_grad_ctx).all()

    if method == "exact":
        assert torch.allclose(result, result_grad_ctx, atol=1e-6)


def test_invalid_method_raises(real_vector_field):
    features = torch.zeros(2, 6)
    actions = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        flow_log_prob(real_vector_field, features, actions, num_steps=4, method="bogus")


def test_invalid_num_probes_raises(real_vector_field):
    features = torch.zeros(2, 6)
    actions = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        flow_log_prob(
            real_vector_field, features, actions, num_steps=4, method="hutch-rade", num_probes=0
        )


def test_invalid_num_steps_raises(real_vector_field):
    features = torch.zeros(2, 6)
    actions = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        flow_log_prob(real_vector_field, features, actions, num_steps=0)


def test_non_time_conditioned_field_raises():
    field = ActorVectorField(
        features_dim=6, action_dim=3, hidden_dims=[8], use_time_conditioning=False
    )
    features = torch.zeros(2, 6)
    actions = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        flow_log_prob(field, features, actions, num_steps=4)
