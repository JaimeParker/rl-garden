from __future__ import annotations

import pytest
import torch

from rl_garden.networks.recurrent import RecurrentLatentEncoder


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_get_initial_state_shapes(rnn_type):
    enc = RecurrentLatentEncoder(input_dim=4, hidden_size=8, rnn_type=rnn_type, num_layers=2)
    state = enc.get_initial_state(batch_size=3, device=torch.device("cpu"))
    if rnn_type == "lstm":
        h, c = state
        assert h.shape == (2, 3, 8)
        assert c.shape == (2, 3, 8)
    else:
        assert state.shape == (2, 3, 8)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_step_output_shape(rnn_type):
    enc = RecurrentLatentEncoder(input_dim=4, hidden_size=6, rnn_type=rnn_type)
    state = enc.get_initial_state(3, torch.device("cpu"))
    out, new_state = enc.step(torch.randn(3, 4), state)
    assert out.shape == (3, 6)
    if rnn_type == "lstm":
        assert new_state[0].shape == (1, 3, 6)
        assert new_state[1].shape == (1, 3, 6)
    else:
        assert new_state.shape == (1, 3, 6)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_forward_sequence_output_shape(rnn_type):
    enc = RecurrentLatentEncoder(input_dim=4, hidden_size=6, rnn_type=rnn_type)
    state = enc.get_initial_state(3, torch.device("cpu"))
    out, _ = enc.forward_sequence(torch.randn(5, 3, 4), state)
    assert out.shape == (5, 3, 6)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_step_by_step_matches_forward_sequence_with_reset(rnn_type):
    """Core correctness property: forward_sequence() must be equivalent to
    manually looping step() with the same per-timestep episode_starts, including
    across a real mid-sequence reset for one env in the batch."""
    torch.manual_seed(0)
    enc = RecurrentLatentEncoder(input_dim=4, hidden_size=5, rnn_type=rnn_type, num_layers=2)
    state = enc.get_initial_state(2, torch.device("cpu"))

    latent = torch.randn(5, 2, 4)
    episode_starts = torch.zeros(5, 2)
    episode_starts[2, 0] = 1.0  # env 0 resets mid-sequence; env 1 never resets

    out_seq, final_state_seq = enc.forward_sequence(latent, state, episode_starts)

    manual_state = state
    manual_outs = []
    for t in range(5):
        out_t, manual_state = enc.step(latent[t], manual_state, episode_starts[t])
        manual_outs.append(out_t)
    manual_out = torch.stack(manual_outs, dim=0)

    assert torch.allclose(out_seq, manual_out)
    if rnn_type == "lstm":
        assert torch.allclose(final_state_seq[0], manual_state[0])
        assert torch.allclose(final_state_seq[1], manual_state[1])
    else:
        assert torch.allclose(final_state_seq, manual_state)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_gradient_flows_to_all_timesteps(rnn_type):
    enc = RecurrentLatentEncoder(input_dim=3, hidden_size=4, rnn_type=rnn_type)
    state = enc.get_initial_state(2, torch.device("cpu"))
    latent = torch.randn(6, 2, 3, requires_grad=True)

    out, _ = enc.forward_sequence(latent, state)
    out.sum().backward()

    assert latent.grad is not None
    for t in range(6):
        assert bool((latent.grad[t].abs().sum() > 0).item()), f"timestep {t} has zero grad"


def test_mask_state_zeros_reset_envs_only():
    h = torch.arange(1, 1 + 1 * 3 * 2, dtype=torch.float32).reshape(1, 3, 2)
    keep_mask = torch.tensor([1.0, 0.0, 1.0])
    masked = RecurrentLatentEncoder.mask_state(h, keep_mask)
    assert torch.all(masked[:, 0] == h[:, 0])
    assert torch.all(masked[:, 1] == 0)
    assert torch.all(masked[:, 2] == h[:, 2])


def test_mask_state_lstm_tuple():
    h = torch.ones(1, 2, 3)
    c = torch.ones(1, 2, 3) * 2
    keep_mask = torch.tensor([0.0, 1.0])
    masked_h, masked_c = RecurrentLatentEncoder.mask_state((h, c), keep_mask)
    assert torch.all(masked_h[:, 0] == 0)
    assert torch.all(masked_c[:, 0] == 0)
    assert torch.all(masked_h[:, 1] == 1)
    assert torch.all(masked_c[:, 1] == 2)


def test_index_state_selects_correct_batch_rows():
    h = torch.arange(1, 1 + 1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    indices = torch.tensor([0, 2])
    indexed = RecurrentLatentEncoder.index_state(h, indices)
    assert indexed.shape == (1, 2, 2)
    assert torch.equal(indexed[:, 0], h[:, 0])
    assert torch.equal(indexed[:, 1], h[:, 2])


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_forward_sequence_with_burn_in_tail_matches_manual_slice(rnn_type):
    """Functional equivalence: with gradients enabled throughout, the tail output
    of forward_sequence_with_burn_in must equal slicing a full forward_sequence()
    call at [burn_in_len:], given the same initial state."""
    torch.manual_seed(0)
    enc = RecurrentLatentEncoder(input_dim=4, hidden_size=5, rnn_type=rnn_type, num_layers=2)
    state = enc.get_initial_state(3, torch.device("cpu"))

    burn_in_len = 2
    latent = torch.randn(7, 3, 4)
    episode_starts = torch.zeros(7, 3)
    episode_starts[4, 1] = 1.0  # mid-sequence reset in the tail portion

    full_out, full_state = enc.forward_sequence(latent, state, episode_starts)
    tail_out, tail_state = enc.forward_sequence_with_burn_in(
        latent, state, episode_starts, burn_in_len
    )

    assert torch.allclose(tail_out, full_out[burn_in_len:])
    if rnn_type == "lstm":
        assert torch.allclose(tail_state[0], full_state[0])
        assert torch.allclose(tail_state[1], full_state[1])
    else:
        assert torch.allclose(tail_state, full_state)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_forward_sequence_with_burn_in_isolates_gradient(rnn_type):
    """Burn-in must carry no gradient; the tail must remain fully differentiable."""
    enc = RecurrentLatentEncoder(input_dim=3, hidden_size=4, rnn_type=rnn_type)
    state = enc.get_initial_state(2, torch.device("cpu"))

    burn_in_len = 3
    latent = torch.randn(6, 2, 3, requires_grad=True)
    episode_starts = torch.zeros(6, 2)

    tail_out, tail_state = enc.forward_sequence_with_burn_in(
        latent, state, episode_starts, burn_in_len
    )

    assert tail_out.requires_grad
    if rnn_type == "lstm":
        assert tail_state[0].requires_grad
        assert tail_state[1].requires_grad
    else:
        assert tail_state.requires_grad

    tail_out.sum().backward()
    assert latent.grad is not None
    # No gradient should reach the burn-in-only prefix positions.
    for t in range(burn_in_len):
        assert bool((latent.grad[t].abs().sum() == 0).item()), f"burn-in timestep {t} has nonzero grad"
    # Every tail position must receive gradient.
    for t in range(burn_in_len, 6):
        assert bool((latent.grad[t].abs().sum() > 0).item()), f"tail timestep {t} has zero grad"


def test_forward_dispatches_on_ndim():
    enc = RecurrentLatentEncoder(input_dim=3, hidden_size=4)
    state = enc.get_initial_state(2, torch.device("cpu"))
    out_step, _ = enc.forward(torch.randn(2, 3), state)
    assert out_step.shape == (2, 4)
    out_seq, _ = enc.forward(torch.randn(5, 2, 3), state)
    assert out_seq.shape == (5, 2, 4)
    with pytest.raises(ValueError):
        enc.forward(torch.randn(2, 3, 4, 5), state)
