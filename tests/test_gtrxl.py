from __future__ import annotations

import torch

from rl_garden.networks.gtrxl import GTrXLLatentEncoder


def _make_encoder(**overrides) -> GTrXLLatentEncoder:
    kwargs = dict(input_dim=8, embed_dim=16, head_dim=8, num_heads=2, num_layers=2, memory_len=4)
    kwargs.update(overrides)
    return GTrXLLatentEncoder(**kwargs)


def test_get_initial_state_shapes():
    enc = _make_encoder()
    memory, memory_valid = enc.get_initial_state(batch_size=3, device=torch.device("cpu"))
    assert memory.shape == (3, 3, 4, 16)  # (num_layers+1, B, memory_len, embed_dim)
    assert memory_valid.shape == (1, 3, 4)
    assert torch.all(memory_valid == 0)


def test_step_output_and_state_shapes():
    enc = _make_encoder()
    state = enc.get_initial_state(3, torch.device("cpu"))
    out, (memory, memory_valid) = enc.step(torch.randn(3, 8), state, torch.zeros(3))
    assert out.shape == (3, 16)
    assert memory.shape == (3, 3, 4, 16)
    assert memory_valid.shape == (1, 3, 4)
    # FIFO: the just-added slot (last one) is valid for every env.
    assert torch.all(memory_valid[0, :, -1] == 1)


def test_forward_sequence_output_shape():
    enc = _make_encoder()
    state = enc.get_initial_state(3, torch.device("cpu"))
    out, _ = enc.forward_sequence(torch.randn(5, 3, 8), state, torch.zeros(5, 3))
    assert out.shape == (5, 3, 16)


def test_mask_state_zeroes_selected_envs_only():
    enc = _make_encoder()
    state = enc.get_initial_state(3, torch.device("cpu"))
    _, state = enc.step(torch.randn(3, 8), state, torch.zeros(3))
    keep = torch.tensor([1.0, 0.0, 1.0])
    memory, memory_valid = enc.mask_state(state, keep)
    assert torch.all(memory[:, 1] == 0)
    assert torch.all(memory_valid[:, 1] == 0)
    assert torch.any(memory[:, 0] != 0) or torch.any(memory_valid[:, 0] != 0)


def test_index_state_selects_envs():
    enc = _make_encoder()
    state = enc.get_initial_state(4, torch.device("cpu"))
    _, state = enc.step(torch.randn(4, 8), state, torch.zeros(4))
    memory, memory_valid = enc.index_state(state, torch.tensor([0, 2]))
    assert memory.shape == (3, 2, 4, 16)
    assert memory_valid.shape == (1, 2, 4)


def test_burn_in_not_implemented():
    enc = _make_encoder()
    state = enc.get_initial_state(2, torch.device("cpu"))
    try:
        enc.forward_sequence_with_burn_in(torch.randn(3, 2, 8), state, torch.zeros(3, 2), burn_in_len=1)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_gradient_isolated_at_segment_boundary_but_flows_within_window():
    """Core correctness property (advisor-flagged fix): the seed state at a
    window's start carries no gradient (it comes from a no_grad rollout, exactly
    like RecurrentPPO's initial_hidden), but gradient DOES flow across timesteps
    within the training window -- step()/forward_sequence() must never detach
    internally, or BPTT would be confined to a single step."""
    torch.manual_seed(0)
    enc = _make_encoder()
    B = 2
    with torch.no_grad():
        state = enc.get_initial_state(B, torch.device("cpu"))
        warmup = torch.randn(3, B, 8)
        _, state = enc.forward_sequence(warmup, state, torch.zeros(3, B))
    seed_memory, _ = state
    assert not seed_memory.requires_grad

    latent = torch.randn(4, B, 8, requires_grad=True)
    out, _ = enc.forward_sequence(latent, state, torch.zeros(4, B))
    out.sum().backward()
    assert latent.grad is not None
    assert all((latent.grad[t].abs().sum() > 0).item() for t in range(4))


def test_gradient_flows_across_steps_through_memory():
    """Discriminates the fix from the original bug in a way the previous test
    does not: ``out[t]`` always gets gradient from ``latent[t]`` via its own
    step's direct forward, regardless of whether memory is detached -- that
    alone doesn't prove cross-step BPTT works. The real test is whether a
    LATER step's output has gradient w.r.t. an EARLIER step's input, which can
    only happen through the memory path."""
    torch.manual_seed(3)
    enc = _make_encoder()  # memory_len=4
    B = 2
    state = enc.get_initial_state(B, torch.device("cpu"))

    T = 4
    latent = torch.randn(T, B, 8, requires_grad=True)
    out, _ = enc.forward_sequence(latent, state, torch.zeros(T, B))
    grad = torch.autograd.grad(out[-1].sum(), latent)[0]
    assert grad[0].abs().sum() > 0, "no gradient from the last step back to the first -- memory is detached"


def test_episode_reset_blocks_information_flow():
    """New logic neither reference implementation has: post-reset output must
    be completely independent of pre-reset latent history."""
    torch.manual_seed(1)
    enc = _make_encoder()
    enc.eval()
    state = enc.get_initial_state(1, torch.device("cpu"))
    T = 6
    latent_a = torch.randn(T, 1, 8)
    latent_b = latent_a.clone()
    latent_b[:3] = torch.randn(3, 1, 8)
    episode_starts = torch.zeros(T, 1)
    episode_starts[3, 0] = 1.0

    with torch.no_grad():
        out_a, _ = enc.forward_sequence(latent_a, state, episode_starts)
        out_b, _ = enc.forward_sequence(latent_b, state, episode_starts)

    assert torch.allclose(out_a[3:], out_b[3:], atol=1e-6)
    assert not torch.allclose(out_a[:3], out_b[:3], atol=1e-6)


def test_encoder_is_a_pure_function_of_explicit_state():
    """Relative (not absolute) positional encoding means step() must be a pure
    function of (latent, state, episode_starts) -- identical (memory,
    memory_valid) content must give identical output regardless of how many
    prior step() calls produced it (no hidden internal time counter)."""
    torch.manual_seed(2)
    enc = _make_encoder()
    enc.eval()
    memory_content = torch.randn(3, 1, 4, 16)
    valid = torch.ones(1, 1, 4)

    state_direct = (memory_content.clone(), valid.clone())
    state_via_many_calls = enc.get_initial_state(1, torch.device("cpu"))
    for _ in range(37):
        _, state_via_many_calls = enc.step(torch.randn(1, 8), state_via_many_calls, torch.zeros(1))
    state_via_many_calls = (memory_content.clone(), valid.clone())

    final_latent = torch.randn(1, 8)
    with torch.no_grad():
        out_direct, _ = enc.step(final_latent, state_direct, torch.zeros(1))
        out_via_many, _ = enc.step(final_latent, state_via_many_calls, torch.zeros(1))
    assert torch.allclose(out_direct, out_via_many, atol=1e-6)


def test_rejects_odd_embed_dim():
    try:
        _make_encoder(embed_dim=15)
        assert False, "expected ValueError"
    except ValueError:
        pass
