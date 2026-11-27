from types import SimpleNamespace

import pytest
import torch

from rl_garden.algorithms.acrlpd import ACRLPDCore


@pytest.mark.parametrize("complete", [[True, False, True], [True, True, True], [False, False, False]])
def test_acrlpd_td_loss_means_over_critics_and_whole_batch(complete):
    target = torch.tensor([[1.0], [2.0], [-1.0]])
    predictions = torch.tensor([
        [[3.0], [100.0], [2.0]],
        [[-1.0], [-50.0], [0.0]],
    ], requires_grad=True)
    valid = torch.tensor([[True, end] for end in complete])
    data = SimpleNamespace(valid=valid)
    host = SimpleNamespace(_target_q=lambda batch: target)
    loss, info = ACRLPDCore._td_loss(host, data, predictions)
    expected = sum(
        (float(predictions[c, b, 0].detach()) - float(target[b, 0])) ** 2
        for c in range(2) for b in range(3) if complete[b]
    ) / 6
    assert loss.item() == pytest.approx(expected)
    assert set(info) == {"td_loss", "target_q"}
    assert info["td_loss"].item() == loss.item()
    assert not info["td_loss"].requires_grad
    assert info["target_q"].item() == pytest.approx(target.mean().item())
    loss.backward()
    expected_gradient = (predictions.detach() - target) * valid[:, -1, None] / 3
    torch.testing.assert_close(predictions.grad, expected_gradient)
