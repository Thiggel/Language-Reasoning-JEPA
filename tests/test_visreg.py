import math

import torch
import torch.nn.functional as F

from textjepa.objectives.visreg import VISReg


def _official_reference(states, projections):
    if states.ndim == 2:
        states = states.unsqueeze(0)
    _, batch, dimension = states.shape
    mean = states.mean(1, keepdim=True)
    center = mean.square().mean()
    centered = states - mean
    std = centered.norm(dim=1).div(math.sqrt(batch)).add(1e-6)
    scale = (std - 1).square().mean()
    normalized = centered / std.detach().unsqueeze(1)
    directions = F.normalize(torch.randn(
        dimension, projections, device=states.device, dtype=states.dtype,
    ), dim=0)
    projected = (normalized @ directions).sort(dim=1).values
    q = torch.linspace(1, batch, batch, dtype=torch.float32) / (batch + 1)
    target = torch.erfinv(2 * q - 1).mul(math.sqrt(2)).to(states.dtype)
    shape = (projected - target.view(1, batch, 1)).square().mean()
    return scale + shape + center


def test_visreg_matches_official_implementation_equations():
    states = torch.randn(2, 13, 9)
    torch.manual_seed(71)
    expected = _official_reference(states, 32)
    torch.manual_seed(71)
    actual = VISReg(32)(states)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_visreg_has_finite_nonzero_gradient_near_collapse():
    torch.manual_seed(9)
    states = (1e-5 * torch.randn(1, 32, 12)).requires_grad_()
    loss = VISReg(24)(states)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(states.grad).all()
    assert states.grad.abs().sum() > 0


def test_visreg_accepts_single_view_matrix_and_rejects_tiny_batches():
    assert torch.isfinite(VISReg(8)(torch.randn(7, 5)))
    try:
        VISReg(8)(torch.randn(1, 5))
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("one-sample VISReg must be rejected")
