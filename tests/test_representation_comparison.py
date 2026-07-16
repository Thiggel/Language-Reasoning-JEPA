import torch

from scripts.compare_token_encoder_representations import linear_cka


def test_linear_cka_identity_and_rotation():
    torch.manual_seed(0)
    x = torch.randn(64, 12)
    q, _ = torch.linalg.qr(torch.randn(12, 12))
    assert abs(linear_cka(x, x) - 1.0) < 1e-5
    assert abs(linear_cka(x, x @ q) - 1.0) < 1e-5
