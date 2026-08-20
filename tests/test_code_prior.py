"""Contract tests for the state-conditioned discrete action prior."""

import torch

from textjepa.planning.code_prior import CodePrior, CodePriorConfig


def _fitted(n_codes=8, dim=6, n=256, residual=True):
    torch.manual_seed(0)
    p = CodePrior(CodePriorConfig(dim=dim, n_codes=n_codes, hidden=32,
                                  n_layers=2, residual=residual))
    a = torch.randn(n, dim)
    s = torch.randn(n, dim)
    p.fit_scaling(a, s)
    p.init_codebook(a, iters=5, generator=torch.Generator().manual_seed(0))
    return p, a, s


def test_whitening_is_applied():
    p, a, s = _fitted()
    w = p._wa(a)
    assert torch.allclose(w.mean(0), torch.zeros(6), atol=1e-5)
    assert torch.allclose(w.std(0), torch.ones(6), atol=1e-3)


def test_quantize_returns_valid_indices_and_is_nearest():
    p, a, _ = _fitted()
    idx = p.quantize(a)
    assert int(idx.min()) >= 0 and int(idx.max()) < 8
    d = torch.cdist(p._wa(a), p.codebook)
    assert torch.equal(idx, d.argmin(1))


def test_codebook_uses_more_than_one_code():
    """A collapsed codebook would make the whole diversity argument vacuous."""
    p, a, _ = _fitted()
    assert len(set(p.quantize(a).tolist())) > 1


def test_propose_returns_distinct_codes_by_construction():
    p, _, s = _fitted()
    idx, vecs = p.propose(s[:5], 4)
    assert idx.shape == (5, 4) and vecs.shape == (5, 4, 6)
    for row in idx:
        assert len(set(row.tolist())) == 4


def test_propose_sampling_is_also_without_replacement():
    p, _, s = _fitted()
    g = torch.Generator().manual_seed(3)
    idx, _ = p.propose(s[:5], 4, sample=True, generator=g)
    for row in idx:
        assert len(set(row.tolist())) == 4


def test_dequantize_without_residual_is_the_bare_centroid():
    p, a, s = _fitted(residual=False)
    idx = p.quantize(a)
    want = p.codebook[idx] * p.a_std + p.a_mean
    assert torch.allclose(p.dequantize(s, idx), want, atol=1e-6)


def test_residual_head_starts_as_the_identity_dequantizer():
    """Zero-init means training starts exactly at the plain-codebook model."""
    p, a, s = _fitted(residual=True)
    idx = p.quantize(a)
    want = p.codebook[idx] * p.a_std + p.a_mean
    assert torch.allclose(p.dequantize(s, idx), want, atol=1e-6)


def test_loss_decreases_with_training():
    p, a, s = _fitted()
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    first = float(p.loss(s, a)["loss"])
    for _ in range(50):
        loss = p.loss(s, a)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(p.loss(s, a)["loss"]) < first


def test_ema_update_moves_codes_toward_the_data():
    p, a, _ = _fitted()
    before = p.codebook.clone()
    for _ in range(5):
        p.ema_update(a)
    assert not torch.allclose(before, p.codebook)


def test_save_load_roundtrip(tmp_path):
    p, a, s = _fitted()
    path = tmp_path / "prior.pt"
    p.save(str(path))
    q = CodePrior.load(str(path))
    assert torch.equal(p.quantize(a), q.quantize(a))
    assert torch.allclose(p.logits(s), q.logits(s), atol=1e-6)
