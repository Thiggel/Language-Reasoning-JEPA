import torch

from scripts.audit_token_closed_loop_decomposition import standardize, summarize


def test_standardize_is_finite_for_constant_scores():
    result = standardize(torch.ones(8))
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))


def test_summary_drops_nonfinite_values():
    result = summarize([1.0, float("nan"), 3.0])
    assert result["mean"] == 2.0
    assert result["n"] == 2
