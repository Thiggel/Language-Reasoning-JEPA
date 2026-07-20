import torch

from scripts.audit_token_oracle_continuation_teacher import rank_of_first, summarize


def test_rank_of_first_uses_lower_cost_as_better():
    assert rank_of_first(torch.tensor([2.0, 1.0, 3.0])) == 2


def test_teacher_summary_reports_sample_count():
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["n"] == 3
