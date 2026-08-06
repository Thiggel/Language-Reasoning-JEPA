import torch
import pytest

from scripts.audit_token_planner_interface import (
    one_indexed_rank,
    pairwise_accuracy,
    summarize_ranks,
)


def test_rank_and_summary_use_cost_ordering():
    cost = torch.tensor([0.4, 0.1, 0.3])
    assert one_indexed_rank(cost, 2) == 2
    summary = summarize_ranks([1, 2, 21])
    assert summary["top1"] == pytest.approx(1 / 3)
    assert summary["top5"] == pytest.approx(2 / 3)
    assert summary["top20"] == pytest.approx(2 / 3)


def test_pairwise_accuracy_detects_correct_and_reversed_order():
    target = torch.tensor([0.1, 0.4, 0.9])
    assert pairwise_accuracy(target, target) == 1.0
    assert pairwise_accuracy(target.flip(0), target) == 0.0
