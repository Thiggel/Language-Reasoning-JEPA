import torch

from scripts.diagnose_alfworld_jepa_memorization import (
    _pairwise_accuracy,
    _summary,
)


def test_pairwise_accuracy_ignores_label_ties():
    energy = torch.tensor([0.1, 0.2, 0.21])
    label = torch.tensor([0.0, 1.0, 1.01])
    assert _pairwise_accuracy(energy, label, gap=0.02) == (2, 2)


def test_pairwise_accuracy_detects_reversed_order():
    energy = torch.tensor([2.0, 1.0, 0.0])
    label = torch.tensor([0.0, 1.0, 2.0])
    assert _pairwise_accuracy(energy, label) == (0, 3)


def test_summary_is_explicit_for_empty_input():
    assert _summary([]) == {"mean": 0.0, "median": 0.0, "count": 0}
