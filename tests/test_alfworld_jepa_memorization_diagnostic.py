import torch

from scripts.diagnose_alfworld_jepa_memorization import (
    _pairwise_accuracy,
    _selected_counterfactual_actions,
    _summary,
)
from textjepa.data.observed_action import Counterfactual


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


def test_selected_counterfactual_actions_matches_deterministic_prefix():
    alternatives = tuple(
        Counterfactual(action=f"action-{index}", outcome=f"outcome-{index}")
        for index in range(6)
    )
    transition = type("Transition", (), {"counterfactuals": alternatives})()
    episode = type("Episode", (), {"episode_id": "episode-a"})()
    first = _selected_counterfactual_actions(episode, transition, seed=7, k=2)
    second = _selected_counterfactual_actions(episode, transition, seed=7, k=2)
    wider = _selected_counterfactual_actions(episode, transition, seed=7, k=5)
    assert first == second
    assert len(first) == 2
    assert first < wider
