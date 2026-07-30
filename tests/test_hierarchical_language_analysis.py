import pytest
import torch

from textjepa.analysis.hierarchical_language import (
    path_geometry,
    proposal_coverage,
    representation_statistics,
    requested_achieved_displacement,
    value_ranking_diagnostics,
)
from textjepa.analysis.compute import ComputeLedger


def test_representation_statistics_detect_collapse_and_rank():
    collapsed = representation_statistics(torch.zeros(8, 4))
    assert collapsed["entropic_effective_rank"] == 1.0
    assert collapsed["participation_ratio"] == 0.0
    varied = representation_statistics(torch.eye(4).repeat(2, 1))
    assert varied["entropic_effective_rank"] > 2


def test_proposal_coverage_is_monotone_in_sample_count():
    valid = torch.tensor([
        [False, True, False, False],
        [False, False, False, True],
    ])
    result = proposal_coverage(valid, (1, 2, 4))
    assert result == {
        "greedy": 0.0, "oracle@1": 0.0,
        "oracle@2": 0.5, "oracle@4": 1.0,
    }


def test_value_ranking_reports_top_one_regret():
    target = torch.tensor([[0.0, 1.0, 3.0]])
    good = value_ranking_diagnostics(
        target, target, torch.ones_like(target, dtype=torch.bool)
    )
    bad = value_ranking_diagnostics(
        target, -target, torch.ones_like(target, dtype=torch.bool)
    )
    assert good["pairwise_accuracy"] == 1
    assert good["top_one_regret"] == 0
    assert bad["top_one_regret"] == 3
    assert good["ndcg"] > bad["ndcg"]


def test_waypoint_displacement_sign_and_path_geometry():
    requested = torch.tensor([[2.0, 1.0]])
    achieved = torch.tensor([[1.5, 2.0]])
    assert requested_achieved_displacement(
        requested, achieved
    ).tolist() == [[0.5, -1.0]]
    straight = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    geometry = path_geometry(straight)
    assert geometry["path_to_chord"].item() == pytest.approx(1)
    assert geometry["turning_cosine"].item() == pytest.approx(1)


def test_compute_ledger_reports_fractional_flops_and_latency():
    ledger = ComputeLedger()
    ledger.add("observed", estimated_flops=75, wall_seconds=3, items=10)
    ledger.add(
        "counterfactual_generation",
        estimated_flops=25, wall_seconds=1, items=2,
    )
    summary = ledger.summary()
    assert summary["components"]["observed"]["flop_fraction"] == 0.75
    assert summary["components"]["counterfactual_generation"][
        "wall_fraction"
    ] == 0.25
