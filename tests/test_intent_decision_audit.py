import numpy as np
import pytest

from textjepa.analysis.intent_decisions import (
    aggregate_rank_metrics,
    geometry_features,
    rank_metrics,
    transition_metrics,
)


def test_rank_metrics_support_multiple_necessary_actions_and_ties():
    result = rank_metrics([0.2, 0.2, 0.5], [False, True, True])
    assert result["top1"] == 1.0
    assert result["reciprocal_rank"] == pytest.approx(0.5)
    assert result["pairwise_accuracy"] == pytest.approx(0.25)
    assert result["margin"] == pytest.approx(0.0)


def test_rank_metrics_rejects_no_positive_candidate():
    with pytest.raises(ValueError, match="positive"):
        rank_metrics([0.0, 1.0], [False, False])


def test_rank_metrics_exact_regret_and_tied_spearman():
    result = rank_metrics(
        [0.4, 0.1, 0.3, 0.2],
        [False, True, False, True],
        [2.0, 0.0, 2.0, 1.0],
    )
    assert result["mean_regret"] == 0.0
    assert result["optimal_top1"] == 1.0
    assert result["optimal_margin"] == pytest.approx(0.1)
    assert result["exact_ordering_accuracy"] == 1.0
    assert result["spearman"] > 0.94


def test_rank_metrics_reports_regret_for_wrong_choice():
    result = rank_metrics([0.0, 1.0], [False, True], [3.0, 1.0])
    assert result["mean_regret"] == 2.0
    assert result["optimal_top1"] == 0.0
    assert result["exact_ordering_accuracy"] == 0.0
    assert result["spearman"] == pytest.approx(-1.0)


def test_aggregate_excludes_noncompetitive_margins():
    rows = [
        rank_metrics([0.0], [True]),
        rank_metrics([0.0, 1.0], [True, False]),
    ]
    result = aggregate_rank_metrics(rows)
    assert result["decisions"] == 2
    assert result["competitive_decisions"] == 1
    assert result["pairwise_accuracy"] == 1.0


def test_transition_retrieval_is_group_local():
    target = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    predicted = target.copy()
    groups = np.array([0, 0, 1, 1])
    result = transition_metrics(predicted, target, groups)
    assert result["within_state_retrieval"] == 1.0
    assert result["layernorm_l1_mean"] == pytest.approx(0.0)


def test_geometry_features_include_goal_relative_interactions():
    state = np.ones((3, 2), dtype=np.float32)
    goal = np.full((3, 2), 2.0, dtype=np.float32)
    features = geometry_features(state, goal)
    assert features.shape == (3, 10)
    np.testing.assert_allclose(features[:, 4:6], -1.0)
