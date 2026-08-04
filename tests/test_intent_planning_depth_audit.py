import numpy as np

from textjepa.analysis.intent_decisions import summarize_depth_decision


def test_depth_audit_exposes_terminal_score_root_reversal():
    sequences = [[0, 2], [0, 3], [1, 4], [1, 5]]
    # Root 0 is truly optimal and wins under its one-step score.  A spurious
    # extreme terminal score under root 1 reverses the deployed decision.
    costs = np.asarray([0.2, 0.3, -2.0, 0.6])
    endpoint = np.asarray([0, 1, 3, 2])
    metrics, rows = summarize_depth_decision(
        sequences, costs, endpoint,
        root_exact_cost={0: 1, 1: 2},
        root_is_necessary={0: True, 1: False},
        one_step_cost={0: 0.1, 1: 0.5},
    )
    assert metrics["one_step_root_score"]["top1"] == 1.0
    assert metrics["deployed_terminal_score"]["top1"] == 0.0
    assert metrics["selected_sequence_endpoint_regret"] == 3.0
    assert rows[1]["selection_optimism"] > rows[0]["selection_optimism"]


def test_depth_audit_oracle_endpoint_validates_candidate_tree():
    sequences = [[0, 2], [0, 3], [1, 4], [1, 5]]
    metrics, _ = summarize_depth_decision(
        sequences, np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([0, 1, 2, 3]),
        root_exact_cost={0: 1, 1: 2},
        root_is_necessary={0: True, 1: False},
        one_step_cost={0: 0.1, 1: 0.2},
    )
    assert metrics["oracle_endpoint_score"]["top1"] == 1.0
    assert metrics["oracle_endpoint_score"]["mean_regret"] == 0.0


def test_depth_audit_rejects_misaligned_inputs():
    try:
        summarize_depth_decision(
            [[0], [1]], np.asarray([0.0]), np.asarray([0.0, 1.0]),
            {0: 1, 1: 2}, {0: True, 1: False}, {0: 0.0, 1: 1.0},
        )
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("misaligned inputs must fail")
