import numpy as np

from textjepa.analysis.intent_depth_localization import (
    prune_beam,
    search_symbolic_tree,
)
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.planning.search import _feasible


def test_root_balanced_pruning_preserves_every_root_with_sufficient_width():
    sequences = [[0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [2, 7]]
    scores = np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    global_beam = prune_beam(sequences, scores, width=3, strategy="global")
    balanced = prune_beam(
        sequences, scores, width=3, strategy="root_balanced"
    )
    assert {sequence[0] for sequence in global_beam} == {0, 1}
    assert {sequence[0] for sequence in balanced} == {0, 1, 2}


def test_root_balanced_search_reports_optimal_root_survival():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=121).problem(0)
    roots = _feasible(problem, frozenset())
    useful = set(roots) & set(problem.query_ancestors)
    assert useful and len(roots) > 1

    def distractor_favoring_score(sequences):
        return np.asarray([
            10.0 if sequence[0] in useful else 0.0
            for sequence in sequences
        ])

    global_result = search_symbolic_tree(
        problem, frozenset(), depth=2, width=1, strategy="global",
        score=distractor_favoring_score,
    )
    balanced_result = search_symbolic_tree(
        problem, frozenset(), depth=2, width=1,
        strategy="root_balanced", score=distractor_favoring_score,
    )
    assert not global_result.levels[-1].optimal_root_survives
    assert balanced_result.levels[-1].optimal_root_survives


def test_pruning_rejects_invalid_inputs_and_strategy():
    try:
        prune_beam([[0]], np.asarray([]), 1, "global")
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("misaligned scores must fail")
    try:
        prune_beam([[0]], np.asarray([0.0]), 1, "unknown")
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown strategy must fail")
