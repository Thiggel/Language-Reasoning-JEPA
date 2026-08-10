import pytest
import torch

from scripts.build_hierarchical_sentence_geometry import (
    _align_examples_to_features,
)
from textjepa.analysis.hierarchical_language import paired_semantic_geometry


def test_paired_semantic_geometry_separates_paraphrases_from_contrasts():
    states = torch.tensor([
        [0.0, 0.0], [0.1, 0.0], [2.0, 0.0],
        [0.0, 3.0], [0.0, 3.1], [2.0, 3.0],
    ])
    metrics = paired_semantic_geometry(
        states,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    assert metrics["euclidean_triplet_accuracy"] == 1.0
    assert metrics["mean_paraphrase_euclidean"] < metrics["mean_contrast_euclidean"]


def test_paired_semantic_geometry_rejects_missing_control():
    with pytest.raises(ValueError, match="one anchor"):
        paired_semantic_geometry(
            torch.randn(3, 4),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 2]),
        )


def test_geometry_joins_feature_subset_to_larger_source_manifest_by_id():
    examples = [
        {"problem_id": "unused", "value": 0},
        {"problem_id": "b", "value": 2},
        {"problem_id": "a", "value": 1},
    ]
    aligned = _align_examples_to_features(examples, ["a", "b"])
    assert [row["value"] for row in aligned] == [1, 2]


@pytest.mark.parametrize(
    "examples, feature_ids, message",
    [
        ([{"problem_id": "a"}, {"problem_id": "a"}], ["a"], "duplicate"),
        ([{"problem_id": "a"}], ["a", "a"], "unique"),
        ([{"problem_id": "a"}], ["a", "b"], "missing"),
    ],
)
def test_geometry_problem_id_join_rejects_ambiguous_or_missing_rows(
    examples, feature_ids, message
):
    with pytest.raises(ValueError, match=message):
        _align_examples_to_features(examples, feature_ids)
