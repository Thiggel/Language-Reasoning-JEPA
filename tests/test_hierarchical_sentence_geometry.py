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


@pytest.mark.parametrize("normalized", [True, False])
def test_both_metrics_expose_the_whiten_interface_the_geometry_export_uses(
    normalized,
):
    """A euclidean run previously crashed in the geometry export with
    ``'SquaredEuclideanMetric' object has no attribute 'whiten'``."""
    from textjepa.objectives.hierarchical_language import (
        EMAShrunkMahalanobis,
        SquaredEuclideanMetric,
    )

    torch.manual_seed(0)
    z = torch.randn(5, 6)
    for metric in (
        SquaredEuclideanMetric(6, normalized=normalized),
        EMAShrunkMahalanobis(6, normalized=normalized),
    ):
        whitened = metric.whiten(z - metric.mean)
        assert whitened.shape == z.shape
        assert torch.isfinite(whitened).all()


def test_euclidean_whitening_preserves_pairwise_distance_ratios():
    from textjepa.objectives.hierarchical_language import (
        SquaredEuclideanMetric,
    )

    torch.manual_seed(0)
    metric = SquaredEuclideanMetric(6)
    z = torch.randn(5, 6)
    whitened = metric.whiten(z - metric.mean)
    before = torch.cdist(z, z)
    after = torch.cdist(whitened, whitened)
    scale = after[before > 0] / before[before > 0]
    assert torch.allclose(scale, scale[0].expand_as(scale), atol=1e-5)
