import pytest
import torch

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
