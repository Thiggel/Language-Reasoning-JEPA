import json
import sys

import torch

from scripts import train_hierarchical_language_jepa as train
from textjepa.models.hierarchical_language_jepa import (
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.objectives.hierarchical_language import (
    SketchedIsotropicGaussianRegularizer,
    SquaredEuclideanMetric,
)
from textjepa.training.hierarchical_language import (
    DenseLossWeights,
    HierarchicalLanguageLearner,
    ResearchStage,
)


def _model():
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=8,
        vocab_size=17,
        pad_id=0,
        d_token=6,
        d_sentence=4,
        d_action=3,
        predictor_width=8,
        token_layers=1,
        sentence_layers=1,
        n_heads=1,
        token_context=4,
        sentence_context=3,
        max_span=4,
    ))


def _dense_batch():
    torch.manual_seed(7)
    hidden = torch.randn(3, 8, 8)
    token_ids = torch.randint(1, 17, (3, 8))
    boundaries = torch.tensor([
        [0, 3, 7],
        [0, 4, 7],
        [0, 3, 7],
    ])
    return hidden, token_ids, boundaries


def test_normalized_euclidean_averages_coordinates():
    metric = SquaredEuclideanMetric(4, normalized=True)
    assert metric(torch.ones(2, 4), torch.zeros(2, 4)).tolist() == [1.0, 1.0]
    assert torch.equal(metric.matrix(), torch.eye(4) / 4)


def test_sigreg_penalizes_collapsed_samples_more_than_gaussian_samples():
    torch.manual_seed(11)
    mask = torch.ones(1024, dtype=torch.bool)
    collapsed = torch.zeros(1024, 8)
    gaussian = torch.randn(1024, 8)
    first = SketchedIsotropicGaussianRegularizer(8, num_slices=128)
    second = SketchedIsotropicGaussianRegularizer(8, num_slices=128)
    assert first(collapsed, mask) > second(gaussian, mask)


def test_joint_dense_training_updates_both_nested_levels():
    model = _model()
    learner = HierarchicalLanguageLearner(
        model,
        ResearchStage.SENTENCE_JEPA,
        weights=DenseLossWeights(
            token_dynamics=25,
            sentence_dynamics=25,
            variance=25,
            covariance=1,
        ),
        dynamics_geometry="euclidean",
        normalize_dynamics=True,
        joint_token_sentence=True,
    )
    total, losses = learner(*_dense_batch())
    assert {
        "token_dynamics", "token_variance", "token_covariance",
        "sentence_dynamics", "sentence_variance", "sentence_covariance",
    } <= losses.keys()
    total.backward()
    assert any(parameter.grad is not None for parameter in model.e0.parameters())
    assert any(
        parameter.grad is not None for parameter in model.e0_to_1.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.p0.parameters())
    assert any(parameter.grad is not None for parameter in model.p1.parameters())


def test_counterfactual_joint_step_regularizes_both_levels():
    torch.manual_seed(13)
    model = _model()
    learner = HierarchicalLanguageLearner(
        model,
        ResearchStage.SENTENCE_JEPA,
        dynamics_geometry="euclidean",
        normalize_dynamics=True,
        joint_token_sentence=True,
    )
    total, losses = learner.counterfactual_loss(
        torch.randn(4, 8),
        torch.randn(4, 3, 8),
        torch.randint(1, 17, (4, 3)),
        torch.full((4,), 3),
        sentence_eligible=torch.ones(4, dtype=torch.bool),
    )
    assert {
        "counterfactual_token_variance",
        "counterfactual_token_covariance",
        "counterfactual_sentence_variance",
        "counterfactual_sentence_covariance",
    } <= losses.keys()
    total.backward()
    assert any(parameter.grad is not None for parameter in model.e0.parameters())
    assert any(
        parameter.grad is not None for parameter in model.e0_to_1.parameters()
    )


def test_joint_stage_trainability_includes_both_predictive_levels():
    model = _model()
    active = train._configure_stage_trainability(
        model,
        ResearchStage.SENTENCE_JEPA,
        joint_token_sentence=True,
    )
    assert active == [
        "e0", "p0", "token_action", "e0_to_1", "a1", "p1"
    ]


def test_joint_training_entrypoint_starts_from_scratch(tmp_path, monkeypatch):
    features = tmp_path / "features.pt"
    output = tmp_path / "output"
    torch.save({
        "hidden_states": torch.randn(4, 7, 8),
        "token_ids": torch.tensor([
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 2, 4, 6, 5, 7],
            [2, 1, 3, 5, 4, 6, 7],
            [2, 3, 1, 6, 4, 5, 7],
        ]),
        "boundaries": torch.tensor([[0, 3, 6]] * 4),
        "dataset_fingerprint": "joint-unit-test",
    }, features)
    config = tmp_path / "joint.yaml"
    config.write_text("""
research:
  stage: SENTENCE_JEPA
  joint_token_sentence: true
model:
  config:
    d_backbone: 8
    vocab_size: 17
    pad_id: 0
    d_token: 6
    d_sentence: 4
    d_action: 3
    d_task: 4
    predictor_width: 8
    token_layers: 1
    sentence_layers: 1
    n_heads: 1
    token_context: 4
    sentence_context: 3
    cache_dropout: 0.0
    max_span: 4
loss:
  dynamics_geometry: euclidean
  normalize_dynamics: true
  anti_collapse: vicreg
  token_dynamics: 25.0
  sentence_dynamics: 25.0
  variance: 25.0
  covariance: 1.0
  token_rollout: 0.0
""")
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features),
        "--allow-unpinned-features",
        "--experiment-config", str(config),
        "--output", str(output),
        "--epochs", "1",
        "--batch-size", "4",
        "--max-optimizer-steps", "1",
        "--device", "cpu",
    ])
    train.main()
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["initialized_from"] is None
    assert checkpoint["trainable_modules"] == [
        "e0", "p0", "token_action", "e0_to_1", "a1", "p1"
    ]
    history = json.loads((output / "metrics.json").read_text())["history"]
    assert "token_dynamics" in history[-1]
    assert "sentence_dynamics" in history[-1]
