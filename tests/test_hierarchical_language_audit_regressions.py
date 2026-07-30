"""Regression tests for specification violations found in the code audit.

These tests intentionally fail against the implementation audited on
2026-07-30.  They describe required contracts; production code is not fixed
as part of the audit.
"""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

from scripts import train_hierarchical_language_jepa as train
from textjepa.analysis.hierarchical_language import representation_statistics
from textjepa.data.hierarchical_language import (
    BoundaryPolicy,
    CounterfactualBatch,
)
from textjepa.models.hierarchical_language_jepa import (
    ContextualControlledPredictor,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.objectives.hierarchical_language import (
    EMAShrunkMahalanobis,
    listwise_value_loss,
    recursive_rollout_loss,
)
from textjepa.planning.hierarchical_language import (
    score_token_candidates,
    value_guided_high_level_cost,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def _tiny_model(*, macro: bool = False, value: bool = False):
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=10,
        vocab_size=20,
        pad_id=0,
        d_token=6,
        d_sentence=4,
        d_action=3,
        predictor_width=8,
        token_layers=1,
        sentence_layers=1,
        n_heads=2,
        token_context=4,
        sentence_context=3,
        max_span=5,
        enable_macro_actions=macro,
        enable_value=value,
    ))


def _identity_metric(dimension: int):
    metric = EMAShrunkMahalanobis(
        dimension, momentum=0.0, shrinkage=1.0, epsilon=1e-8
    )
    metric.covariance.copy_(torch.eye(dimension))
    return metric


@pytest.mark.parametrize("boundaries", [[-2, 3], [0, 3, 9]])
def test_boundary_policy_rejects_every_out_of_range_boundary(boundaries):
    with pytest.raises(ValueError, match="outside"):
        BoundaryPolicy().normalize(boundaries, sequence_length=5)


def test_counterfactual_batch_rejects_misaligned_temperatures():
    batch = CounterfactualBatch(
        token_ids=torch.zeros(2, 3, 4, dtype=torch.long),
        hidden_states=torch.zeros(2, 3, 4, 5),
        lengths=torch.full((2, 3), 4),
        root_ids=torch.zeros(2, 3, dtype=torch.long),
        # The same element count currently slips through flatten(), despite
        # not assigning one scalar temperature to each root/candidate pair.
        temperatures=torch.zeros(2, 3, 1),
    )
    with pytest.raises(ValueError, match="temperature"):
        batch.flatten()


def test_counterfactual_batch_is_directly_consumable_by_offline_learner():
    batch = CounterfactualBatch(
        token_ids=torch.zeros(1, 2, 4, dtype=torch.long),
        hidden_states=torch.zeros(1, 2, 4, 5),
        lengths=torch.tensor([[4, 3]]),
        root_ids=torch.zeros(1, 2, dtype=torch.long),
        temperatures=torch.tensor([[0.0, 0.7]]),
    )
    flat = batch.flatten()
    assert "boundaries" in flat


def test_dense_forward_rejects_boundary_after_padding():
    model = _tiny_model()
    hidden = torch.randn(1, 6, 10)
    token_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])
    boundaries = torch.tensor([[0, 2, 5]])
    with pytest.raises(ValueError, match="padding"):
        model.dense_forward(hidden, token_ids, boundaries)


def test_dense_forward_rejects_noncontiguous_boundary_padding():
    model = _tiny_model()
    hidden = torch.randn(1, 6, 10)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match="contiguous"):
        model.dense_forward(hidden, token_ids, torch.tensor([[0, -1, 5]]))


def test_token_only_stage_does_not_execute_sentence_stack():
    model = _tiny_model()
    calls = []
    handle = model.sentence_predictor.register_forward_hook(
        lambda *unused: calls.append(True)
    )
    hidden = torch.randn(1, 6, 10)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    boundaries = torch.tensor([[0, 3, 5]])
    HierarchicalLanguageLearner(
        model, ResearchStage.TOKEN_JEPA
    )(hidden, token_ids, boundaries, random_context_truncation=False)
    handle.remove()
    assert calls == []


def test_recursive_rollout_loss_preserves_predictor_context():
    torch.manual_seed(3)
    predictor = ContextualControlledPredictor(
        d_state=4,
        d_action=3,
        width=8,
        layers=1,
        heads=2,
        max_context=4,
    ).eval()
    start = torch.randn(1, 4)
    actions = torch.randn(1, 3, 3)
    with torch.no_grad():
        targets, _ = predictor.rollout(start[0], actions)
    loss = recursive_rollout_loss(
        predictor,
        start,
        actions,
        targets,
        torch.ones(1, 3, dtype=torch.bool),
        _identity_metric(4),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_macro_prior_context_does_not_contain_current_observed_action():
    torch.manual_seed(5)
    model = _tiny_model(macro=True).eval()
    states = torch.randn(1, 2, model.config.d_sentence)
    valid = torch.ones(1, 2, dtype=torch.bool)
    action_a = torch.zeros(1, 2, model.config.d_action)
    action_b = action_a.clone()
    action_b[:, 0] = 10
    with torch.no_grad():
        _, context_a = model.sentence_predictor(
            states, action_a, valid, return_context=True
        )
        _, context_b = model.sentence_predictor(
            states, action_b, valid, return_context=True
        )
    # p(u_0 | xi_0, q) is causal: its conditioning context cannot change
    # when the very u_0 it is supposed to predict is changed.
    assert torch.allclose(context_a[:, 0], context_b[:, 0], atol=1e-7)


def test_value_stage_requires_an_enabled_value_head():
    model = _tiny_model(macro=True, value=False)
    with pytest.raises(ValueError, match="value"):
        HierarchicalLanguageLearner(model, ResearchStage.VALUE_DISTILLATION)


def test_counterfactual_stage_accepts_sparse_recursive_rollout_batches():
    parameters = inspect.signature(HierarchicalLanguageLearner.forward).parameters
    assert {
        "token_rollout_actions",
        "token_rollout_targets",
        "token_rollout_mask",
    } <= parameters.keys()


def test_value_receives_current_state_as_well_as_predictor_context():
    model = _tiny_model(value=True)
    parameters = inspect.signature(model.value.forward).parameters
    assert tuple(parameters) == ("state", "context", "task")


def test_listwise_value_loss_rejects_root_without_actions():
    with pytest.raises(ValueError, match="valid action"):
        listwise_value_loss(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[False, False]]),
            teacher_temperature=1.0,
            value_temperature=1.0,
        )


def test_value_guided_cost_accepts_final_context_without_time_axis():
    context = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    cost = value_guided_high_level_cost(
        context,
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        lambda final_context, task: final_context.sum(-1),
        step_cost=0.0,
        prior_weight=0.0,
    )
    assert cost.tolist() == pytest.approx([3.0, 9.0])


def test_token_candidate_scoring_rejects_nonvector_log_probabilities():
    with pytest.raises(ValueError, match="log probability"):
        score_token_candidates(
            predicted_endpoints=torch.zeros(2, 2),
            candidate_tokens=torch.tensor([[1], [2]]),
            candidate_log_probability=torch.zeros(2, 1),
            waypoint=torch.zeros(2),
            fine_to_coarse=lambda value: value,
            metric=_identity_metric(2),
            prior_weight=1.0,
            temperature=1.0,
            vocab_size=3,
        )


def test_token_candidate_scoring_rejects_negative_prior_weight():
    with pytest.raises(ValueError, match="nonnegative"):
        score_token_candidates(
            predicted_endpoints=torch.zeros(2, 2),
            candidate_tokens=torch.tensor([[1], [2]]),
            candidate_log_probability=torch.zeros(2),
            waypoint=torch.zeros(2),
            fine_to_coarse=lambda value: value,
            metric=_identity_metric(2),
            prior_weight=-1.0,
            temperature=1.0,
            vocab_size=3,
        )


def test_ema_covariance_accounts_for_between_batch_mean_shift():
    metric = EMAShrunkMahalanobis(
        1, momentum=0.5, shrinkage=0.0, epsilon=0.0
    )
    metric.update(torch.zeros(2, 1))
    metric.update(torch.full((2, 1), 10.0))
    assert metric.covariance.item() > 0


def test_rank_deficient_representation_has_infinite_condition_number():
    states = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    stats = representation_statistics(states)
    assert stats["condition_number"] == float("inf")


def test_feature_loader_rejects_noninteger_token_ids(tmp_path):
    path = tmp_path / "bad_features.pt"
    torch.save({
        "hidden_states": torch.randn(2, 4, 5),
        "token_ids": torch.ones(2, 4, dtype=torch.float32),
        "boundaries": torch.tensor([[0, 3], [0, 3]]),
    }, path)
    with pytest.raises(ValueError, match="token.*integer"):
        train.load_features(path)


def test_value_training_stage_cannot_skip_predecessor_checkpoint(
    tmp_path, monkeypatch
):
    features = tmp_path / "features.pt"
    output = tmp_path / "run"
    torch.save({
        "hidden_states": torch.randn(1, 4, 6),
        "token_ids": torch.tensor([[1, 2, 3, 4]]),
        "boundaries": torch.tensor([[0, 3]]),
    }, features)
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features),
        "--allow-unpinned-features",
        "--output", str(output),
        "--stage", "VALUE_DISTILLATION",
        "--vocab-size", "8",
        "--pad-id", "0",
        "--d-token", "4",
        "--d-sentence", "4",
        "--d-action", "2",
        "--predictor-width", "4",
        "--token-layers", "1",
        "--sentence-layers", "1",
        "--heads", "1",
        "--epochs", "0",
        "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="requires --admission"):
        train.main()


def test_checkpoint_preserves_learned_mahalanobis_geometry(
    tmp_path, monkeypatch
):
    features = tmp_path / "features.pt"
    output = tmp_path / "run"
    torch.save({
        "hidden_states": torch.randn(2, 4, 6),
        "token_ids": torch.tensor([[1, 2, 3, 4], [1, 3, 2, 4]]),
        "boundaries": torch.tensor([[0, 3], [0, 3]]),
    }, features)
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features),
        "--allow-unpinned-features",
        "--output", str(output),
        "--vocab-size", "8",
        "--pad-id", "0",
        "--d-token", "4",
        "--d-sentence", "4",
        "--d-action", "2",
        "--predictor-width", "4",
        "--token-layers", "1",
        "--sentence-layers", "1",
        "--heads", "1",
        "--epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
    ])
    train.main()
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert "learner" in checkpoint
    assert "token_metric.covariance" in checkpoint["learner"]
