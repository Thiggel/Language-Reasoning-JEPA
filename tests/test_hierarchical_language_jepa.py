import inspect

import pytest
import torch

from textjepa.models.hierarchical_language_jepa import (
    ContextualControlledPredictor,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.objectives.hierarchical_language import (
    EMAShrunkMahalanobis,
    diagonal_gaussian_kl,
    listwise_value_loss,
    terminal_set_discrepancy,
    value_teacher_softmin,
    vicreg_floor_and_covariance,
)
from textjepa.training.hierarchical_language import (
    DenseLossWeights,
    HierarchicalLanguageLearner,
    ResearchStage,
    value_distillation_loss,
)


def tiny_model(cache_dropout=0.0, enable_macro=False, enable_value=False):
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=12, vocab_size=31, pad_id=0,
        d_token=8, d_sentence=6, d_action=4, predictor_width=16,
        token_layers=1, sentence_layers=1, n_heads=2,
        token_context=4, sentence_context=3, cache_dropout=cache_dropout,
        max_span=5,
        enable_macro_actions=enable_macro, enable_value=enable_value,
    ))


def batch():
    torch.manual_seed(4)
    hidden = torch.randn(2, 9, 12)
    tokens = torch.randint(1, 31, (2, 9))
    tokens[1, 8] = 0
    boundaries = torch.tensor([[0, 3, 8, -1], [0, 4, 7, -1]])
    return hidden, tokens, boundaries


def test_dense_forward_supervises_all_token_and_sentence_positions():
    model = tiny_model()
    hidden, tokens, boundaries = batch()
    out = model.dense_forward(hidden, tokens, boundaries)
    assert out["token_predictions"].shape == (2, 8, 8)
    assert out["token_targets"].shape == (2, 8, 8)
    assert out["sentence_predictions"].shape == (2, 3, 6)
    assert out["sentence_valid"].tolist() == [
        [True, True, False], [True, True, False]
    ]
    assert out["fine_at_boundaries"].shape == (2, 4, 8)


def test_context_bound_is_sliding_not_loss_of_dense_targets():
    predictor = ContextualControlledPredictor(6, 4, 12, 1, 2, 3)
    states = torch.randn(2, 11, 6)
    actions = torch.randn(2, 11, 4)
    calls = []
    handle = predictor.blocks.register_forward_hook(
        lambda *unused: calls.append(1)
    )
    prediction = predictor(
        states, actions, random_truncation=True
    )
    handle.remove()
    assert prediction.shape == states.shape
    assert len(calls) == 1


def test_each_rollout_branch_carries_distinct_history():
    predictor = ContextualControlledPredictor(6, 4, 12, 1, 2, 4)
    predictor.eval()
    actions = torch.randn(2, 2, 4)
    history = torch.randn(2, 3, 6)
    action_history = torch.randn(2, 2, 4)
    rollout, _ = predictor.rollout(
        history[:, -1], actions,
        state_history=history, action_history=action_history,
    )
    assert rollout.shape == (2, 2, 6)
    assert not torch.allclose(rollout[0], rollout[1])


def test_rollout_rejects_zero_horizon_and_misaligned_cache():
    predictor = ContextualControlledPredictor(6, 4, 12, 1, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        predictor.rollout(torch.zeros(6), torch.empty(2, 0, 4))
    with pytest.raises(ValueError, match="one fewer"):
        predictor.rollout(
            torch.zeros(6), torch.randn(2, 1, 4),
            state_history=torch.randn(2, 3, 6),
            action_history=torch.randn(2, 1, 4),
        )


def test_target_projectors_are_frozen_and_update_only_by_ema():
    model = tiny_model()
    assert not any(p.requires_grad for p in model.token_target.parameters())
    before = [p.clone() for p in model.token_target.parameters()]
    with torch.no_grad():
        next(model.token_projector.parameters()).add_(1)
    model.update_targets(0.5)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, model.token_target.parameters())
    )


def test_boundary_validation_rejects_nonincreasing_and_long_spans():
    model = tiny_model()
    hidden, tokens, _ = batch()
    with pytest.raises(ValueError, match="strictly increasing"):
        model.dense_forward(
            hidden, tokens, torch.tensor([[0, 4, 3], [0, 2, 7]])
        )
    with pytest.raises(ValueError, match="exceeds max_span"):
        model.dense_forward(
            hidden, tokens, torch.tensor([[0, 8], [0, 4]])
        )


def test_initial_stage_has_no_sentence_alignment_macro_or_value_loss():
    model = tiny_model()
    learner = HierarchicalLanguageLearner(model, ResearchStage.TOKEN_JEPA)
    total, losses = learner(*batch())
    assert set(losses) == {
        "token_dynamics", "token_variance", "token_covariance", "total"
    }
    total.backward()
    assert model.value is None
    assert model.macro_actions is None
    assert not any(p.grad is not None for p in model.e0_to_1.parameters())


def test_sentence_and_dynamic_commutation_losses_are_explicitly_gated():
    model = tiny_model()
    _, sentence = HierarchicalLanguageLearner(
        model, ResearchStage.SENTENCE_JEPA
    )(*batch())
    assert "sentence_dynamics" in sentence
    assert "alignment" not in sentence
    _, cross = HierarchicalLanguageLearner(
        model, ResearchStage.CROSS_LEVEL
    )(*batch())
    assert "alignment" not in cross
    assert "commutation" not in cross
    with pytest.raises(ValueError, match="commutation"):
        HierarchicalLanguageLearner(
            model, ResearchStage.TOKEN_JEPA,
            DenseLossWeights(commutation=1.0),
        )


def test_sentence_state_is_strictly_nested_and_has_no_direct_projector():
    model = tiny_model()
    hidden = torch.randn(2, 4, 12)
    token_state = model.e0(hidden)
    expected = model.e0_to_1(token_state)
    assert torch.allclose(model.encode_sentence(hidden), expected)
    assert "sentence_projector" not in model._modules
    assert not any(
        "sentence_projector" in name for name in model.state_dict()
    )


def test_sentence_target_is_the_nested_ema_tower():
    model = tiny_model()
    hidden = torch.randn(2, 12)
    expected = model.e0_to_1_target(model.e0_target(hidden))
    assert torch.allclose(
        model.encode_sentence(hidden, target=True), expected
    )


def test_a1_depends_only_on_sentence_tokens():
    model = tiny_model()
    span = torch.tensor([[[2, 3, 4, 0, 0]]])
    mask = span.ne(0)
    first = model.a1(span, mask)
    # There is deliberately no state or LM-hidden argument to vary.
    second = model.a1(span.clone(), mask.clone())
    assert torch.allclose(first, second)
    assert tuple(inspect.signature(model.a1.forward).parameters) == (
        "span_token_ids", "span_mask"
    )


def test_nested_sentence_stage_does_not_execute_token_predictor():
    model = tiny_model()
    calls = []
    handle = model.p0.register_forward_hook(
        lambda *unused: calls.append(True)
    )
    _, losses = HierarchicalLanguageLearner(
        model, ResearchStage.NESTED_SENTENCE
    )(*batch())
    handle.remove()
    assert not calls
    assert "token_dynamics" not in losses


def test_frozen_token_encoder_still_trains_nested_sentence_modules():
    model = tiny_model()
    model.e0.requires_grad_(False)
    model.p0.requires_grad_(False)
    total, _ = HierarchicalLanguageLearner(
        model, ResearchStage.NESTED_SENTENCE
    )(*batch())
    total.backward()
    assert not any(parameter.grad is not None for parameter in model.e0.parameters())
    assert any(
        parameter.grad is not None for parameter in model.e0_to_1.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.a1.parameters())
    assert any(parameter.grad is not None for parameter in model.p1.parameters())


def test_dense_and_rollout_share_bounded_position_semantics():
    torch.manual_seed(9)
    predictor = ContextualControlledPredictor(6, 4, 12, 1, 2, 4)
    predictor.eval()
    states = torch.randn(1, 7, 6)
    actions = torch.randn(1, 7, 4)
    dense = predictor(states, actions)
    rollout, _ = predictor.rollout(
        states[0, -1],
        actions[:, -1:],
        state_history=states[:, -4:],
        action_history=actions[:, -4:-1],
    )
    assert torch.allclose(dense[:, -1], rollout[:, 0], atol=1e-6)


def test_macro_stage_trains_posterior_prior_and_dynamics():
    model = tiny_model(enable_macro=True)
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.MACRO_ACTION, macro_free_bits=0.01
    )
    total, losses = learner(*batch())
    assert torch.isfinite(total)
    assert losses["macro"] > 0
    total.backward()
    assert any(p.grad is not None for p in model.macro_actions.parameters())


def test_value_has_no_budget_argument_or_budget_parameter():
    initial = tiny_model()
    assert initial.value is None
    assert initial.macro_actions is None
    signature = inspect.signature(tiny_model(enable_value=True).value.forward)
    assert tuple(signature.parameters) == ("state", "context", "task")
    names = " ".join(name.lower() for name, _ in tiny_model().named_parameters())
    assert "budget" not in names


def test_value_replay_task_hidden_trains_current_task_projection():
    model = tiny_model(enable_macro=True, enable_value=True)
    task_hidden = torch.randn(2, 12)
    task = model.task_projection(task_hidden)
    loss = value_distillation_loss(
        model,
        torch.randn(2, 3, 6),
        torch.randn(2, 3, 16),
        task,
        torch.zeros(2, 3),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]),
        torch.ones(2, 3, dtype=torch.bool),
        step_cost=0, prior_weight=0,
        teacher_temperature=1, value_temperature=1,
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.task_projection.parameters()
    )


def test_mahalanobis_is_invariant_to_joint_invertible_linear_transform():
    torch.manual_seed(0)
    states = torch.randn(50, 3)
    left, right = states[:5], states[5:10]
    metric = EMAShrunkMahalanobis(3, momentum=0, shrinkage=0, epsilon=1e-7)
    metric.update(states)
    transform = torch.tensor([[2.0, 0.3, 0.1], [0.2, 1.5, 0.0], [0.0, 0.4, 0.8]])
    transformed = states @ transform.T
    other = EMAShrunkMahalanobis(3, momentum=0, shrinkage=0, epsilon=1e-7)
    other.update(transformed)
    assert torch.allclose(
        metric(left, right),
        other(left @ transform.T, right @ transform.T),
        rtol=2e-4, atol=2e-4,
    )


def test_mahalanobis_promotes_low_precision_difference_for_stable_solve():
    metric = EMAShrunkMahalanobis(2)
    distance = metric(
        torch.ones(3, 2, dtype=torch.float16),
        torch.zeros(3, 2, dtype=torch.float16),
    )
    assert distance.dtype == torch.float32
    assert torch.isfinite(distance).all()


def test_terminal_set_soft_min_and_mask():
    metric = EMAShrunkMahalanobis(2, shrinkage=1)
    metric.covariance.copy_(torch.eye(2))
    state = torch.tensor([[0.0, 0.0]])
    goals = torch.tensor([[[1.0, 0.0], [10.0, 0.0]]])
    masked = terminal_set_discrepancy(
        state, goals, metric, 0.1, torch.tensor([[True, False]])
    )
    assert masked.item() == pytest.approx(metric(state, goals[:, 0]).item())


def test_vicreg_uses_squared_variance_hinge():
    states = torch.zeros(4, 3)
    variance, covariance = vicreg_floor_and_covariance(
        states, torch.ones(4, dtype=torch.bool), variance_floor=1
    )
    assert variance.item() == pytest.approx((1 - 0.01) ** 2)
    assert covariance.item() == 0


def test_value_teacher_depth_is_reduced_not_exposed_as_input():
    costs = torch.tensor([[[[3.0, 2.0], [1.0, 4.0]]]])
    mask = torch.ones_like(costs, dtype=torch.bool)
    target = value_teacher_softmin(costs, mask, temperature=0.01)
    assert target.shape == (1, 1)
    assert target.item() == pytest.approx(1.0, abs=1e-3)


def test_value_distillation_reads_successor_context_and_task_only():
    model = tiny_model(enable_value=True)
    loss = value_distillation_loss(
        model,
        successor_state=torch.randn(2, 3, 6),
        successor_context=torch.randn(2, 3, 16),
        task=torch.randn(2, 256),
        first_action_log_probability=torch.randn(2, 3),
        teacher_cost=torch.randn(2, 3),
        action_mask=torch.ones(2, 3, dtype=torch.bool),
        step_cost=0.1, prior_weight=0.2,
        teacher_temperature=1, value_temperature=1,
    )
    loss.backward()
    assert any(p.grad is not None for p in model.value.parameters())


def test_listwise_value_loss_rewards_correct_within_root_ranking():
    teacher = torch.tensor([[0.0, 1.0, 3.0]])
    good = listwise_value_loss(
        teacher, teacher, torch.ones_like(teacher, dtype=torch.bool), 1, 1
    )
    bad = listwise_value_loss(
        teacher, -teacher, torch.ones_like(teacher, dtype=torch.bool), 1, 1
    )
    assert good < bad


def test_diagonal_gaussian_kl_is_zero_for_identical_distributions():
    mean = torch.randn(2, 4)
    logvar = torch.randn(2, 4).clamp(-2, 2)
    assert torch.allclose(
        diagonal_gaussian_kl(mean, logvar, mean, logvar),
        torch.zeros(2), atol=1e-6,
    )


def test_end_to_end_token_learner_reduces_fixed_batch_objective():
    torch.manual_seed(11)
    model = tiny_model()
    model.eval()
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.TOKEN_JEPA,
        DenseLossWeights(variance=0.0, covariance=0.0),
        covariance_momentum=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    data = batch()
    initial = float(learner(*data, random_context_truncation=False)[0])
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = learner(*data, random_context_truncation=False)
        loss.backward()
        optimizer.step()
    final = float(learner(*data, random_context_truncation=False)[0])
    assert final < 0.7 * initial
