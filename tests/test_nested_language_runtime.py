import torch

from textjepa.models.hierarchical_language_jepa import (
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.planning.nested_language_runtime import (
    advance_sentence_planning_state,
    contextual_prior_cem,
    make_sentence_planning_state,
    rollout_prior_noise,
    sentence_pre_action_context,
    sentence_planning_state_from_trace,
)


def tiny_model(*, context=4):
    torch.manual_seed(3)
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=8,
        vocab_size=17,
        pad_id=0,
        d_token=6,
        d_sentence=4,
        d_action=2,
        d_task=3,
        predictor_width=8,
        token_layers=1,
        sentence_layers=1,
        n_heads=2,
        token_context=4,
        sentence_context=context,
        max_span=6,
        enable_macro_actions=True,
        enable_value=True,
    )).eval()


def test_pre_action_context_does_not_see_dummy_current_action():
    model = tiny_model()
    states = torch.randn(2, 3, 4)
    actions = torch.randn(2, 2, 2)
    expected = sentence_pre_action_context(model, states, actions)
    aligned = torch.cat([actions, torch.randn(2, 1, 2) * 100], 1)
    valid = torch.ones(2, 3, dtype=torch.bool)
    _, context = model.p1(states, aligned, valid, return_context=True)
    torch.testing.assert_close(expected, context[:, -1])


def test_successor_context_preserves_and_bounds_the_real_history():
    model = tiny_model(context=3)
    states = torch.randn(1, 3, 4)
    actions = torch.randn(1, 2, 2)
    root = make_sentence_planning_state(model, states, actions)
    successor = advance_sentence_planning_state(
        model, root, torch.randn(1, 2)
    )
    assert successor.state_history.shape == (1, 3, 4)
    assert successor.action_history.shape == (1, 2, 2)
    torch.testing.assert_close(
        successor.context,
        sentence_pre_action_context(
            model, successor.state_history, successor.action_history
        ),
    )


def test_trace_reconstruction_uses_global_prefix_boundaries():
    model = tiny_model()
    hidden = torch.randn(9, 8)
    tokens = torch.arange(9) + 1
    boundaries = torch.tensor([3, 5, 9])
    state = sentence_planning_state_from_trace(
        model, hidden, tokens, boundaries, boundary_index=2
    )
    expected_states = model.encode_sentence(hidden[boundaries - 1])
    torch.testing.assert_close(state.state_history[0], expected_states)
    ids = torch.tensor([[4, 5, 0, 0], [6, 7, 8, 9]])
    mask = ids.ne(0)
    expected_actions = model.a1(ids, mask)
    torch.testing.assert_close(state.action_history[0], expected_actions)


def test_rollout_prior_noise_keeps_prefix_sensitive_contexts():
    model = tiny_model()
    current = torch.randn(1, 1, 4)
    root = make_sentence_planning_state(
        model, current, torch.empty(1, 0, 2)
    )
    noise = torch.zeros(5, 3, 2)
    rollout = rollout_prior_noise(model, root, torch.zeros(3), noise)
    assert rollout.states.shape == (5, 3, 4)
    assert rollout.contexts.shape == (5, 3, 8)
    assert rollout.log_probabilities.shape == (5, 3)
    torch.testing.assert_close(
        rollout.contexts[:, -1], rollout.final_planning_state.context
    )


def test_contextual_cem_scores_variable_prefixes_and_returns_winner():
    model = tiny_model()
    root = make_sentence_planning_state(
        model, torch.zeros(1, 1, 4), torch.empty(1, 0, 2)
    )

    def objective(rollout):
        prefix_cost = rollout.states[..., 0].square()
        return prefix_cost.min(-1)

    result = contextual_prior_cem(
        model, root, torch.zeros(3), objective,
        horizon=3, population=16, iterations=2,
        generator=torch.Generator().manual_seed(4),
    )
    assert result.rollout.states.shape == (1, 3, 4)
    assert result.selected_prefix in {0, 1, 2}
    assert len(result.diagnostics) == 2


def test_contextual_cem_uses_grounded_cost_for_elite_updates():
    model = tiny_model()
    root = make_sentence_planning_state(
        model, torch.zeros(1, 1, 4), torch.empty(1, 0, 2)
    )

    def objective(rollout):
        # Predicted search favors low first action coordinate.
        cost = rollout.actions[:, 0, 0]
        return cost, torch.zeros_like(cost, dtype=torch.long)

    def ground(noise, rollout, ids):
        # Exact worker grounding favors the opposite direction.
        return -noise[:, 0, 0]

    result = contextual_prior_cem(
        model, root, torch.zeros(3), objective,
        horizon=1, population=64, iterations=3,
        elite_fraction=0.25, ground=ground,
        ground_topn=32, ground_random=32, select_grounded=True,
        generator=torch.Generator().manual_seed(7),
    )
    assert result.noise[0, 0] > 0
    assert all(row["grounded_candidates"] == 64 for row in result.diagnostics)
