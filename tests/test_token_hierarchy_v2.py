import torch
from unittest.mock import patch

from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.models.action import MacroActionModel
from textjepa.planning.token_hierarchy import (
    feedback_levels_to_invalidate,
    macro_codes,
    remaining_to_boundary,
)


def test_multilevel_token_hierarchy_shapes_and_dense_rollout():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=80,
        pad_id=0,
        d_model=32,
        encoder_layers=1,
        predictor_layers=1,
        n_heads=2,
        ff_mult=2,
        max_len=64,
        d_action=8,
        level_spans=[4, 8],
        level_dims=[6, 4],
        variational_levels=[False, False],
        concat_width=2,
        low_dense_depth=3,
        high_dense_depth=2,
    )
    tokens = torch.randint(1, 80, (3, 28))
    prompt_len = torch.tensor([8, 9, 10])
    out = model(tokens, prompt_len)
    assert out["low_pred"].shape[:2] == out["valid"].shape
    assert len(out["low_dense_predictions"]) == 3
    assert out["token_prior_logits"] is None
    assert [level["span"] for level in out["levels"]] == [4, 8]
    assert out["levels"][0]["codes"].shape[-1] == 6
    assert out["levels"][1]["codes"].shape[-1] == 4
    for level in out["levels"]:
        assert level["pred"].shape == level["target"].shape
        assert level["recursive_low_endpoint"].shape == level["target"].shape
        assert torch.isfinite(level["prior_nll"]).all()


def test_multilevel_token_hierarchy_requires_nested_spans():
    try:
        MultilevelTokenHierarchyJEPA(
            vocab_size=20, pad_id=0, d_model=16, encoder_layers=1,
            predictor_layers=1, n_heads=2, level_spans=[6, 10],
            level_dims=[4, 4], variational_levels=[False],
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-nested spans must be rejected")


def test_partial_macro_history_does_not_construct_empty_higher_level():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=20, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4, 16], level_dims=[6, 4],
        variational_levels=[False, False], concat_width=2,
    )
    tokens = torch.randint(1, 20, (1, 4))
    codes = macro_codes(model, tokens, through_level=0)
    assert len(codes) == 1
    assert codes[0].shape == (1, 1, 6)


def test_recursive_low_endpoint_retains_causal_prefix_history():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=30, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4], level_dims=[6],
        variational_levels=[False], concat_width=2,
    ).eval()
    tokens = torch.randint(1, 30, (1, 20))
    prompt_len = torch.tensor([8])
    with torch.no_grad():
        out = model(tokens, prompt_len)
        level = out["levels"][0]
        # The second macro begins after four reasoning tokens.
        explicit = model.low_predictor.rollout(
            level["prev"][:, 1], level["raw_action_windows"][:, 1],
            state_history=out["prev"][:, :5],
            action_history=out["token_actions"][:, :4],
        )[:, -1]
    assert torch.allclose(level["recursive_low_endpoint"][:, 1], explicit)


def test_cross_level_feedback_invalidation_policy():
    assert feedback_levels_to_invalidate("boundary", 2.0, .5, 3) == ()
    assert feedback_levels_to_invalidate("l1_feedback", .1, .5, 3) == (1, 2)
    assert feedback_levels_to_invalidate("adaptive", .4, .5, 3) == ()
    assert feedback_levels_to_invalidate("adaptive", .6, .5, 3) == (1, 2)


def test_phase_augmented_upper_level_uses_valid_lower_boundary_offsets():
    torch.manual_seed(7)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=40, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4, 8], level_dims=[6, 4],
        variational_levels=[False, False],
        phase_augmented_levels=[False, True], concat_width=2,
    ).train()
    tokens = torch.randint(1, 40, (8, 28))
    prompt_len = torch.full((8,), 8)
    out = model(tokens, prompt_len)
    level = out["levels"][1]
    assert set(level["phase_offsets"].tolist()).issubset({0, 4})
    for row, offset in enumerate(level["phase_offsets"].tolist()):
        if level["valid"][row, 0]:
            assert level["raw_action_ids"][row, 0, 0] == out["action_ids"][row, offset]


def test_three_level_phase_augmentation_uses_available_lower_macro_grid():
    torch.manual_seed(13)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=40, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=256,
        d_action=8, level_spans=[8, 32, 96], level_dims=[8, 6, 4],
        variational_levels=[False],
        phase_augmented_levels=[False, True, True], concat_width=4,
    ).train()
    tokens = torch.randint(1, 40, (12, 180))
    prompt_len = torch.full((12,), 12)
    out = model(tokens, prompt_len)
    for level in out["levels"]:
        for row, offset in enumerate(level["phase_offsets"].tolist()):
            if level["valid"][row, 0]:
                assert (
                    level["raw_action_ids"][row, 0, 0]
                    == out["action_ids"][row, offset]
                )
    assert out["levels"][2]["codes"].shape[-1] == 4
    assert out["levels"][2]["valid"].any()


def test_token_mpc_horizon_counts_down_to_fixed_boundary():
    assert [remaining_to_boundary(p, 8) for p in range(9)] == [8, 7, 6, 5, 4, 3, 2, 1, 8]


def test_transformer_macro_encoder_ignores_padded_actions():
    torch.manual_seed(3)
    model = MacroActionModel(
        d_action=8, d_state=16, d_macro=6, span=8, kind="transformer"
    ).eval()
    actions = torch.randn(2, 5, 8)
    valid = torch.tensor([[True, True, True, False, False]] * 2)
    actions[1, :3] = actions[0, :3]
    actions[1, 3:] = 100 * torch.randn_like(actions[1, 3:])
    states = torch.randn(2, 16)
    code, _ = model.training_code(actions, states, valid)
    assert torch.allclose(code[0], code[1], atol=1e-6)


def test_state_conditioned_token_prior_shape_and_detached_encoder_gradient():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=30, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4], level_dims=[6],
        variational_levels=[False], use_token_prior=True,
        token_prior_hidden=12, token_prior_detach_state=True,
    )
    tokens = torch.randint(1, 30, (2, 20))
    out = model(tokens, torch.tensor([8, 8]))
    assert out["token_prior_logits"].shape == (*out["valid"].shape, 30)
    assert len(out["token_prior_rollout_logits"]) == model.low_dense_depth
    for horizon, logits in enumerate(out["token_prior_rollout_logits"], 1):
        assert logits.shape == (2, out["valid"].shape[1] - horizon, 30)
    model.zero_grad(set_to_none=True)
    loss = out["token_prior_logits"].sum() + sum(
        logits.sum() for logits in out["token_prior_rollout_logits"]
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.token_prior.parameters())


def test_distinct_level_states_are_causal_and_goal_liftable():
    torch.manual_seed(17)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=40, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[2, 4], level_dims=[6, 4],
        variational_levels=[False], distinct_level_states=True,
        level_state_encoder_layers=1,
    ).eval()
    assert all(level.state_encoder is not None for level in model.levels)
    path = torch.randn(2, 9, 16)
    changed = path.clone()
    changed[:, 6:] = 100 * torch.randn_like(changed[:, 6:])
    lifted = model.lift_state_path(path)
    changed_lifted = model.lift_state_path(changed)
    assert len(lifted) == 2
    assert lifted[0].shape == path.shape
    assert torch.allclose(lifted[0][:, :6], changed_lifted[0][:, :6], atol=1e-5)
    assert lifted[1].shape[-1] == 16
    teacher_lifted = model.lift_state_path(path, teacher=True)
    assert all(not value.requires_grad for value in teacher_lifted)


def test_distinct_level_state_teacher_updates_with_ema():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=20, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=32,
        d_action=8, level_spans=[2], level_dims=[4],
        variational_levels=[False], distinct_level_states=True,
        level_state_encoder_layers=1,
    )
    online = next(model.levels[0].state_encoder.parameters())
    target = next(model.levels[0].state_teacher.parameters())
    with torch.no_grad():
        online.add_(1.0)
    before = target.clone()
    model.update_teacher(0.5)
    assert not torch.allclose(before, target)


def test_distinct_reachability_lifts_primitive_rollouts_with_ema_coordinates():
    torch.manual_seed(13)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=30, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[2, 4], level_dims=[6, 4],
        variational_levels=[False], distinct_level_states=True,
        level_state_encoder_layers=1,
    ).eval()
    tokens = torch.randint(1, 30, (2, 20))
    with patch.object(model, "lift_state_path", wraps=model.lift_state_path) as lift:
        out = model(tokens, torch.tensor([8, 8]))
    assert out["levels"][0]["recursive_low_endpoint"].shape == out["levels"][0]["target"].shape
    assert out["levels"][1]["recursive_low_endpoint"].shape == out["levels"][1]["target"].shape
    assert lift.call_count > 0
    assert all(call.kwargs["teacher"] is True for call in lift.call_args_list)
    assert {call.kwargs["through_level"] for call in lift.call_args_list} == {0, 1}
