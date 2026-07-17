from types import SimpleNamespace

import torch

from scripts.audit_faithful_token_edits import (
    depth_summary,
    operation_summary,
    pad_concat_2d,
    safe_correlation,
    shuffled_action_prediction,
)


def test_error_strata_report_counts_and_human_operation_names():
    values = torch.tensor([[1.0, 2.0, 9.0], [3.0, 5.0, 7.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    operations = torch.tensor([[0, 1, -1], [2, 0, 1]])

    by_depth = depth_summary(values, mask)
    assert by_depth == {
        "1": {"ln_l1": 2.0, "steps": 2},
        "2": {"ln_l1": 3.5, "steps": 2},
        "3": {"ln_l1": 7.0, "steps": 1},
    }
    by_operation = operation_summary(values, mask, operations)
    assert by_operation["delete"] == {"ln_l1": 3.0, "steps": 2}
    assert by_operation["insert"] == {"ln_l1": 4.5, "steps": 2}
    assert by_operation["replace"] == {"ln_l1": 3.0, "steps": 1}


def test_variable_batch_trajectory_widths_are_padded_before_aggregation():
    combined = pad_concat_2d([
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0, 5.0]]),
    ], fill=-1)
    assert combined.tolist() == [[1.0, 2.0, -1.0], [3.0, 4.0, 5.0]]


def test_shuffled_control_uses_deranged_valid_actions_and_core_prefix_api():
    class Core:
        def _predict_counterfactuals(self, prev, actions, alternatives, mask):
            self.call = (prev, actions, alternatives, mask)
            return alternatives

    core = Core()
    model = SimpleNamespace(core=core, attn_pred=None)
    actions = torch.tensor([[[1.0], [2.0]], [[3.0], [0.0]]])
    valid = torch.tensor([[True, True], [True, False]])
    out = SimpleNamespace(
        actions=actions,
        prev_states=torch.zeros(2, 2, 1),
        step_mask=valid,
    )

    prediction, reason, changed = shuffled_action_prediction(model, out)
    assert reason is None
    assert prediction[valid].flatten().tolist() == [3.0, 1.0, 2.0]
    assert torch.equal(changed, valid)
    # Invalid padding is not introduced into the shuffled pool.
    assert prediction[~valid].item() == 0.0
    assert core.call[2].shape == (2, 2, 1, 1)


def test_shuffled_control_declines_when_independent_prefix_is_unavailable():
    model = SimpleNamespace(core=object(), attn_pred=None)
    out = SimpleNamespace(
        actions=torch.zeros(1, 2, 1),
        prev_states=torch.zeros(1, 2, 1),
        step_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    prediction, reason, changed = shuffled_action_prediction(model, out)
    assert prediction is None
    assert changed is None
    assert reason == "independent_causal_prefix_api_unavailable"


def test_privileged_correlation_is_json_safe_for_constant_values():
    assert safe_correlation(torch.ones(3).numpy(), torch.arange(3).numpy()) is None
