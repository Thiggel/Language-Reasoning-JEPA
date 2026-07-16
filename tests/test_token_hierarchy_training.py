from types import SimpleNamespace

import torch

from scripts.train_token_hierarchy_v2 import compute_losses
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def objective(**overrides):
    values = dict(
        low_prediction=0.0, low_dense=0.0, low_value=0.0,
        goal_prediction=0.0, high_prediction=0.0, high_dense=0.0,
        high_level_weights=[1.0], reachability=0.0, high_value=0.0,
        macro_prior=0.0, support=0.0, vicreg=1.0, covariance=0.04,
        dense_discount=0.7, token_prior=0.0,
        token_prior_rollout=0.0, token_prior_rollout_discount=0.7,
        token_prior_label_smoothing=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_model():
    return MultilevelTokenHierarchyJEPA(
        vocab_size=30, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4], level_dims=[6],
        variational_levels=[False], concat_width=2,
    )


def test_vicreg_regularizes_online_encoder_states():
    model = tiny_model()
    out = model(torch.randint(1, 30, (2, 20)), torch.tensor([8, 8]))
    total, _ = compute_losses(out, SimpleNamespace(objective=objective()))
    total.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.encoder.parameters()
    )


def test_high_level_weights_must_match_active_levels():
    model = tiny_model()
    out = model(torch.randint(1, 30, (2, 20)), torch.tensor([8, 8]))
    cfg = SimpleNamespace(objective=objective(high_level_weights=[1.0, 2.0]))
    try:
        compute_losses(out, cfg)
    except ValueError as error:
        assert "high_level_weights" in str(error)
    else:
        raise AssertionError("mismatched level weights must be rejected")
