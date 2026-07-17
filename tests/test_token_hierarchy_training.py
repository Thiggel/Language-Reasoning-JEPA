from types import SimpleNamespace

import torch

from scripts.train_token_hierarchy_v2 import (
    compute_losses,
    geometric_preference_loss,
)
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
        geo_rank_low=0.0, geo_rank_high=0.0,
        geo_rank_level_weights=[1.0], geo_rank_k=2,
        geo_rank_horizon=2, geo_rank_continuations=2,
        geo_rank_macro_proposals="global", geo_rank_conditional_k=8,
        geo_rank_margin=0.5, geo_rank_label_gap=0.0,
        geo_rank_objective="pairwise", geo_rank_temperature=0.1,
        geo_rank_pairwise=1.0, geo_rank_regression=0.0,
        geo_rank_detach_prediction=False,
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


def test_end_to_end_geometry_ranking_updates_encoder_predictors_and_value_heads():
    torch.manual_seed(23)
    model = tiny_model().train()
    tokens = torch.randint(1, 30, (4, 24))
    prompt_len = torch.tensor([8, 8, 8, 8])
    batch = {"tokens": tokens, "prompt_len": prompt_len}
    out = model(tokens, prompt_len)
    cfg = SimpleNamespace(objective=objective(
        vicreg=0.0, geo_rank_low=1.0, geo_rank_high=1.0,
        geo_rank_horizon=2,
    ))
    total, items = compute_losses(out, cfg, model=model, batch=batch)
    assert torch.isfinite(total)
    assert "geo_low_pair" in items and "geo_level1_pair" in items
    total.backward()
    modules = (
        model.encoder, model.low_predictor, model.low_goal_value,
        model.levels[0].predictor, model.levels[0].goal_value,
    )
    for module in modules:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_advantage_mse_can_be_ablated_independently_of_pairwise_ranking():
    torch.manual_seed(29)
    model = tiny_model().train()
    tokens = torch.randint(1, 30, (4, 24))
    prompt_len = torch.tensor([8, 8, 8, 8])
    out = model(tokens, prompt_len)
    cfg = SimpleNamespace(objective=objective(
        vicreg=0.0, geo_rank_low=1.0, geo_rank_high=1.0,
        geo_rank_pairwise=0.0, geo_rank_regression=1.0,
    ))
    total, items = compute_losses(
        out, cfg, model=model,
        batch={"tokens": tokens, "prompt_len": prompt_len},
    )
    assert torch.isfinite(total)
    assert items["geo_low_regression"].item() > 0
    total.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.low_goal_value.parameters()
    )


def test_all_geometric_preference_objectives_prefer_correct_ordering():
    distance = torch.tensor([[0.1, 0.4, 0.9]])
    correct = torch.tensor([[0.1, 0.4, 0.9]])
    reversed_energy = correct.flip(1)
    for name in ("pairwise", "listwise", "regression"):
        good = geometric_preference_loss(
            correct, distance, name, 0.1, 0.0, 0.1
        )
        bad = geometric_preference_loss(
            reversed_energy, distance, name, 0.1, 0.0, 0.1
        )
        assert good < bad, name
