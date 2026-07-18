import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.models import DiscourseJEPA
from textjepa.planning import (
    HierarchicalLatentPlanner,
    LatentPlanner,
    evaluate_planning,
)


def test_planner_runs_end_to_end():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=3, seed=0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    ).eval()
    planner = LatentPlanner(model, vocab, torch.device("cpu"), lookahead=1)
    results = evaluate_planning(planner, ds, n_episodes=3, slack=2)
    assert set(results) == {
        "latent_planner", "random_policy", "first_feasible_policy", "oracle"
    }
    assert results["oracle"]["success"] == 1.0
    for m in results.values():
        assert 0.0 <= m["success"] <= 1.0


def _distinct_model(vocab):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        predictor_heads=2, high_predictor_heads=2, macro_k=2,
        distinct_high_state_space=True, high_state_encoder_layers=1,
        dropout=0.0,
    ).eval()


def test_distinct_planner_lifts_complete_candidate_paths_with_ema_encoder():
    vocab = build_vocab(23)
    model = _distinct_model(vocab)
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), n_samples=2,
    )
    observed = torch.randn(1, 3, 64)
    future = torch.randn(2, 2, 64)
    valid = torch.tensor([[True, True], [True, False]])

    actual = planner._lift_low_candidate_paths(observed, future, valid)
    complete = torch.cat([observed.expand(2, -1, -1), future], dim=1)
    complete_valid = torch.cat([
        torch.ones(2, 3, dtype=torch.bool), valid
    ], dim=1)
    lifted = model.encode_high_state_path(
        complete, complete_valid, teacher=True
    )
    expected = lifted[torch.arange(2), torch.tensor([4, 3])]
    torch.testing.assert_close(actual, expected)

    isolated = model.encode_high_state_path(
        future, valid, teacher=True
    )[torch.arange(2), torch.tensor([1, 0])]
    assert not torch.allclose(actual, isolated)


def test_distinct_planner_rejects_cross_coordinate_oracle_goal():
    vocab = build_vocab(23)
    model = _distinct_model(vocab)
    with pytest.raises(ValueError, match="low-state diagnostic"):
        HierarchicalLatentPlanner(
            model, vocab, torch.device("cpu"), energy="oracle_goal"
        )


def test_distinct_hierarchical_planner_runs_coordinate_correct_path():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=0)
    planner = HierarchicalLatentPlanner(
        _distinct_model(vocab), vocab, torch.device("cpu"),
        n_samples=4, high_horizon=1, low_horizon=2,
        low_action_source="all_problem", low_max_expand=8,
    )
    problem, _ = dataset.problem(0)
    result = planner.plan_episode(problem, slack=0)
    assert result.steps <= result.n_necessary
