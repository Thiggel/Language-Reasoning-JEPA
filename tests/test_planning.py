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


def test_multistep_search_requires_oracle_action_opt_in():
    with pytest.raises(ValueError, match="oracle-action diagnostic"):
        LatentPlanner(None, None, torch.device("cpu"), lookahead=2)
    planner = LatentPlanner(
        None, None, torch.device("cpu"), lookahead=2,
        allow_oracle_future_actions=True,
    )
    assert planner.allow_oracle_future_actions is True


def test_hierarchical_low_horizon_requires_oracle_action_opt_in():
    with pytest.raises(ValueError, match="hierarchy diagnostic"):
        HierarchicalLatentPlanner(
            None, None, torch.device("cpu"), low_horizon=3,
            low_action_source="oracle_feasible",
        )


def test_hierarchical_evaluation_reports_macro_usage():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=2, seed=0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), high_horizon=1,
        n_samples=4, flat_fallback_threshold=1e6,
    )
    metrics = evaluate_planning(planner, ds, n_episodes=2, slack=0)[
        "latent_planner"
    ]
    assert metrics["macro_decision_rate"] == 0.0
    assert metrics["macro_decisions"] == 0.0
    assert metrics["flat_decisions"] > 0.0


def test_low_level_cem_optimizes_then_projects_to_feasible_action():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=1, seed=31)
    problem, _ = ds.problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), low_method="cem",
        low_horizon=3, low_cem_samples=16, low_cem_iters=2,
        low_cem_elites=4,
    )
    from textjepa.data.igsm.env import SymbolicEnv

    env = SymbolicEnv(problem)
    state = torch.zeros(1, 64)
    chosen = planner._low_action(
        problem, env.feasible_actions(), state, state, state,
        frozenset(env.resolved_set),
    )
    assert chosen in env.feasible_actions()
    assert len(planner.low_cem_traces[-1]) == 2


def test_cem_trace_and_classical_update_run():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        d_model=64,
        chunk_layers=1,
        chunk_heads=2,
        state_layers=2,
        state_heads=2,
        d_action=8,
        d_macro=4,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        method="cem",
        high_horizon=2,
        n_samples=16,
        cem_iters=3,
        n_elites=4,
        mean_ema=0.5,
        variance_ema=0.5,
        scale_update="std",
        cem_return="best",
        density_weight=0.0,
        energy="oracle_goal",
    )
    start = torch.zeros(1, 64)
    codes = planner._cem_codes(start, start, torch.ones(1, 64))
    assert codes.shape == (1, 2, 4)
    trace = planner.cem_traces[-1]
    assert len(trace) == 3
    best = [row["best_so_far"] for row in trace]
    assert all(b <= a for a, b in zip(best, best[1:]))


def test_cem_can_optimize_conditional_prior_noise():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), method="cem", high_horizon=2,
        n_samples=8, cem_iters=2, n_elites=2, cem_domain="prior_noise",
        energy="oracle_goal",
    )
    start = torch.zeros(1, 64)
    codes = planner._cem_codes(start, start, torch.ones(1, 64))
    assert codes.shape == (1, 2, 4)
    assert torch.isfinite(codes).all()


def test_epistemic_ensemble_penalty_runs():
    vocab = build_vocab(23)
    kwargs = dict(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    )
    model = DiscourseJEPA(**kwargs).eval()
    ensemble = [DiscourseJEPA(**kwargs).eval() for _ in range(2)]
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), method="cem", high_horizon=1,
        n_samples=8, cem_iters=2, n_elites=2, energy="oracle_goal",
        ensemble_models=ensemble, epistemic_weight=1.0,
    )
    start = torch.zeros(1, 64)
    codes = planner._cem_codes(start, start, torch.ones(1, 64))
    assert codes.shape == (1, 1, 4)


def test_reachability_can_rerank_final_cem_candidates():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), method="cem", high_horizon=1,
        n_samples=8, cem_iters=2, n_elites=2, energy="oracle_goal",
        reachability_weight=1.0, reachability_mode="rerank",
        reachability_topk=3,
    )
    planner._low_reachable_states = torch.randn(4, 64)
    start = torch.zeros(1, 64)
    codes = planner._cem_codes(start, start, torch.ones(1, 64))
    assert codes.shape == (1, 1, 4)


def test_local_full_covariance_macro_gmm_is_finite():
    planner = object.__new__(HierarchicalLatentPlanner)
    planner.macro_gmm_components = 3
    planner.macro_gmm_ridge = 0.05
    codes = torch.randn(20, 8)
    planner._fit_macro_gmm(codes)
    nll = planner._macro_gmm_nll(codes)
    assert nll.shape == (20,)
    assert torch.isfinite(nll).all()


def test_discrete_sequence_cap_balances_executed_first_action():
    sequences = [
        [first, second]
        for first in range(3)
        for second in range(10)
    ]
    selected = HierarchicalLatentPlanner._balanced_first_action_cap(
        sequences, 8
    )
    counts = [sum(seq[0] == first for seq in selected) for first in range(3)]
    assert max(counts) - min(counts) <= 1
    assert set(seq[0] for seq in selected) == {0, 1, 2}


def test_discrete_hierarchy_runs_on_faithful_igsm():
    from textjepa.data.faithful import (
        FaithfulDataset,
        cached_faithful_vocab,
    )

    vocab = cached_faithful_vocab()
    dataset = FaithfulDataset(
        vocab, size=1, seed=19, max_op=8, max_edge=10,
        op_range=(3, 5), distractor_prob=0.0, max_distractors=0,
    )
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        d_action=8, d_macro=4, macro_k=3,
    ).eval()
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device("cpu"), energy="value",
        subgoal_source="discrete_all", high_horizon=1,
        low_max_expand=8, discrete_execute_macro=True,
    )
    results = evaluate_planning(planner, dataset, n_episodes=1, slack=0)
    assert results["oracle"]["success"] == 1.0
    assert 0.0 <= results["latent_planner"]["success"] <= 1.0
