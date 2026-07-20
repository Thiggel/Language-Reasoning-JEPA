import math
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.models import DiscourseJEPA
from textjepa.planning import (
    HierarchicalLatentPlanner,
    LatentPlanner,
    evaluate_planning,
)
from textjepa.planning.search import validate_learned_catalogue_checkpoint


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


def test_action_prior_top_m_planning_and_missing_head_guard():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=2, seed=10)
    plain = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
    ).eval()
    with pytest.raises(ValueError, match="action-prior checkpoint"):
        LatentPlanner(
            plain, vocab, torch.device("cpu"), prior_top_m=2
        )

    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), prior_top_m=2
    )
    results = evaluate_planning(planner, ds, n_episodes=2, slack=1)
    metrics = results["latent_planner"]
    assert metrics["prior_decisions"] >= 2
    assert 0.0 <= metrics["prior_root_necessary_recall"] <= 1.0


def test_learned_catalogue_allows_nonoracle_multistep_planning():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()

    planner = LatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        lookahead=3,
        proposal_source="learned_catalogue",
        proposal_top_m=3,
        proposal_beam_width=2,
    )
    assert planner.lookahead == 3


def test_learned_catalogue_proposals_never_query_symbolic_feasibility():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=1, seed=17)
    problem, _ = ds.problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    planner = LatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        lookahead=3,
        proposal_source="learned_catalogue",
        proposal_top_m=3,
        proposal_beam_width=2,
    )
    state = torch.randn(1, 64)
    state_history = state.unsqueeze(1)
    action_history = torch.empty(1, 0, 8)

    with patch(
        "textjepa.data.igsm.env.SymbolicEnv.feasible_actions",
        side_effect=AssertionError("proposal generation queried feasibility"),
    ):
        sequences, costs = planner._learned_catalogue_sequences(
            state,
            problem,
            executed=[],
            state_history=state_history,
            action_history=action_history,
        )

    assert sequences
    assert len(sequences) == len(costs)
    assert all(len(sequence) == 3 for sequence in sequences)


def test_learned_catalogue_beam_is_root_balanced():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=1, seed=19)
    problem, _ = ds.problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    planner = LatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        lookahead=3,
        proposal_source="learned_catalogue",
        proposal_top_m=3,
        proposal_beam_width=2,
    )
    state = torch.randn(1, 64)
    sequences, _ = planner._learned_catalogue_sequences(
        state,
        problem,
        executed=[],
        state_history=state.unsqueeze(1),
        action_history=torch.empty(1, 0, 8),
    )

    roots = {sequence[0] for sequence in sequences}
    assert len(roots) == min(3, len(problem.vars))
    assert all(
        sum(sequence[0] == root for sequence in sequences) <= 2
        for root in roots
    )


def test_learned_catalogue_respects_global_expansion_cap():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    planner = LatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        lookahead=4,
        max_expand=6,
        proposal_source="learned_catalogue",
        proposal_top_m=3,
        proposal_beam_width=8,
    )
    ds = IGSMDataset(vocab, size=1, seed=23)
    problem, _ = ds.problem(0)
    state = torch.randn(1, 64)
    sequences, _ = planner._learned_catalogue_sequences(
        state,
        problem,
        executed=[],
        state_history=state.unsqueeze(1),
        action_history=torch.empty(1, 0, 8),
    )
    assert len(sequences) <= 6
    assert len({sequence[0] for sequence in sequences}) == 3


def test_learned_catalogue_passes_imagined_action_history_to_support_head():
    class RecordingSupport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.history_lengths = []

        def forward(self, state, action, history=None, history_mask=None):
            self.history_lengths.append(0 if history is None else history.shape[-2])
            return state.new_zeros(state.shape[:-1])

    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    recorder = RecordingSupport()
    model.core.action_support_head = recorder
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2,
        proposal_source="learned_catalogue", proposal_top_m=2,
    )
    problem, _ = IGSMDataset(vocab, size=1, seed=24).problem(0)
    state = torch.randn(1, 64)
    executed = [problem.vars[0].idx]
    history = planner._action_codes(problem, executed).unsqueeze(0)
    planner._episode_catalogue = [variable.idx for variable in problem.vars]
    planner._learned_catalogue_sequences(
        state, problem, executed=executed,
        state_history=torch.stack([state, state], dim=1),
        action_history=history,
    )
    assert recorder.history_lengths[0] == 1
    assert all(length == 2 for length in recorder.history_lengths[1:])


def test_learned_catalogue_invalid_execution_is_reported_as_failure():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=1, seed=29)
    problem, _ = ds.problem(0)
    infeasible = next(variable.idx for variable in problem.vars if variable.parents)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    planner = LatentPlanner(
        model,
        vocab,
        torch.device("cpu"),
        lookahead=2,
        proposal_source="learned_catalogue",
        proposal_top_m=2,
        prior_only=True,
    )
    with patch.object(
        planner,
        "_learned_catalogue_sequences",
        return_value=([[infeasible]], torch.tensor([0.0])),
    ):
        result = planner.plan_episode(problem)
    assert not result.solved
    assert result.steps == 1
    assert result.n_invalid == 1


def test_proposal_reranking_preserves_endpoints_and_normalizes_scales():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    latent = torch.tensor([0.0, 1.0, 2.0, 3.0])
    proposal = torch.tensor([300.0, 200.0, 100.0, 0.0])

    pure = LatentPlanner(
        model, vocab, torch.device("cpu"),
        proposal_source="learned_catalogue", proposal_top_m=4,
    )
    assert torch.equal(
        pure._combine_proposal_and_latent_costs(latent, proposal), latent
    )

    mixed = LatentPlanner(
        model, vocab, torch.device("cpu"),
        proposal_source="learned_catalogue", proposal_top_m=4,
        proposal_rerank_weight=2.0,
    )
    combined = mixed._combine_proposal_and_latent_costs(latent, proposal)
    assert combined.argmin().item() == 3
    assert torch.allclose(
        combined,
        mixed._combine_proposal_and_latent_costs(latent, proposal / 100.0),
    )


def test_proposal_reranking_rejects_invalid_protocol_combinations():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        action_prior=True,
    ).eval()
    with pytest.raises(ValueError, match="learned-catalogue"):
        LatentPlanner(
            model, vocab, torch.device("cpu"), proposal_rerank_weight=1.0
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        LatentPlanner(
            model, vocab, torch.device("cpu"),
            proposal_source="learned_catalogue", proposal_top_m=4,
            proposal_rerank_weight=1.0, prior_only=True,
        )
    with pytest.raises(ValueError, match="non-negative"):
        LatentPlanner(
            model, vocab, torch.device("cpu"),
            proposal_source="learned_catalogue", proposal_top_m=4,
            proposal_rerank_weight=-1.0,
        )


def test_learned_catalogue_checkpoint_gate_rejects_untrained_support():
    cfg = OmegaConf.create({
        "model": {
            "action_prior": True,
            "action_support_states": "all",
            "action_prior_states": "all",
            "action_prior_candidate_scope": "feasible",
        },
        "data": {"all_action_supervision": True},
        "objective": {
            "action_feasibility": {"weight": 0.0},
            "action_prior": {"weight": 1.0},
        },
    })
    with pytest.raises(ValueError, match="action_feasibility"):
        validate_learned_catalogue_checkpoint(cfg)
    cfg.objective.action_feasibility.weight = 1.0
    with pytest.raises(ValueError, match="action_prior_candidate_scope"):
        validate_learned_catalogue_checkpoint(cfg)
    cfg.model.action_prior_candidate_scope = "catalogue"
    validate_learned_catalogue_checkpoint(cfg)


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
    assert planner.n_macro_decisions == math.ceil(
        result.steps / planner.model.core.macro_k
    )
    assert planner._high_state_history.shape[1] == (
        planner._macro_action_history.shape[1] + 1
    )
