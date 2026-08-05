import random
from collections import Counter

import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.models import DiscourseJEPA
from textjepa.planning import (
    HierarchicalLatentPlanner,
    LatentPlanner,
    evaluate_planning,
)
from textjepa.planning.search import _feasible, _sequences


def test_multistep_candidates_are_balanced_fixed_depth_and_absorbing():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=41)
    problem, _ = dataset.problem(0)
    # Put the query itself on the current feasible menu. Choosing it must
    # create an absorbing rollout rather than a shorter candidate.
    resolved = frozenset(problem.query_ancestors - {problem.query})
    roots = set(_feasible(problem, resolved))
    sequences = _sequences(
        problem, resolved, depth=4, cap=64,
        rng=__import__("random").Random(19),
    )

    assert {sequence[0] for sequence in sequences} == roots
    assert all(len(sequence) == 4 for sequence in sequences)
    counts = Counter(sequence[0] for sequence in sequences)
    assert max(counts.values()) - min(counts.values()) <= 1
    for sequence in sequences:
        if sequence[0] == problem.query:
            assert sequence[1:] == [None, None, None]
        if None in sequence:
            first = sequence.index(None)
            assert all(action is None for action in sequence[first:])


def test_multistep_candidate_sampling_and_order_are_seed_reproducible():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=43).problem(0)
    resolved = frozenset()
    import random

    first = _sequences(problem, resolved, 4, 64, random.Random(7))
    replay = _sequences(problem, resolved, 4, 64, random.Random(7))
    second = _sequences(problem, resolved, 4, 64, random.Random(8))
    assert first == replay
    assert first != second


def test_absorbing_candidates_receive_constant_horizon_offset(monkeypatch):
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=47).problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat",
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=3,
        allow_oracle_future_actions=True,
    )
    monkeypatch.setattr(
        planner, "_action_codes",
        lambda _problem, actions: torch.zeros(len(actions), 8),
    )
    monkeypatch.setattr(
        planner, "_energy",
        lambda _cur, _s0, steps, _goal: steps,
    )
    feasible = _feasible(problem, frozenset())
    costs = planner._flat_costs(
        torch.zeros(1, 64), torch.zeros(1, 64), problem,
        [[feasible[0], None, None], [feasible[0], feasible[0], feasible[0]]],
        None,
    )
    assert torch.equal(costs, torch.tensor([3.0, 3.0]))


def test_score_controls_are_deterministic_and_reject_unknown_values():
    zero = LatentPlanner(None, None, torch.device("cpu"), score_control="zero")
    assert zero._controlled_argmin(torch.tensor([4.0, 1.0, 2.0]), "x") == 0
    shuffled = LatentPlanner(
        None, None, torch.device("cpu"), score_control="shuffle"
    )
    first = shuffled._controlled_argmin(torch.tensor([4.0, 1.0, 2.0]), "x")
    assert first == shuffled._controlled_argmin(
        torch.tensor([4.0, 1.0, 2.0]), "x"
    )
    with pytest.raises(ValueError, match="unknown score control"):
        LatentPlanner(None, None, torch.device("cpu"), score_control="bad")
    with pytest.raises(ValueError, match="unknown search algorithm"):
        LatentPlanner(None, None, torch.device("cpu"), search_algorithm="bad")


def test_transition_advantage_energy_is_summed_across_rollout():
    from types import SimpleNamespace
    from torch import nn

    class Predictor(nn.Module):
        def forward(self, state, action):
            return state + action

    class Energy(nn.Module):
        def forward(self, state, successor, initial):
            return successor[..., 0]

    model = SimpleNamespace(
        predictor=Predictor(), geo_rank_score_mode="transition",
        geo_energy_target="advantage",
        core=SimpleNamespace(
            macro_k=0, d_action=1, transition_energy_head=Energy()
        ),
    )
    planner = LatentPlanner(
        model, None, torch.device("cpu"), lookahead=2,
        allow_oracle_future_actions=True,
        transition_energy_composition="cumulative",
    )
    planner._action_codes = lambda _problem, actions: torch.tensor(
        actions, dtype=torch.float32
    ).unsqueeze(-1)
    costs = planner._flat_costs(
        torch.zeros(1, 1), torch.zeros(1, 1), None, [[1, 2]], None
    )
    # Predicted states are 1 then 3, so cumulative Energy is 1 + 3.
    torch.testing.assert_close(costs, torch.tensor([4.0]))


def test_transition_energy_can_score_only_terminal_or_root_edge():
    from types import SimpleNamespace
    from torch import nn

    class Predictor(nn.Module):
        def forward(self, state, action):
            return state + action

    class Energy(nn.Module):
        def forward(self, state, successor, initial):
            return successor[..., 0]

    model = SimpleNamespace(
        predictor=Predictor(), geo_rank_score_mode="transition",
        geo_energy_target="advantage",
        core=SimpleNamespace(
            macro_k=0, d_action=1, transition_energy_head=Energy()
        ),
    )
    common = dict(
        model=model, vocab=None, device=torch.device("cpu"), lookahead=2,
        allow_oracle_future_actions=True,
    )
    terminal = LatentPlanner(
        **common, transition_energy_composition="terminal"
    )
    root = LatentPlanner(**common, transition_energy_composition="root")
    for planner in (terminal, root):
        planner._action_codes = lambda _problem, actions: torch.tensor(
            actions, dtype=torch.float32
        ).unsqueeze(-1)
    args = (torch.zeros(1, 1), torch.zeros(1, 1), None, [[1, 2]], None)
    torch.testing.assert_close(terminal._flat_costs(*args), torch.tensor([3.0]))
    torch.testing.assert_close(root._flat_costs(*args), torch.tensor([1.0]))


def test_unknown_transition_energy_composition_is_rejected():
    with pytest.raises(ValueError, match="unknown transition Energy composition"):
        LatentPlanner(
            None, None, torch.device("cpu"),
            transition_energy_composition="bad",
        )


def test_genuine_beam_returns_root_of_best_complete_sequence(monkeypatch):
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=121).problem(0)
    roots = _feasible(problem, frozenset())
    assert len(roots) >= 2
    planner = LatentPlanner(
        None, vocab, torch.device("cpu"), lookahead=2, max_expand=len(roots),
        allow_oracle_future_actions=True, search_algorithm="beam",
    )

    def costs(_s, _s0, _problem, sequences, *_args):
        values = []
        for sequence in sequences:
            # Root 0 is initially attractive. A continuation under root 1 is
            # the best complete beam, so receding-horizon choice must use 1.
            if len(sequence) == 1:
                values.append(float(0 if sequence[0] == roots[0] else 1))
            else:
                values.append(float(-10 if sequence[0] == roots[1] else 0))
        return torch.tensor(values)

    monkeypatch.setattr(planner, "_flat_costs", costs)
    best = planner._beam_search(
        torch.zeros(1, 1), torch.zeros(1, 1), problem, frozenset(), None,
        torch.zeros(1, 1, 1), torch.zeros(1, 0, 1), "seed",
    )
    assert best[0] == roots[1]


def test_symbolic_direct_control_uses_direct_head_end_to_end():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=53).problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        geo_rank_score_mode="direct",
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2,
        max_expand=8, simulator="symbolic",
        allow_oracle_future_actions=True,
    )
    result = planner.plan_episode(problem, slack=0, seed=17)
    assert isinstance(result.solved, bool)
    assert result.steps == problem.n_necessary_steps


def test_symbolic_transition_control_uses_exact_transition_head():
    from torch import nn

    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=57).problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat", geo_rank_score_mode="transition",
    ).eval()

    class ExactPairEnergy(nn.Module):
        def forward(self, predecessor, successor, initial):
            return predecessor[..., 0] + 2 * successor[..., 0]

    model.core.transition_energy_head = ExactPairEnergy()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2,
        simulator="symbolic", allow_oracle_future_actions=True,
    )
    env = SymbolicEnv(problem)
    prompt = prompt_sentences(problem, random.Random(3))
    prompt_tokens = planner._tokens(prompt)
    prompt_mask = torch.ones(1, len(prompt), dtype=torch.bool)
    current = planner._current_state(prompt_tokens, prompt_mask, [])
    initial = planner._s0(prompt_tokens, prompt_mask)
    first = env.feasible_actions()[0]
    clone = env.clone()
    clone.step(first)
    second = clone.feasible_actions()[0]
    sequence = [[first, second]]
    observed = planner._symbolic_costs(
        problem, env, [], current, initial, prompt_tokens, prompt_mask,
        sequence, None,
    )
    texts = []
    exact_env = env.clone()
    texts.append(exact_env.step(first))
    predecessor = planner._encode_steps(prompt_tokens, prompt_mask, texts)
    texts.append(exact_env.step(second))
    successor = planner._encode_steps(prompt_tokens, prompt_mask, texts)
    expected = predecessor[:, 0] + 2 * successor[:, 0]
    torch.testing.assert_close(observed, expected)


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


def test_symbolic_distance_is_an_explicit_exact_cost_to_go_control():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=61).problem(0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
    ).eval()
    with pytest.raises(ValueError, match="requires simulator=symbolic"):
        LatentPlanner(model, vocab, torch.device("cpu"), energy="symbolic_distance")
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        simulator="symbolic", energy="symbolic_distance",
    )
    result = planner.plan_episode(problem, slack=0, seed=23)
    assert result.solved
    assert result.n_distractor == 0


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


def test_faithful_multistep_candidates_are_balanced_and_absorbing():
    import random
    from textjepa.data.faithful import (
        FaithfulDataset, FaithfulEnv, cached_faithful_vocab,
    )
    from textjepa.planning.faithful_search import FaithfulPlanner

    vocab = cached_faithful_vocab()
    problem, _ = FaithfulDataset(
        vocab, size=1, seed=59, max_op=8, max_edge=10,
        op_range=(3, 5), distractor_prob=0.2, max_distractors=1,
    ).problem(0)
    environment = FaithfulEnv(problem)
    while problem.query not in environment.feasible_actions():
        necessary = [
            action for action in environment.feasible_actions()
            if action in problem.necessary
        ]
        environment.step(necessary[0])
    roots = set(environment.feasible_actions())
    planner = FaithfulPlanner(
        None, None, torch.device("cpu"), lookahead=4, max_expand=64,
        allow_oracle_future_actions=True,
    )
    sequences = planner._sequences(environment, random.Random(23))
    counts = Counter(sequence[0] for sequence in sequences)
    assert set(counts) == roots
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(len(sequence) == 4 for sequence in sequences)
    for sequence in sequences:
        if sequence[0] == problem.query:
            assert sequence[1:] == [None, None, None]
