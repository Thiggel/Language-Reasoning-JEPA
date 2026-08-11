"""Slack-curve planner evaluation on the observed-action domains."""

import torch

from textjepa.data.observed_action import (
    ObservedActionDataset,
    build_observed_action_vocab,
)
from textjepa.data.planbench import (
    compile_blocksworld_episode,
    parse_blocksworld_pddl,
)
from textjepa.models import DiscourseJEPA
from textjepa.planning.evaluate import _aggregate
from textjepa.planning.observed_action_search import (
    evaluate_observed_action_planning,
)
from textjepa.planning.search import EpisodeResult

from tests.test_planbench_adapter import PDDL

SCALAR_KEYS = set(_aggregate([EpisodeResult(True, 1, 1, 0)]))
CURVE_KEYS = {"success_by_slack", "excess_steps"}


def _dataset():
    episode = compile_blocksworld_episode(parse_blocksworld_pddl(PDDL), "test")
    vocab = build_observed_action_vocab([episode])
    return ObservedActionDataset([episode], vocab), vocab


def _model(vocab):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        max_chunk_len=64, max_chunks=32,
    ).eval()


def _check(results, slack):
    for name, metrics in results.items():
        assert set(metrics) == SCALAR_KEYS | CURVE_KEYS, name
        curve = metrics["success_by_slack"]
        assert list(curve) == [str(value) for value in range(slack + 1)]
        values = list(curve.values())
        assert values == sorted(values), f"{name} slack curve is not monotone"
        assert metrics["success"] == curve[str(slack)]
        assert len(metrics["excess_steps"]) == 1
        for excess in metrics["excess_steps"]:
            assert excess is None or 0 <= excess <= slack


def test_feasible_menu_slack_curve_matches_shared_schema():
    dataset, vocab = _dataset()
    results = evaluate_observed_action_planning(
        _model(vocab), dataset, vocab, torch.device("cpu"),
        n_episodes=1, slack=3, slack_curve=True,
        candidate_interface="feasible_menu",
    )
    assert set(results) == {
        "latent_planner", "random_policy", "first_feasible_policy", "oracle",
    }
    _check(results, slack=3)
    # Privileged expert replay must solve within the reference plan length.
    assert results["oracle"]["success_by_slack"]["0"] == 1.0
    assert results["oracle"]["invalid_action_rate"] == 0.0


def test_full_catalogue_slack_curve_uses_noop_invalid_semantics():
    dataset, vocab = _dataset()
    results = evaluate_observed_action_planning(
        _model(vocab), dataset, vocab, torch.device("cpu"),
        n_episodes=1, slack=2, slack_curve=True,
        candidate_interface="full_catalogue",
    )
    _check(results, slack=2)
    # A full catalogue exposes infeasible proposals, which are executed as
    # no-ops and counted, never crashing the episode.
    assert results["random_policy"]["invalid_action_rate"] >= 0.0


def test_scalar_only_schema_without_slack_curve():
    dataset, vocab = _dataset()
    results = evaluate_observed_action_planning(
        _model(vocab), dataset, vocab, torch.device("cpu"),
        n_episodes=1, slack=1, slack_curve=False,
    )
    assert set(results["latent_planner"]) == SCALAR_KEYS


def test_faithful_single_pass_slack_curve_matches_fixed_slack_evaluations():
    """One generous run, scored at every slack, equals separate fixed runs."""
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
    from textjepa.planning.faithful_search import (
        FaithfulPlanner,
        evaluate_faithful_planning,
    )

    vocab = cached_faithful_vocab()
    dataset = FaithfulDataset(
        vocab, size=3, seed=917, max_op=15, max_edge=20, op_range=(3, 5),
        distractor_prob=0.0,
    )
    planner = FaithfulPlanner(_model(vocab), vocab, torch.device("cpu"))
    curve = evaluate_faithful_planning(
        planner, dataset, 3, slack=3, seed=5, slack_curve=True,
    )["latent_planner"]
    assert set(curve) == SCALAR_KEYS | CURVE_KEYS
    for slack in range(4):
        fixed = evaluate_faithful_planning(
            planner, dataset, 3, slack=slack, seed=5,
        )["latent_planner"]
        assert curve["success_by_slack"][str(slack)] == fixed["success"]
    values = list(curve["success_by_slack"].values())
    assert values == sorted(values)


def test_depth_two_full_catalogue_search_and_menu_depth_guard():
    import pytest

    from textjepa.planning.observed_action_search import ObservedActionPlanner

    dataset, vocab = _dataset()
    results = evaluate_observed_action_planning(
        _model(vocab), dataset, vocab, torch.device("cpu"),
        n_episodes=1, slack=1, slack_curve=True, lookahead=2, max_expand=8,
        candidate_interface="full_catalogue",
    )
    _check(results, slack=1)
    # Deeper search on the current feasible menu would require cloning the
    # external executor, which is impossible; it must be refused, not faked.
    with pytest.raises(ValueError, match="lookahead > 1 on the feasible menu"):
        ObservedActionPlanner(
            _model(vocab), vocab, torch.device("cpu"), lookahead=2,
            candidate_interface="feasible_menu",
        )
