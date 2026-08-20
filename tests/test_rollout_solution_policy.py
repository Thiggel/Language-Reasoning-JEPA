"""Rollout policy switch: follow the reference solution instead of random.

Training rollouts used to step by sampling UNIFORMLY AT RANDOM among legal
actions, so at imagined depth >= 1 "the continuation that actually occurred"
carried no information about quality -- the energy head could only learn
legal-vs-illegal.  ``rollout_solution_prob`` makes a rollout step continue
along the environment's reference solution with the given probability, using
exactly the rule that already generates the factual trajectory.

The default (0.0) must be BIT-IDENTICAL to the historical behaviour, which
requires that no extra draw is taken from the item RNG.
"""

import json

import pytest

from textjepa.data.faithful import (
    FaithfulDataset, FaithfulEnv, cached_faithful_vocab,
)

KW = dict(
    size=12, seed=1, max_op=15, max_edge=20, op_range=(3, 15),
    distractor_prob=0.0, max_distractors=2, geo_rank_k=2, geo_rank_horizon=8,
    geo_rank_horizons=[1, 2, 4, 8], geo_rank_rollout_for_h1=True,
    geo_rank_rollouts=4, invalid_counterfactual_k=8,
    invalid_counterfactual_unresolved_only=True,
    invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
    all_action_supervision=True,
)


@pytest.fixture(scope="module")
def vocab():
    return cached_faithful_vocab(15, 20)


def _dump(item):
    return json.dumps(item, default=str, sort_keys=True)


def test_default_is_bit_identical(vocab):
    a = FaithfulDataset(vocab, **KW)
    b = FaithfulDataset(vocab, rollout_solution_prob=0.0, **KW)
    for i in range(len(a)):
        assert _dump(a[i]) == _dump(b[i])


def test_bad_probability_rejected(vocab):
    with pytest.raises(ValueError):
        FaithfulDataset(vocab, rollout_solution_prob=1.5, **KW)


def _necessary_fraction(ds, n):
    """Oracle MEASUREMENT ONLY: how many rollout steps are on the solution."""
    num = den = 0
    for i in range(n):
        fp, _ = ds.problem(i)
        item = ds[i]
        rollouts = item.get("ga_rollout_actions")
        if not rollouts:
            continue
        env = FaithfulEnv(fp)
        by_text = {
            tuple(ds.vocab.encode(env.action_text(q))): q
            for q in fp.action_order
        }
        for cand in rollouts:
            for rollout in cand:
                # index 0 is the ranking candidate itself, not a policy step
                for action in rollout[1:]:
                    q = by_text.get(tuple(action))
                    if q is None:
                        continue
                    den += 1
                    num += int(q in fp.necessary)
    assert den > 0
    return num / den


def test_prob_one_follows_the_solution(vocab):
    ds = FaithfulDataset(vocab, rollout_solution_prob=1.0, **KW)
    assert _necessary_fraction(ds, len(ds)) == 1.0


def test_probability_interpolates(vocab):
    lo = _necessary_fraction(FaithfulDataset(vocab, **KW), 12)
    mid = _necessary_fraction(
        FaithfulDataset(vocab, rollout_solution_prob=0.5, **KW), 12)
    assert lo < mid < 1.0


def test_rollout_actions_stay_in_the_catalogue(vocab):
    """The flat stream indexes rollout actions by phrase; they must resolve."""
    from textjepa.data.flat_stream import FlatIntentStreamDataset

    ds = FlatIntentStreamDataset(
        FaithfulDataset(vocab, rollout_solution_prob=1.0, **KW))
    for i in range(4):
        item = ds[i]
        if "ga_roll_act" not in item:
            continue
        for cand in item["ga_roll_act"]:
            for rollout in cand:
                assert all(isinstance(j, int) and j >= 0 for j in rollout)
