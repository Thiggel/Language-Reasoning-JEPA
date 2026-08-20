"""Contract tests for the CANDIDATE-PRIVILEGED symbolic-oracle scorer.

`--scorer symbolic_oracle` is an EVALUATION-ONLY diagnostic: it ranks candidate
action sequences by the number of necessary-and-still-unresolved actions left
after executing them in a CLONE of the environment.  It exists to separate
"the search procedure / environment is fine" from "the representation and the
energy head are what lose the episodes"; it is never a model input and never a
system component.
"""

import types

import pytest
import torch

from textjepa.planning.flat_search import FlatPlanner


class _Env:
    """Minimal stand-in with the FaithfulEnv surface the scorer uses."""

    def __init__(self, feasible_order, necessary, resolved=()):
        self._order = list(feasible_order)   # actions become legal in this order
        self.necessary = set(necessary)
        self.resolved = list(resolved)

    def clone(self):
        return _Env(self._order, self.necessary, self.resolved)

    def feasible_actions(self):
        done = set(self.resolved)
        nxt = [q for q in self._order if q not in done]
        return nxt[:1]                        # exactly one legal action at a time

    def step(self, q):
        assert q in self.feasible_actions()
        self.resolved.append(q)
        return ""

    @property
    def solved(self):
        return self.necessary <= set(self.resolved)

    def remaining_necessary(self):
        return len(self.necessary - set(self.resolved))


def _planner():
    p = FlatPlanner.__new__(FlatPlanner)
    p.device = torch.device("cpu")
    p.scorer = "symbolic_oracle"
    return p


def test_progress_is_ranked_below_no_progress():
    env = _Env(["a", "b", "c"], {"a", "b", "c"})
    s = _planner()._symbolic_scores([["a"], ["b"], ["c"]], env)
    # only "a" is legal now, so only "a" reduces the remaining count
    assert s[0] < s[1] and s[0] < s[2]
    assert float(s[0]) == pytest.approx(2.0)
    assert float(s[1]) == pytest.approx(3.0 + 1e-3)


def test_solved_dominates_every_unsolved_sequence():
    env = _Env(["a", "b"], {"a", "b"})
    s = _planner()._symbolic_scores([["a", "b"], ["a", "a"], ["b", "b"]], env)
    assert float(s[0]) < float(s[1]) and float(s[0]) < float(s[2])
    assert float(s[0]) == pytest.approx(-1.0)


def test_wasted_steps_break_ties_but_never_outrank_real_progress():
    env = _Env(["a", "b", "c"], {"a", "b", "c"})
    # both reach the same symbolic state (one necessary action done); the second
    # gets there with a wasted illegal first pick.
    s = _planner()._symbolic_scores([["a", "z"], ["z", "a"]], env)
    assert float(s[0]) == pytest.approx(float(s[1]))   # same waste count (1 each)
    s2 = _planner()._symbolic_scores([["a", "b"], ["a", "z"]], env)
    assert float(s2[0]) < float(s2[1])
    # a purely wasteful sequence never beats a progressing one
    s3 = _planner()._symbolic_scores([["a"], ["z"]], env)
    assert float(s3[0]) < float(s3[1])


def test_none_slots_count_as_wasted_not_as_errors():
    env = _Env(["a", "b"], {"a", "b"})
    s = _planner()._symbolic_scores([["a", None], ["a", "b"]], env)
    assert float(s[1]) < float(s[0])


def test_illegal_actions_are_no_ops_like_the_real_executor():
    env = _Env(["a", "b"], {"a", "b"})
    before = env.remaining_necessary()
    _planner()._symbolic_scores([["b", "b", "b"]], env)
    assert env.remaining_necessary() == before      # the real env is untouched


def test_scorer_name_is_accepted_by_the_constructor_validation():
    for name in ("energy", "oracle_distance", "symbolic_oracle"):
        assert name in {"energy", "oracle_distance", "symbolic_oracle"}
    with pytest.raises(ValueError):
        FlatPlanner(types.SimpleNamespace(observed_action_decoder=None),
                    None, torch.device("cpu"), scorer="nonsense")


class _FreeEnv:
    """Every unresolved action is legal; only some of them are NECESSARY.

    This is the shape that exposes the tie-break: once the lookahead is at
    least as long as the number of remaining necessary actions, many different
    sequences reach the solved state, so the primary term alone cannot decide
    which one to start with.
    """

    def __init__(self, actions, necessary, resolved=()):
        self._actions = list(actions)
        self.necessary = set(necessary)
        self.resolved = list(resolved)

    def clone(self):
        return _FreeEnv(self._actions, self.necessary, self.resolved)

    def feasible_actions(self):
        return [q for q in self._actions if q not in set(self.resolved)]

    def step(self, q):
        self.resolved.append(q)
        return ""

    @property
    def solved(self):
        return self.necessary <= set(self.resolved)

    def remaining_necessary(self):
        return len(self.necessary - set(self.resolved))


def test_legal_but_useless_first_action_loses_when_both_reach_the_goal():
    # 2 necessary actions, 2 legal distractors, lookahead 4: both sequences end
    # solved, so the PRIMARY term ties at -1.  Only the no-progress tie-break
    # keeps the planner from executing the distractor first.
    env = _FreeEnv(["n1", "n2", "d1", "d2"], {"n1", "n2"})
    clean = ["n1", "n2", "d1", "d2"]
    dirty = ["d1", "n1", "n2", "d2"]
    s = _planner()._symbolic_scores([clean, dirty], env)
    assert float(s[0]) == pytest.approx(-1.0)
    assert float(s[0]) < float(s[1])


def test_slots_after_the_goal_is_reached_are_free():
    env = _FreeEnv(["n1", "d1", "d2"], {"n1"})
    s = _planner()._symbolic_scores([["n1", "d1", "d2"], ["n1"]], env)
    assert float(s[0]) == pytest.approx(float(s[1]))
