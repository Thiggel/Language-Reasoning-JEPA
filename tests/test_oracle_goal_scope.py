from types import SimpleNamespace

import torch

from scripts.plan_token_hierarchy_oracle_cem import OracleCEMPlanner


def planner(scope):
    value = SimpleNamespace(
        goal_score="learned_value", goal_score_scope=scope, value_weight=1.0
    )
    model = SimpleNamespace(levels=[object(), object(), object()])
    instance = OracleCEMPlanner.__new__(OracleCEMPlanner)
    instance.args = value
    instance.model = model
    return instance


def head(states, goals):
    return (states - goals).square().mean(-1)


def test_top_scope_enables_only_topmost_macro_value():
    instance = planner("top")
    assert instance.geometric_goal_cost(head, "low") is None
    assert instance.geometric_goal_cost(head, "macro", level_index=0) is None
    assert instance.geometric_goal_cost(head, "macro", level_index=1) is None
    score = instance.geometric_goal_cost(head, "macro", level_index=2)
    assert score is not None
    states = torch.tensor([[1.0, 0.0]])
    goals = torch.zeros_like(states)
    assert torch.equal(score(states, goals), head(states, goals))


def test_all_scope_retains_levelwise_value_behavior():
    instance = planner("all")
    assert instance.geometric_goal_cost(head, "low") is not None
    assert instance.geometric_goal_cost(head, "macro", level_index=0) is not None
