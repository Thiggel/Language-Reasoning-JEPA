import pytest

from textjepa.data.planbench import (
    BlocksAction,
    action_catalogue,
    compile_blocksworld_episode,
    goal_reached,
    is_applicable,
    parse_blocksworld_pddl,
    shortest_plan,
    transition,
)


PDDL = """
(define (problem BW-test)
(:domain blocksworld-4ops)
(:objects a b c)
(:init (handempty) (ontable a) (ontable b) (ontable c)
       (clear a) (clear b) (clear c))
(:goal (and (on a b) (on b c)))
)
"""


def test_parse_solve_and_execute_official_four_operator_problem():
    problem = parse_blocksworld_pddl(PDDL)
    catalogue = action_catalogue(problem.objects)
    plan = shortest_plan(problem.initial, problem.goal, catalogue)
    assert plan is not None
    assert len(plan) == 4
    state = problem.initial
    for action in plan:
        assert is_applicable(state, action)
        state = transition(state, action)
    assert goal_reached(state, problem.goal)


def test_inapplicable_action_is_rejected_without_state_mutation():
    problem = parse_blocksworld_pddl(PDDL)
    action = BlocksAction("stack", "a", "b")
    assert not is_applicable(problem.initial, action)
    with pytest.raises(ValueError, match="inapplicable"):
        transition(problem.initial, action)


def test_compiled_episode_uses_full_catalogue_but_feasible_counterfactuals():
    problem = parse_blocksworld_pddl(PDDL)
    episode = compile_blocksworld_episode(
        problem, "train", teacher_horizon=4
    )
    expected_catalogue = len(problem.objects) * 2 + (
        len(problem.objects) * (len(problem.objects) - 1) * 2
    )
    assert len(episode.transitions[0].catalogue) == expected_catalogue
    assert len(episode.transitions[0].available) < expected_catalogue
    assert episode.transitions[-1].outcome
    assert all(
        alternative.action in transition.catalogue
        for transition in episode.transitions
        for alternative in transition.counterfactuals
    )
