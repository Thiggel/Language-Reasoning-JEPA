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


def test_compiler_can_observe_every_invalid_catalogue_consequence():
    problem = parse_blocksworld_pddl(PDDL)
    episode = compile_blocksworld_episode(
        problem, "train", teacher_horizon=4,
        counterfactual_k=None, invalid_counterfactual_k=-1,
    )
    for transition_value in episode.transitions:
        alternatives = {
            item.action: item for item in transition_value.counterfactuals
        }
        assert set(alternatives) == set(transition_value.catalogue) - {
            transition_value.action
        }
        for action in (
            set(transition_value.catalogue) - set(transition_value.available)
        ):
            assert "state is unchanged" in alternatives[action].outcome


def test_canonical_identity_is_invariant_to_block_renaming():
    """Split disjointness must hold modulo which letters name the blocks.

    A relabelled copy of a test problem poses the identical reasoning task,
    so it would be leakage if it appeared in training.
    """
    from textjepa.data.planbench import canonical_identity

    renamed = PDDL.replace("(:objects a b c)", "(:objects x y z)")
    for source, target in (("a", "x"), ("b", "y"), ("c", "z")):
        renamed = renamed.replace(f" {source})", f" {target})")
        renamed = renamed.replace(f" {source} ", f" {target} ")
    original = parse_blocksworld_pddl(PDDL)
    permuted = parse_blocksworld_pddl(renamed)
    assert permuted.objects != original.objects
    assert canonical_identity(permuted) == canonical_identity(original)


def test_generated_instances_are_solvable_and_non_trivial():
    import random

    from textjepa.data.planbench import random_blocksworld_problem

    rng = random.Random(7)
    for index in range(20):
        problem = random_blocksworld_problem(rng, 4, f"gen-{index}")
        assert not problem.goal.issubset(problem.initial)
        plan = shortest_plan(
            problem.initial, problem.goal, action_catalogue(problem.objects)
        )
        assert plan and len(plan) >= 1


def test_compiled_counterfactuals_record_teacher_rollout_actions():
    problem = parse_blocksworld_pddl(PDDL)
    episode = compile_blocksworld_episode(
        problem, "train", teacher_horizon=4, counterfactual_k=-1,
        invalid_counterfactual_k=2,
    )
    alternatives = [
        alternative for step in episode.transitions
        for alternative in step.counterfactuals
    ]
    assert alternatives
    for alternative in alternatives:
        assert len(alternative.teacher_rollout_actions) == len(
            alternative.teacher_rollouts
        )
        for states, actions in zip(
            alternative.teacher_rollouts, alternative.teacher_rollout_actions
        ):
            assert len(states) == len(actions)
