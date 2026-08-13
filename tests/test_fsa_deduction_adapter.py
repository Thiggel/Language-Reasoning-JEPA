from dataclasses import asdict

import pytest

from textjepa.data.fsa_deduction import (
    canonical_identity,
    compile_fsa_episode,
    expert_derivation,
    sample_fsa_problem,
)
from textjepa.data.observed_action import ObservedActionEpisode
from textjepa.planning.catalogue import environment_from_episode


def _problem(depth: int = 6, index: int = 0, seed: int = 41):
    return sample_fsa_problem(seed=seed, index=index, depth=depth)


def test_depth_knob_controls_the_number_of_inference_steps():
    for depth in (1, 5, 17, 33):
        problem = _problem(depth=depth)
        assert len(expert_derivation(problem)) == 2 * depth - 1
        assert len(problem.rules) == 2 * problem.branching_factor * depth


def test_exactly_one_rule_is_applicable_at_every_point():
    episode = compile_fsa_episode(_problem(depth=9), "train")
    assert all(len(step.available) == 1 for step in episode.transitions)
    assert all(
        step.action in step.catalogue for step in episode.transitions
    )


def test_generation_is_deterministic_and_identity_is_content_based():
    first, second = _problem(), _problem()
    assert canonical_identity(first) == canonical_identity(second)
    assert canonical_identity(_problem(index=1)) != canonical_identity(first)


def test_compiled_episode_validates_and_carries_rollout_actions():
    episode = compile_fsa_episode(_problem(depth=8), "val")
    ObservedActionEpisode.from_dict(asdict(episode))
    counterfactuals = [
        value for step in episode.transitions for value in step.counterfactuals
    ]
    assert counterfactuals
    for value in counterfactuals:
        assert value.teacher_rollout_actions
        assert len(value.teacher_rollout_actions) == len(value.teacher_rollouts)
        assert all(
            len(states) == len(actions)
            for states, actions in zip(
                value.teacher_rollouts, value.teacher_rollout_actions
            )
        )


def test_executor_replays_the_expert_trajectory_to_the_goal():
    episode = compile_fsa_episode(_problem(depth=11), "test")
    environment = environment_from_episode(episode)
    for step in episode.transitions:
        assert step.action in environment.catalogue
        assert environment.step(step.action) == step.outcome
    assert environment.solved
    assert environment.invalid_actions == 0


def test_wrong_branch_is_rejected_without_state_progress():
    episode = compile_fsa_episode(_problem(depth=7), "test")
    environment = environment_from_episode(episode)
    wrong = next(
        action for action in episode.transitions[0].catalogue
        if action != episode.transitions[0].action
    )
    assert environment.step(wrong) == (
        "The proposed inference is invalid and no fact is added ."
    )
    assert environment.invalid_actions == 1
    assert environment.step(episode.transitions[0].action) == (
        episode.transitions[0].outcome
    )


def test_branching_factor_must_be_at_least_two():
    with pytest.raises(ValueError):
        sample_fsa_problem(seed=1, index=0, depth=3, branching_factor=1)
