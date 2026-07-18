import random

import torch

from scripts.plan_faithful_token_edits import (
    buffer_distance,
    canonical_oracle_edit,
    pad_token_state_for_insertions,
    proposal_tokens,
    propose_edits,
    run_episode,
)


def test_deployable_proposals_use_only_declared_observed_token_source():
    prompt = [[9, 2]]
    current = [[1, 2], [3, 4]]
    strict = proposal_tokens(prompt, current, "current_buffer")
    extended = proposal_tokens(prompt, current, "prompt_plus_current")
    assert strict == [1, 2, 3, 4]
    assert extended == [9, 2, 1, 3, 4]

    edits = propose_edits(current, strict, 10_000, random.Random(0))
    assert all(action[2] in {None, 1, 2, 3, 4} for action in edits)
    # Final tokens of official steps are never deleted.
    assert ("delete", 1, None) not in edits
    assert ("delete", 3, None) not in edits


def test_oracle_edit_is_shortest_boundary_preserving_and_exposes_source_ceiling():
    current = [[1, 2], [3, 4]]
    target = [[1, 9, 2], [3, 4]]
    oracle = canonical_oracle_edit(current, target)
    assert oracle == ("insert", 1, 9)
    assert buffer_distance(current, target) == 1
    assert oracle[2] not in proposal_tokens([], current, "current_buffer")
    assert oracle[2] in proposal_tokens([[9]], current, "prompt_plus_current")


def test_unrepresentable_nonfinal_step_append_is_reported_without_crashing():
    current = [[1], [2]]
    target = [[1, 9], [2]]
    assert canonical_oracle_edit(current, target) is None

    result = run_episode(
        [[9]], current, target, "prompt_plus_current", "random", None,
        random.Random(0), max_candidates=16, max_steps=2,
    )
    assert not result.recovered
    assert result.oracle_unreachable
    assert result.decisions == 0


def test_oracle_injected_gar_receding_search_exactly_recovers():
    prompt = [[9]]
    initial = [[1, 2]]
    target = [[1, 9, 2]]

    def oracle_scorer(current, candidates):
        oracle = canonical_oracle_edit(current, target)
        return [float(action == oracle) for action in candidates]

    result = run_episode(
        prompt, initial, target, "current_buffer", "gar_greedy",
        oracle_scorer, random.Random(1), max_candidates=4, max_steps=3,
        inject_oracle=True,
    )
    assert result.recovered
    assert result.final_distance == 0
    assert result.selected_advantages == [1]
    assert result.source_ceiling_hits == 0
    assert result.oracle_injections == 1


def test_receding_search_detects_a_two_state_loop():
    initial = [[1]]
    target = [[9]]

    def toggling_scorer(current, candidates):
        desired = 2 if current == [[1]] else 1
        return [
            float(action == ("replace", 0, desired)) for action in candidates
        ]

    result = run_episode(
        [[1, 2]], initial, target, "prompt_plus_current", "gar_greedy",
        toggling_scorer, random.Random(3), max_candidates=32, max_steps=4,
    )
    assert result.looped
    assert not result.recovered
    assert result.invalid_actions == 0


def test_insert_scoring_reserves_one_more_token_state():
    states = torch.randn(2, 5, 7)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])
    padded, padded_mask = pad_token_state_for_insertions(states, mask)
    assert padded.shape == (2, 6, 7)
    assert padded_mask.shape == (2, 6)
    assert torch.equal(padded[:, :5], states)
    assert torch.equal(padded_mask[:, :5], mask)
    assert not padded_mask[:, -1].any()
