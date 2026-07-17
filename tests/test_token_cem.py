import torch
import torch.nn.functional as F
import pytest

from textjepa.planning.token_cem import (
    batched_categorical_min_cost,
    categorical_cem,
    conditional_bank,
    continuous_cem,
    gaussian_mixture_nll,
    project_to_bank,
)


def test_projection_returns_exact_bank_members_and_raw_distance():
    bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    raw = torch.tensor([[[1.9, 0.1], [0.0, 2.5]]])
    projected, distance = project_to_bank(raw, bank)
    assert torch.equal(projected, torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]))
    assert torch.allclose(distance, torch.tensor([[0.02, 0.25]]), atol=1e-6)


def test_conditional_bank_uses_nearest_start_states_only():
    states = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    actions = torch.tensor([[1.0], [2.0], [9.0]])
    selected = conditional_bank(torch.tensor([0.8, 0.0]), states, actions, 2)
    assert set(selected.squeeze(-1).tolist()) == {1.0, 2.0}


def test_categorical_cem_finds_discrete_sequence_without_lm():
    horizon, vocab = 3, 4
    secret = torch.tensor([3, 1, 2])
    goal = F.one_hot(secret, vocab).float().flatten()

    def rollout(tokens):
        encoded = F.one_hot(tokens, vocab).float().flatten(1)
        return encoded[:, None].expand(-1, horizon, -1)

    result = categorical_cem(
        rollout, goal, horizon, vocab, candidates=512, iterations=5,
        elites=32, generator=torch.Generator().manual_seed(4),
    )
    assert torch.equal(result.actions, secret)
    assert result.cost == 0.0


def test_categorical_cem_extra_cost_can_override_goal_only_choice():
    goal = torch.tensor([1.0, -1.0])

    def rollout(tokens):
        final = torch.stack([tokens[:, 0].float(), -tokens[:, 0].float()], -1)
        return final[:, None]

    plain = categorical_cem(
        rollout, goal, 1, 3, candidates=512, iterations=4, elites=32,
        generator=torch.Generator().manual_seed(31),
    )
    penalized = categorical_cem(
        rollout, goal, 1, 3, candidates=512, iterations=4, elites=32,
        extra_cost=lambda tokens, states: (
            10.0 * tokens[:, 0].eq(1).float(),
            {"support_penalty": tokens[:, 0].eq(1).float()},
        ),
        generator=torch.Generator().manual_seed(31),
    )
    assert plain.actions.item() == 1
    assert penalized.actions.item() != 1
    assert "support_penalty" in penalized.diagnostics


def test_categorical_cem_scores_in_transformed_goal_coordinates():
    goal = torch.tensor([1.0, -1.0])

    def rollout(tokens):
        state = torch.stack([tokens[:, 0].float(), -tokens[:, 0].float()], -1)
        return state[:, None]

    # Negating the rollout coordinates reverses which token is closest.
    plain = categorical_cem(
        rollout, goal, 1, 3, candidates=256, iterations=3, elites=16,
        generator=torch.Generator().manual_seed(22),
    )
    transformed = categorical_cem(
        rollout, goal, 1, 3, candidates=256, iterations=3, elites=16,
        goal_states=lambda states: -states,
        generator=torch.Generator().manual_seed(22),
    )
    assert plain.actions.item() != transformed.actions.item()


def test_categorical_cem_can_use_learned_geometric_cost():
    goal = torch.tensor([1.0, -1.0])

    def rollout(tokens):
        state = torch.stack([tokens[:, 0].float(), -tokens[:, 0].float()], -1)
        return state[:, None]

    plain = categorical_cem(
        rollout, goal, 1, 3, candidates=256, iterations=3, elites=16,
        generator=torch.Generator().manual_seed(42),
    )
    learned = categorical_cem(
        rollout, goal, 1, 3, candidates=256, iterations=3, elites=16,
        goal_cost_fn=lambda states, goals: -states[:, 0],
        generator=torch.Generator().manual_seed(42),
    )
    assert plain.actions.item() == 1
    assert learned.actions.item() == 2


def test_gmm_nll_prefers_component_centres():
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    cov = torch.eye(2).repeat(2, 1, 1) * 0.1
    at_mode = torch.tensor([[[0.0, 0.0]]])
    between = torch.tensor([[[2.5, 2.5]]])
    assert gaussian_mixture_nll(at_mode, weights, means, cov) < gaussian_mixture_nll(
        between, weights, means, cov
    )


def test_reachability_refinement_changes_selected_subgoal():
    goal = torch.tensor([1.0, -1.0])

    def rollout(actions):
        # Both coordinates are directly optimizer-controlled.
        return actions

    def unreachable_positive_first(subgoals):
        return 20.0 * subgoals[:, 0].relu()

    plain = continuous_cem(
        rollout, goal, 1, 2, candidates=512, iterations=4, elites=32,
        generator=torch.Generator().manual_seed(3),
    )
    refined = continuous_cem(
        rollout, goal, 1, 2, candidates=512, iterations=4, elites=32,
        reachability=unreachable_positive_first, reach_topn=64,
        reach_weight=1.0, generator=torch.Generator().manual_seed(3),
    )
    assert plain.states[0, 0] > 0.5
    assert refined.states[0, 0] <= 0.05
    assert "reachability" in refined.diagnostics


def test_cem_can_optimize_base_noise_through_action_transform():
    goal = torch.tensor([1.0, -1.0])

    def conditional_prior_rollout(noise):
        codes = torch.stack([2.0 * noise[..., 0], noise[..., 1] - 1.0], -1)
        return codes, codes

    result = continuous_cem(
        conditional_prior_rollout, goal, 1, 2, candidates=512,
        iterations=5, elites=32,
        generator=torch.Generator().manual_seed(8),
    )
    # Returned actions must be decoded codes, not the optimized base noise.
    returned_cost = (
        F.layer_norm(result.actions[0], result.actions[0].shape)
        - F.layer_norm(goal, goal.shape)
    ).abs().mean()
    assert torch.allclose(returned_cost, torch.tensor(result.cost), atol=1e-6)


def test_continuous_cem_scores_in_parent_state_coordinates():
    goal = torch.tensor([1.0, -1.0])
    plain = continuous_cem(
        lambda actions: actions, goal, 1, 2, candidates=512,
        iterations=4, elites=32,
        generator=torch.Generator().manual_seed(18),
    )
    lifted = continuous_cem(
        lambda actions: actions, goal, 1, 2, candidates=512,
        iterations=4, elites=32, goal_states=lambda states: -states,
        generator=torch.Generator().manual_seed(18),
    )
    assert torch.sign(plain.states[0, 0]) != torch.sign(lifted.states[0, 0])


def test_continuous_cem_can_use_learned_geometric_cost():
    goal = torch.tensor([1.0, -1.0])
    learned = continuous_cem(
        lambda actions: actions, goal, 1, 2, candidates=512,
        iterations=4, elites=32,
        goal_cost_fn=lambda states, goals: states[:, 0].square(),
        generator=torch.Generator().manual_seed(21),
    )
    assert abs(float(learned.states[0, 0])) < 0.1


def test_reachability_retains_enough_candidates_for_all_elites():
    goal = torch.tensor([1.0, -1.0])
    result = continuous_cem(
        lambda actions: actions, goal, 1, 2, candidates=64, iterations=2,
        elites=16, reachability=lambda states: states.square().mean(-1),
        reach_topn=2, reach_weight=1.0,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.isfinite(torch.tensor(result.cost))


def test_cem_rejects_empty_search_or_support_domains():
    with pytest.raises(ValueError):
        project_to_bank(torch.zeros(1, 1, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError):
        categorical_cem(
            lambda x: torch.zeros(len(x), 1, 2), torch.zeros(2), 1, 2,
            forbidden=(0, 1),
        )
    with pytest.raises(ValueError):
        continuous_cem(lambda x: x, torch.zeros(2), 0, 2)


def test_batched_reachability_optimizes_distinct_subgoals_together():
    vocab, horizon = 4, 2
    secrets = torch.tensor([[1, 2], [3, 1]])
    goals = F.one_hot(secrets, vocab).float().flatten(1)

    def rollout(tokens):
        state = F.one_hot(tokens, vocab).float().flatten(1)
        return state[:, None].expand(-1, horizon, -1)

    costs = batched_categorical_min_cost(
        rollout, goals, horizon, vocab, candidates=256, iterations=4,
        elites=16, generator=torch.Generator().manual_seed(19),
    )
    assert torch.equal(costs, torch.zeros_like(costs))


def test_batched_reachability_can_lift_rollouts_before_scoring():
    goals = torch.tensor([[-1.0, 1.0]])

    def rollout(tokens):
        state = torch.stack([tokens[:, 0].float(), -tokens[:, 0].float()], -1)
        return state[:, None]

    cost = batched_categorical_min_cost(
        rollout, goals, 1, 3, candidates=256, iterations=3, elites=16,
        goal_states=lambda states: -states,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.allclose(cost, torch.zeros_like(cost))
