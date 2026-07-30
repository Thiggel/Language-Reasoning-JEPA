import pytest
import torch

from textjepa.objectives.hierarchical_language import EMAShrunkMahalanobis
from textjepa.planning.hierarchical_language import (
    autoregressive_population_search,
    exact_endpoint_control,
    optimizer_curse_curve,
    prior_coordinate_cem,
    receding_horizon_step,
    score_token_candidates,
)


def identity_metric(dimension=2):
    metric = EMAShrunkMahalanobis(dimension, shrinkage=1, epsilon=1e-8)
    metric.covariance.copy_(torch.eye(dimension))
    return metric


def test_token_oracle_reweights_complete_spans_and_next_token_marginal():
    endpoints = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.1, 0.0]])
    tokens = torch.tensor([[4, 8], [5, 8], [4, 9]])
    result = score_token_candidates(
        endpoints, tokens, torch.zeros(3), torch.zeros(2),
        lambda value: value, identity_metric(),
        prior_weight=0, temperature=0.1, vocab_size=10,
    )
    assert result.best_index == 0
    assert result.next_token_probability.sum().item() == pytest.approx(1)
    assert result.next_token_probability[4] > result.next_token_probability[5]


def test_exact_endpoint_control_separates_geometry_from_dynamics_failure():
    predicted = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
    exact = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    control = exact_endpoint_control(
        predicted, exact, torch.tensor([[1], [2]]), torch.zeros(2),
        torch.zeros(2), lambda value: value, identity_metric(),
        prior_weight=0, temperature=1, vocab_size=4,
    )
    assert control.predicted.best_index == 1
    assert control.exact.best_index == 0
    assert control.endpoint_cost_gap.tolist() == pytest.approx([9, -9])


def test_lm_log_probability_penalizes_unsupported_candidate():
    endpoints = torch.zeros(2, 2)
    result = score_token_candidates(
        endpoints, torch.tensor([[1], [2]]), torch.tensor([-1.0, -10.0]),
        torch.zeros(2), lambda value: value, identity_metric(),
        prior_weight=0.5, temperature=1, vocab_size=3,
    )
    assert result.best_index == 0


def test_autoregressive_population_search_preserves_elite_prefixes():
    calls = []

    def sample(prefixes, population):
        calls.append(None if prefixes is None else prefixes.clone())
        tokens = torch.arange(population)[:, None].repeat(1, 3) % 5
        if prefixes is not None:
            chosen = prefixes[torch.arange(population) % len(prefixes)]
            tokens[:, :chosen.shape[1]] = chosen
        return tokens, torch.zeros(population)

    result = autoregressive_population_search(
        sample, lambda tokens, logp: tokens.float().sum(-1),
        population=6, iterations=2, elite_fraction=0.5, preserve_prefix=2,
    )
    assert calls[0] is None
    assert calls[1].shape == (3, 2)
    assert result.tokens.shape == (3,)


def test_prior_coordinate_cem_optimizes_supported_transition():
    def prior(state, context, task):
        return torch.zeros(len(context), 1), torch.zeros(len(context), 1)

    def advance(state, context, action):
        successor = state + action
        return successor, successor

    def objective(states, contexts, logp):
        return (states[:, -1, 0] - 1.0).square() - 0.01 * logp.sum(-1)

    generator = torch.Generator().manual_seed(2)
    result = prior_coordinate_cem(
        advance, prior, objective, torch.zeros(1), torch.zeros(1),
        torch.zeros(1), horizon=1, action_dim=1, population=256,
        iterations=4, generator=generator,
        prior_flops_per_candidate_step=2,
        advance_flops_per_candidate_step=3,
        objective_flops_per_candidate=1,
    )
    assert result.states[-1, 0].item() == pytest.approx(1, abs=0.2)
    assert len(result.diagnostics) == 4
    assert all("prior_noise_norm" in row for row in result.diagnostics)
    assert all("iteration_seconds" in row for row in result.diagnostics)
    assert all(row["estimated_flops"] > 0 for row in result.diagnostics)


def test_grounded_cem_updates_elites_using_grounded_selection_cost():
    means = []

    def prior(state, context, task):
        return torch.zeros(len(state), 1), torch.zeros(len(state), 1)

    def advance(state, context, action):
        return state + action, context + action

    def objective(states, contexts, logp):
        # Predicted model prefers negative actions.
        return states[:, -1, 0]

    def ground(actions, states):
        # Grounded execution prefers positive actions.
        return {"achieved_cost": -actions[:, -1, 0]}

    result = prior_coordinate_cem(
        advance, prior, objective, torch.zeros(1), torch.zeros(1),
        torch.zeros(1), horizon=1, action_dim=1, population=64,
        iterations=3, elite_fraction=0.25, ground=ground,
        ground_topn=32, ground_random=32, select_grounded=True,
        generator=torch.Generator().manual_seed(7),
    )
    means.append(result.noise.mean())
    assert means[-1] > 0


def test_optimizer_curse_curve_reports_selected_and_best_exact():
    curve = optimizer_curse_curve(
        torch.tensor([2.0, 1.0, -10.0]),
        torch.tensor([2.0, 1.0, 20.0]),
        [1, 2, 3],
    )
    assert curve[-1]["best_predicted"] == -10
    assert curve[-1]["selected_exact"] == 20
    assert curve[-1]["best_exact"] == 1


def test_receding_horizon_discards_unexecuted_imagination():
    planned = torch.tensor([1, 2, 3, 4])

    def execute(prefix):
        return prefix + 10

    def exact_encode(actual):
        return actual.float().sum()[None], actual.float().mean()[None]

    step = receding_horizon_step(
        planned, execute, exact_encode, n_exec=2
    )
    assert step.executed_text.tolist() == [11, 12]
    assert step.exact_token_state.item() == 23
    assert step.exact_sentence_state.item() == 11.5


def test_execution_interval_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        receding_horizon_step(
            torch.tensor([1]), lambda x: x,
            lambda x: (x.float(), x.float()), n_exec=0,
        )
