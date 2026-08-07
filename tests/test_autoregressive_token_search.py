import torch

from textjepa.planning.autoregressive_token_search import (
    TokenTrajectory,
    elite_prefix_cem,
    factorized_position_cem,
    first_order_markov_cem,
    jepa_beam_search,
)


def endpoint(trajectories):
    return torch.tensor([
        [float(sum(tokens.tolist())), float(len(tokens))]
        for tokens in trajectories
    ])


def test_jepa_beam_uses_semantics_not_proposal_probability():
    # Token 1 is overwhelmingly preferred by the proposal; token 2 reaches
    # the semantic goal. Newline token 9 completes the action.
    def propose(prefixes, branch):
        ids, logp = [], []
        for prefix in prefixes:
            if len(prefix) == 0:
                ids.append([1, 2])
                logp.append([-0.01, -20.0])
            else:
                ids.append([9, 8])
                logp.append([-0.01, -20.0])
        return torch.tensor(ids), torch.tensor(logp)

    def complete(tokens):
        return (int(tokens[-1]) == 9, False, tokens)

    semantic = lambda states: (states[:, 0] - 11).abs()
    jepa = jepa_beam_search(
        propose, endpoint, semantic, complete,
        horizon=2, beam_width=2, branch_factor=2, objective="jepa",
    )
    lm = jepa_beam_search(
        propose, endpoint, semantic, complete,
        horizon=2, beam_width=2, branch_factor=2, objective="lm",
    )
    assert jepa.trajectory.tokens.tolist() == [2, 9]
    assert lm.trajectory.tokens.tolist() == [1, 9]


def test_beam_preserves_complete_prefix_dependencies():
    calls = []

    def propose(prefixes, branch):
        calls.append([tuple(prefix.tolist()) for prefix in prefixes])
        ids = []
        for prefix in prefixes:
            key = tuple(prefix.tolist())
            if not key:
                ids.append([1, 2])
            elif key == (2,):
                ids.append([3, 4])
            else:
                ids.append([9, 8])
        return torch.tensor(ids), torch.zeros(len(prefixes), branch)

    result = jepa_beam_search(
        propose, endpoint, lambda states: (states[:, 0] - 14).abs(),
        lambda tokens: (int(tokens[-1]) == 9, False, tokens),
        horizon=3, beam_width=2, branch_factor=2,
    )
    assert any((2,) in row for row in calls)
    assert result.trajectory.tokens[-1].item() == 9


def test_elite_prefix_cem_passes_intact_elites_to_next_round():
    observed = []

    def sample(prefixes, population, iteration):
        observed.append(None if prefixes is None else [
            tuple(prefix.tolist()) for prefix in prefixes
        ])
        base = [
            TokenTrajectory(torch.tensor([1, 9]), False, -1.0),
            TokenTrajectory(torch.tensor([4, 9]), False, -2.0),
            TokenTrajectory(torch.tensor([7, 9]), False, -3.0),
            TokenTrajectory(torch.tensor([2, 9]), False, -4.0),
        ]
        return base

    result = elite_prefix_cem(
        sample, endpoint, lambda states: (states[:, 0] - 13).abs(),
        population=4, iterations=2, elite_fraction=0.25,
        preserve_prefix=1,
    )
    assert observed[0] is None
    assert observed[1] == [(4,)]
    assert result.trajectory.tokens.tolist() == [4, 9]


def test_markov_cem_samples_only_observed_first_order_transitions():
    initial = [
        TokenTrajectory(torch.tensor([1, 2, 9]), False, 0.0),
        TokenTrajectory(torch.tensor([1, 3, 9]), False, 0.0),
        TokenTrajectory(torch.tensor([4, 3, 9]), False, 0.0),
    ]
    result = first_order_markov_cem(
        initial, endpoint, lambda states: (states[:, 0] - 13).abs(),
        horizon=3, population=64, iterations=4, elite_fraction=0.2,
        generator=torch.Generator().manual_seed(8),
    )
    tokens = result.trajectory.tokens.tolist()
    assert tokens[0] in {1, 4}
    assert tokens[1] in ({2, 3} if tokens[0] == 1 else {3})
    assert tokens[-1] == 9


def test_markov_cem_does_not_generate_independent_position_mixtures():
    initial = [
        TokenTrajectory(torch.tensor([1, 2]), True, 0.0),
        TokenTrajectory(torch.tensor([3, 4]), True, 0.0),
    ]
    seen = []

    def rollout(trajectories):
        seen.extend(tuple(tokens.tolist()) for tokens in trajectories)
        return endpoint(trajectories)

    first_order_markov_cem(
        initial, rollout, lambda states: states[:, 0] * 0,
        horizon=2, population=128, iterations=2, elite_fraction=0.5,
        generator=torch.Generator().manual_seed(11),
    )
    assert set(seen) <= {(1, 2), (3, 4)}


def test_factorized_cem_is_explicitly_capable_of_invalid_prefix_mixtures():
    initial = [
        TokenTrajectory(torch.tensor([1, 2]), True, 0.0),
        TokenTrajectory(torch.tensor([3, 4]), True, 0.0),
    ]
    seen = []

    def rollout(trajectories):
        seen.extend(tuple(tokens.tolist()) for tokens in trajectories)
        return endpoint(trajectories)

    factorized_position_cem(
        initial, rollout, lambda states: states[:, 0] * 0,
        horizon=2, population=512, iterations=1, elite_fraction=0.5,
        generator=torch.Generator().manual_seed(17),
    )
    assert {(1, 4), (3, 2)} & set(seen)
