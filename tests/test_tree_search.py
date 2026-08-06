import torch

from textjepa.planning.tree_search import best_first_search, puct_search


def transition(actions):
    return actions.float().cumsum(0).unsqueeze(-1)


def propose(state, topk):
    actions = torch.tensor([0, 1], device=state.device)[:topk]
    probabilities = torch.tensor([0.2, 0.8], device=state.device)[:topk]
    return actions, probabilities / probabilities.sum()


def leaf_cost(states):
    return (states[:, 0] - 2.0).abs()


def test_beam_and_astar_recover_supported_goal_sequence():
    start = torch.tensor([0.0])
    example = torch.zeros(1, dtype=torch.long)
    for mode in ("beam", "astar"):
        result = best_first_search(
            transition, propose, leaf_cost, start, example,
            horizon=2, width=8, topk=2, mode=mode,
        )
        assert result.actions.tolist() == [1, 1]
        assert result.cost == 0.0


def test_puct_recovers_supported_goal_sequence():
    result = puct_search(
        transition, propose, leaf_cost, torch.tensor([0.0]),
        torch.zeros(1, dtype=torch.long), horizon=2,
        simulations=32, topk=2,
    )
    assert result.actions.tolist() == [1, 1]
    assert result.cost == 0.0
