from types import SimpleNamespace

from scripts.run_hierarchical_mpc_search_matrix import _cell_id, _cells


def args(**overrides):
    values = dict(
        splits=["id_test", "near_length_ood"],
        worker_search=["beam", "markov_cem"],
        worker_objectives=["jepa", "combined"],
        manager_support=["ambient", "prior_trust"],
        execution=["open_loop", "closed_loop"],
        worker_execution_tokens=[0],
        k0=[16, 32], k1=[1, 4], shard_index=0, shard_count=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_matrix_covers_cartesian_product_and_excludes_invalid_markov_likelihood():
    cells = _cells(args())
    # Beam has two objectives; Markov only the JEPA objective.
    assert len(cells) == 2 * 3 * 2 * 2 * 2 * 2
    assert all(not (
        cell["worker_search"] == "markov_cem"
        and cell["worker_objective"] != "jepa"
    ) for cell in cells)
    assert len({_cell_id(cell) for cell in cells}) == len(cells)


def test_deterministic_shards_are_disjoint_and_complete():
    complete = {_cell_id(cell) for cell in _cells(args())}
    shards = [
        {_cell_id(cell) for cell in _cells(args(
            shard_index=index, shard_count=3
        ))}
        for index in range(3)
    ]
    assert set.union(*shards) == complete
    assert all(shards[left].isdisjoint(shards[right])
               for left in range(3) for right in range(left + 1, 3))


def test_matrix_crosses_token_execution_intervals_but_not_for_one_shot():
    cells = _cells(args(
        splits=["id_test"], worker_search=["one_shot", "beam"],
        worker_objectives=["jepa"], manager_support=["prior_trust"],
        execution=["closed_loop"], worker_execution_tokens=[0, 1, 4],
        k0=[16], k1=[1],
    ))
    assert len(cells) == 4
    one_shot = [cell for cell in cells if cell["worker_search"] == "one_shot"]
    beam = [cell for cell in cells if cell["worker_search"] == "beam"]
    assert [cell["worker_execution_tokens"] for cell in one_shot] == [0]
    assert {cell["worker_execution_tokens"] for cell in beam} == {0, 1, 4}
    assert len({_cell_id(cell) for cell in cells}) == len(cells)
