import json

import pytest
import torch

from textjepa.data.hierarchical_language import (
    BoundaryPolicy,
    CandidateSource,
    CounterfactualBatch,
    InformationProvenance,
    PlannerReplayRecord,
    read_replay_jsonl,
    write_replay_jsonl,
)


def test_boundary_policy_splits_long_and_merges_tiny_spans():
    policy = BoundaryPolicy(min_span=2, max_span=4)
    assert policy.normalize([0, 1, 7, 9], 10) == [0, 4, 7, 9]


def test_counterfactual_batch_flattens_root_candidate_rectangle():
    batch = CounterfactualBatch(
        token_ids=torch.arange(24).reshape(2, 3, 4),
        hidden_states=torch.randn(2, 3, 4, 5),
        lengths=torch.tensor([[4, 3, 2], [4, 4, 1]]),
        root_ids=torch.tensor([[10, 10, 10], [11, 11, 11]]),
        temperatures=torch.tensor([[0.0, 0.3, 0.7], [0.0, 1.0, 1.2]]),
    )
    flat = batch.flatten()
    assert flat["token_ids"].shape == (6, 4)
    assert flat["hidden_states"].shape == (6, 4, 5)
    assert flat["root_ids"].tolist() == [10, 10, 10, 11, 11, 11]


def test_counterfactual_batch_rejects_invalid_lengths():
    batch = CounterfactualBatch(
        token_ids=torch.zeros(1, 1, 2, dtype=torch.long),
        hidden_states=torch.zeros(1, 1, 2, 3),
        lengths=torch.tensor([[3]]),
        root_ids=torch.tensor([[0]]),
        temperatures=torch.tensor([[0.7]]),
    )
    with pytest.raises(ValueError, match="length"):
        batch.flatten()


def test_replay_round_trip_preserves_oracle_and_symbolic_labels(tmp_path):
    record = PlannerReplayRecord(
        problem_id="q1", root_id="q1:step2",
        source=CandidateSource.CEM.value, action_tokens=[4, 5],
        predicted_endpoint_cost=1.2, exact_endpoint_cost=1.8,
        achieved_endpoint_cost=2.0, action_log_probability=-4.2,
        valid_symbolic_step=False, symbolic_distance_before=3,
        symbolic_distance_after=4, cem_iteration=2, wall_seconds=0.2,
        predictor_flops=1200,
        provenance=InformationProvenance(
            oracle_terminal=True, symbolic=True, candidate_privileged=True
        ),
    )
    path = tmp_path / "replay.jsonl"
    write_replay_jsonl(path, [record])
    restored = read_replay_jsonl(path)
    assert restored == [record]
    raw = json.loads(path.read_text().strip())
    assert raw["provenance"]["oracle_terminal"] is True
    assert raw["provenance"]["cross_project"] is False


def test_replay_requires_candidate_level_compute_metadata():
    record = PlannerReplayRecord(
        problem_id="q", root_id="r", source="observed", action_tokens=[1],
        predicted_endpoint_cost=0, exact_endpoint_cost=0,
        achieved_endpoint_cost=None, action_log_probability=0,
        valid_symbolic_step=None, symbolic_distance_before=None,
        symbolic_distance_after=None, cem_iteration=None,
        wall_seconds=-1, predictor_flops=0,
        provenance=InformationProvenance(),
    )
    with pytest.raises(ValueError, match="compute"):
        record.validate()
