from types import SimpleNamespace
import sys

import pytest
import torch

from scripts import collect_hierarchical_language_features as collector
from scripts import train_hierarchical_language_jepa as trainer
from textjepa.analysis.compute import ComputeLedger
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    PRIMARY_EOS_TOKEN_ID,
    TRANSFORMERS_VERSION,
)


def _examples(count: int):
    return [
        SimpleNamespace(
            problem_id=f"problem-{index}",
            template_family="template",
            graph_family="graph",
            symbolically_verified=True,
            step_boundaries=torch.tensor([1, 4]),
        )
        for index in range(count)
    ]


def _feature_payload(examples):
    count = len(examples)
    tokens = torch.tensor(
        [[7, 8, 9, 10, PRIMARY_EOS_TOKEN_ID]] * count,
        dtype=torch.long,
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "hidden_states": torch.randn(count, 5, 6),
        "input_ids": tokens,
        "attention_mask": torch.ones(count, 5, dtype=torch.bool),
        "prompt_len": torch.ones(count, dtype=torch.long),
        "solution_end": torch.full((count,), 4, dtype=torch.long),
        "boundaries": torch.tensor([[1, 4]] * count),
        "step_boundary_mask": torch.ones(count, 2, dtype=torch.bool),
        "reasoning_depth": torch.zeros(count, dtype=torch.long),
        "canonical_state_ids": torch.tensor([[0, 1]] * count),
        "problem_id": [example.problem_id for example in examples],
        "template_family": ["template"] * count,
        "graph_family": ["graph"] * count,
        "symbolically_verified": [True] * count,
    }


def _args(tmp_path, *, output_shard_size=2):
    return SimpleNamespace(
        output=tmp_path / "features.pt",
        counterfactual_output=tmp_path / "counterfactual.pt",
        output_shard_size=output_shard_size,
        counterfactual_shard_size=2,
        device="cpu",
        batch_size=2,
        dtype="bfloat16",
        example_limit=0,
        shard_index=0,
        num_shards=1,
        resume=False,
        seed=3,
        generation_batch_size=2,
        reencode_batch_size=2,
        counterfactual_example_limit=0,
        counterfactual_engine="legacy",
    )


def test_observed_collection_never_encodes_more_than_part_bound(
    tmp_path, monkeypatch
):
    calls = []

    def bounded_encode(examples, model, device, batch_size, ledger):
        del model, device, batch_size, ledger
        calls.append(len(examples))
        return _feature_payload(examples)

    monkeypatch.setattr(collector, "encode_examples", bounded_encode)
    monkeypatch.setattr(collector, "_backend_metadata", lambda: {})
    monkeypatch.setattr(collector, "_text_parameter_count", lambda model: 1)
    args = _args(tmp_path)
    manifest = collector._save_observed_feature_parts(
        _examples(5), object(), args, "input-fingerprint", ComputeLedger()
    )
    collector._atomic_torch_save(manifest, args.output)

    assert calls == [2, 2, 1]
    assert manifest["example_count"] == 5
    assert len(manifest["parts"]) == 3
    assert all(
        int(part["example_count"]) <= 2 for part in manifest["parts"]
    )


def test_feature_manifest_is_lazily_consumable_by_trainer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector, "encode_examples",
        lambda examples, model, device, batch_size, ledger: (
            _feature_payload(examples)
        ),
    )
    monkeypatch.setattr(collector, "_backend_metadata", lambda: {})
    monkeypatch.setattr(collector, "_text_parameter_count", lambda model: 1)
    args = _args(tmp_path)
    manifest = collector._save_observed_feature_parts(
        _examples(5), object(), args, "input-fingerprint", ComputeLedger()
    )
    collector._atomic_torch_save(manifest, args.output)

    collection = trainer.FeatureCollection(
        args.output, require_provenance=True
    )
    assert collection.example_count == 5
    assert collection._verified == set()
    first = collection.load(0)
    assert len(first["hidden_states"]) == 2
    assert collection._verified == {0}
    assert len(collection.load(2)["hidden_states"]) == 1
    assert collection._verified == {0, 2}


def test_completed_manifest_rejects_missing_or_corrupted_parts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector, "encode_examples",
        lambda examples, model, device, batch_size, ledger: (
            _feature_payload(examples)
        ),
    )
    monkeypatch.setattr(collector, "_backend_metadata", lambda: {})
    monkeypatch.setattr(collector, "_text_parameter_count", lambda model: 1)
    args = _args(tmp_path)
    manifest = collector._save_observed_feature_parts(
        _examples(3), object(), args, "input-fingerprint", ComputeLedger()
    )
    collector._atomic_torch_save(manifest, args.output)

    collector._validate_manifest_parts(
        args.output,
        manifest,
        count_key="example_count",
        expected_total=3,
    )
    first = args.output.parent / manifest["parts"][0]["path"]
    first.write_bytes(first.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupted part"):
        collector._validate_manifest_parts(
            args.output,
            manifest,
            count_key="example_count",
            expected_total=3,
        )
def test_counterfactual_collection_commits_bounded_parts(
    tmp_path, monkeypatch
):
    calls = []

    def fake_collect(examples, *args, **kwargs):
        del args, kwargs
        calls.append(len(examples))
        return [{"slot": index} for index in range(8 * len(examples))]

    monkeypatch.setattr(collector, "collect_counterfactuals", fake_collect)
    monkeypatch.setattr(collector, "_backend_metadata", lambda: {})
    args = _args(tmp_path)
    manifest = collector._save_counterfactual_parts(
        _examples(5), object(), object(), args, "input-fingerprint",
        "d" * 64, ComputeLedger(),
    )

    assert calls == [2, 2, 1]
    assert len(manifest["parts"]) == 3
    assert manifest["record_count"] == 40
    assert manifest["candidate_accounting"]["empty_or_failed_slots"] == 0


def test_training_entrypoint_streams_multiple_feature_parts(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "features.pt"
    parts_dir = tmp_path / "features.pt.parts"
    parts_dir.mkdir()
    parts = []
    for index in range(2):
        path = parts_dir / f"features-{index:05d}.pt"
        torch.save({
            "hidden_states": torch.randn(2, 6, 6),
            "input_ids": torch.randint(1, 8, (2, 6)),
            "boundaries": torch.tensor([[0, 3, 5], [0, 2, 5]]),
        }, path)
        parts.append({
            "path": str(path.relative_to(tmp_path)),
            "example_count": 2,
        })
    torch.save({
        "artifact_kind": "hierarchical_feature_manifest",
        "parts": parts,
        "example_count": 4,
        "dataset_fingerprint": "d" * 64,
    }, manifest_path)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(manifest_path), "--allow-unpinned-features",
        "--output", str(output), "--vocab-size", "8", "--pad-id", "0",
        "--d-token", "4", "--d-sentence", "4", "--d-action", "2",
        "--predictor-width", "4", "--token-layers", "1",
        "--sentence-layers", "1", "--heads", "1", "--max-span", "5",
        "--epochs", "2", "--batch-size", "2",
        "--max-optimizer-steps", "3", "--device", "cpu",
    ])

    trainer.main()

    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["optimizer_step"] == 3


def test_training_entrypoint_streams_counterfactual_parts(
    tmp_path, monkeypatch
):
    features = tmp_path / "features.pt"
    torch.save({
        "dataset_fingerprint": "d" * 64,
        "hidden_states": torch.randn(2, 6, 6),
        "input_ids": torch.randint(1, 8, (2, 6)),
        "boundaries": torch.tensor([[0, 3, 5], [0, 2, 5]]),
    }, features)
    replay_manifest = tmp_path / "counterfactual.pt"
    replay_dir = tmp_path / "counterfactual.pt.parts"
    replay_dir.mkdir()
    replay_parts = []
    for part_index in range(2):
        records = []
        for _ in range(2):
            root = torch.randn(6)
            records.append({
                "source": "observed",
                "temperature": None,
                "token_ids": torch.tensor([1, 2]),
                "root_hidden": root,
                "suffix_hidden": torch.randn(2, 6),
                "completed_boundary": True,
                "sentence_eligible": True,
                "terminal_eos": False,
                "lm_log_probability": torch.tensor(-1.0),
                "root_token_history_hidden": root[None],
                "root_token_history_action_ids": torch.empty(
                    0, dtype=torch.long
                ),
                "root_sentence_history_hidden": root[None],
                "root_sentence_history_spans": [],
            })
        path = replay_dir / f"counterfactual-{part_index:05d}.pt"
        torch.save({
            "records": records,
            "candidate_accounting": {
                "expected_slots": 2,
                "materialized_nonempty_slots": 2,
                "empty_or_failed_slots": 0,
            },
        }, path)
        replay_parts.append({
            "path": str(path.relative_to(tmp_path)),
            "record_count": 2,
        })
    torch.save({
        "artifact_kind": "hierarchical_counterfactual_manifest",
        "parts": replay_parts,
        "record_count": 4,
        "candidate_accounting": {
            "expected_slots": 4,
            "materialized_nonempty_slots": 4,
            "empty_or_failed_slots": 0,
        },
    }, replay_manifest)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features), "--allow-unpinned-features",
        "--counterfactual-features", str(replay_manifest),
        "--output", str(output), "--vocab-size", "8", "--pad-id", "0",
        "--d-token", "4", "--d-sentence", "4", "--d-action", "2",
        "--predictor-width", "4", "--token-layers", "1",
        "--sentence-layers", "1", "--heads", "1", "--max-span", "5",
        "--epochs", "1", "--batch-size", "2", "--replay-batch-size", "2",
        "--max-optimizer-steps", "3", "--device", "cpu",
    ])

    trainer.main()

    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["optimizer_step"] == 3
