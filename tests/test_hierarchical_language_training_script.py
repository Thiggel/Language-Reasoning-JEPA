import argparse
import json
import hashlib
import sys

import torch
import pytest

from scripts import evaluate_hierarchical_language_oracles as evaluate
from scripts.run_hierarchical_language_pilot import replay_batch_size
from scripts import train_hierarchical_language_jepa as train
from textjepa.analysis.compute import ComputeLedger
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    PRIMARY_EOS_TOKEN_ID,
    TRANSFORMERS_VERSION,
)
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def test_offline_training_entrypoint_end_to_end(tmp_path, monkeypatch):
    features = tmp_path / "features.pt"
    output = tmp_path / "run"
    torch.save({
        "hidden_states": torch.randn(4, 7, 10),
        "token_ids": torch.randint(1, 20, (4, 7)),
        "boundaries": torch.tensor([
            [0, 3, 6], [0, 2, 6], [0, 3, 6], [0, 2, 6],
        ]),
    }, features)
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features),
        "--allow-unpinned-features",
        "--output", str(output),
        "--vocab-size", "20",
        "--pad-id", "0",
        "--d-token", "8",
        "--d-sentence", "6",
        "--d-action", "4",
        "--predictor-width", "16",
        "--token-layers", "1",
        "--sentence-layers", "1",
        "--heads", "2",
        "--token-context", "4",
        "--sentence-context", "3",
        "--max-span", "5",
        "--epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
    ])
    train.main()
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    metrics = json.loads((output / "metrics.json").read_text())
    assert checkpoint["stage"] == "TOKEN_JEPA"
    assert checkpoint["architecture"] == HIERARCHICAL_LANGUAGE_ARCHITECTURE
    assert checkpoint["config"]["enable_value"] is False
    assert checkpoint["config"]["enable_macro_actions"] is False
    assert len(metrics["history"]) == 1
    assert metrics["history"][0]["total"] > 0
    assert "compute" in metrics
    assert metrics["compute"]["components"]["dense_p0"][
        "estimated_flops"
    ] > 0
    assert metrics["compute"]["components"]["dense_training_wall"][
        "wall_seconds"
    ] > 0


def test_collector_schema_artifact_is_directly_loadable_by_trainer(tmp_path):
    path = tmp_path / "features.pt"
    torch.save({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "hidden_states": torch.randn(1, 5, 3),
        "input_ids": torch.tensor([[3, 4, 5, 6, PRIMARY_EOS_TOKEN_ID]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "prompt_len": torch.tensor([1]),
        "solution_end": torch.tensor([4]),
        "boundaries": torch.tensor([[1, 4]]),
        "step_boundary_mask": torch.ones(1, 2, dtype=torch.bool),
        "reasoning_depth": torch.tensor([0]),
        "canonical_state_ids": torch.tensor([[0, 1]]),
        "problem_id": ["p"],
        "template_family": ["t"],
        "graph_family": ["g"],
        "symbolically_verified": [True],
        "dataset_fingerprint": "0" * 64,
    }, path)
    loaded = train.load_features(path, require_provenance=True)
    assert torch.equal(loaded["input_ids"], torch.tensor(
        [[3, 4, 5, 6, PRIMARY_EOS_TOKEN_ID]]
    ))


def test_stage_zero_is_validation_only():
    model = HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=6, vocab_size=10, d_token=4, d_sentence=3,
        d_action=2, predictor_width=4, token_layers=1,
        sentence_layers=1, n_heads=1, max_span=4,
    ))
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.DATA_VALIDATION
    )
    total, losses = learner(
        torch.randn(1, 4, 6), torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[0, 3]]),
    )
    assert total.item() == 0
    assert set(losses) == {"total"}


def test_sentence_stage_loads_admitted_token_checkpoint(
    tmp_path, monkeypatch
):
    features = tmp_path / "features.pt"
    token_output = tmp_path / "token"
    sentence_output = tmp_path / "sentence"
    torch.save({
        "dataset_fingerprint": "unit-test",
        "hidden_states": torch.randn(2, 6, 6),
        "token_ids": torch.tensor([
            [1, 2, 3, 4, 5, 6], [1, 3, 2, 4, 5, 6],
        ]),
        "boundaries": torch.tensor([[0, 3, 5], [0, 2, 5]]),
    }, features)
    common = [
        "--features", str(features), "--allow-unpinned-features",
        "--vocab-size", "8", "--pad-id", "0",
        "--d-token", "4", "--d-sentence", "4",
        "--d-action", "2", "--predictor-width", "4",
        "--token-layers", "1", "--sentence-layers", "1",
        "--heads", "1", "--max-span", "5", "--epochs", "1",
        "--batch-size", "2", "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py", *common,
        "--output", str(token_output),
    ])
    train.main()
    token_checkpoint = torch.load(
        token_output / "model.pt", weights_only=True
    )
    admission = tmp_path / "admission.json"
    checkpoint_sha = hashlib.sha256(
        (token_output / "model.pt").read_bytes()
    ).hexdigest()
    admission.write_text(json.dumps({
        "passed": True,
        "admitted_stage": "FLAT_ORACLE_TOKEN",
        "metrics": {
            "exact_oracle_gain": 0.1,
            "model_oracle_gap": 0.05,
        },
        "dataset_fingerprint": "unit-test",
        "admitted_checkpoint_sha256": checkpoint_sha,
    }))
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py", *common,
        "--stage", "SENTENCE_JEPA",
        "--init-checkpoint", str(token_output / "model.pt"),
        "--admission", str(admission),
        "--output", str(sentence_output),
    ])
    train.main()
    sentence_checkpoint = torch.load(
        sentence_output / "model.pt", weights_only=True
    )
    for prefix in ("e0.", "p0.", "token_action."):
        for name, value in token_checkpoint["model"].items():
            if name.startswith(prefix):
                assert torch.equal(value, sentence_checkpoint["model"][name])
    assert {"e0", "p0", "token_action"} <= set(
        sentence_checkpoint["frozen_modules"]
    )
    assert sentence_checkpoint["trainable_modules"] == [
        "e0_to_1", "a1", "p1"
    ]


def test_value_stage_rejects_token_checkpoint_even_with_later_admission(
    tmp_path
):
    token_model = HierarchicalLanguageJEPA(
        HierarchicalLanguageJEPAConfig(
            d_backbone=6, vocab_size=8, d_token=4, d_sentence=4,
            d_action=2, predictor_width=4, token_layers=1,
            sentence_layers=1, n_heads=1, max_span=5,
            enable_macro_actions=False, enable_value=False,
        )
    )
    path = tmp_path / "token.pt"
    torch.save({
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model": token_model.state_dict(),
        "learner": {},
        "config": token_model.config.__dict__,
        "stage": "TOKEN_JEPA",
    }, path)
    value_model = HierarchicalLanguageJEPA(
        HierarchicalLanguageJEPAConfig(
            **{
                **token_model.config.__dict__,
                "enable_macro_actions": True,
                "enable_value": True,
            }
        )
    )
    learner = HierarchicalLanguageLearner(
        value_model, ResearchStage.VALUE_DISTILLATION
    )
    with pytest.raises(ValueError, match="requires a MACRO_ACTION checkpoint"):
        train._initialize_from_checkpoint(
            value_model, learner, path,
            ResearchStage.VALUE_DISTILLATION,
        )


def test_sentence_stage_accepts_truncation_only_replay_batch():
    model = HierarchicalLanguageJEPA(
        HierarchicalLanguageJEPAConfig(
            d_backbone=6, vocab_size=8, d_token=4, d_sentence=4,
            d_action=2, predictor_width=4, token_layers=1,
            sentence_layers=1, n_heads=1, max_span=5,
        )
    )
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.SENTENCE_JEPA
    )
    train._configure_stage_trainability(model, ResearchStage.SENTENCE_JEPA)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    loss, _ = learner.counterfactual_loss(
        torch.randn(2, 6), torch.randn(2, 3, 6),
        torch.tensor([[1, 2, 3], [2, 3, 4]]),
        torch.tensor([3, 3]),
        sentence_eligible=torch.zeros(2, dtype=torch.bool),
    )
    assert not loss.requires_grad
    assert not train._step_if_trainable(loss, optimizer, model, 0.99)


def test_flat_oracle_cli_scores_minimal_population(tmp_path, monkeypatch):
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=6, vocab_size=8, d_token=4, d_sentence=3,
        d_action=2, predictor_width=4, token_layers=1,
        sentence_layers=1, n_heads=1,
    )
    model = HierarchicalLanguageJEPA(config)
    learner = HierarchicalLanguageLearner(model, ResearchStage.TOKEN_JEPA)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model": model.state_dict(),
        "learner": learner.state_dict(),
        "config": config.__dict__,
        "stage": "TOKEN_JEPA",
    }, checkpoint)
    candidates = tmp_path / "candidates.pt"
    torch.save({
        "candidate_tokens": torch.tensor([[[1, 2], [3, 4]]]),
        "candidate_log_probability": torch.zeros(1, 2),
        "waypoint": torch.zeros(1, 4),
        "predicted_endpoints": torch.zeros(1, 2, 4),
        "exact_endpoints": torch.zeros(1, 2, 4),
    }, candidates)
    output = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_hierarchical_language_oracles.py",
        "--checkpoint", str(checkpoint),
        "--candidates", str(candidates),
        "--output", str(output), "--mode", "flat_token",
        "--allow-unbound-candidates",
    ])
    evaluate.main()
    assert json.loads(output.read_text())["rows"][0][
        "predicted_best"
    ] == 0


def test_multistep_pilot_uses_independent_replay_microbatch():
    args = argparse.Namespace(
        replay_batch_size=32, multistep_replay_batch_size=4,
    )
    assert replay_batch_size("counterfactual", args) == 32
    assert replay_batch_size("multistep", args) == 4


def test_recursive_replay_records_additional_predictor_flops():
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=6, vocab_size=8, d_token=4, d_sentence=3,
        d_action=2, predictor_width=4, token_layers=1,
        sentence_layers=1, n_heads=1,
    )
    model = HierarchicalLanguageJEPA(config)
    ledger = ComputeLedger()
    branch = {
        "lengths": torch.tensor([4, 2]),
        "sentence_eligible": torch.tensor([False, False]),
    }
    train._record_counterfactual_flops(
        ledger, model, branch, rollout_horizon=4
    )
    summary = ledger.summary()["components"]
    assert summary["replay_p0_rollout"]["items"] == 4
    assert summary["replay_p0_rollout"]["estimated_flops"] > 0


def test_evaluation_only_stage_rejects_dense_training(tmp_path, monkeypatch):
    features = tmp_path / "features.pt"
    torch.save({
        "hidden_states": torch.randn(1, 4, 6),
        "token_ids": torch.tensor([[1, 2, 3, 4]]),
        "boundaries": torch.tensor([[0, 3]]),
    }, features)
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features), "--allow-unpinned-features",
        "--output", str(tmp_path / "run"),
        "--stage", "FLAT_ORACLE_TOKEN", "--vocab-size", "8",
        "--pad-id", "0",
        "--epochs", "1", "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="evaluation-only"):
        train.main()


def test_evaluation_only_stage_rejects_zero_epoch_checkpoint(
    tmp_path, monkeypatch
):
    features = tmp_path / "features.pt"
    torch.save({
        "hidden_states": torch.randn(1, 4, 6),
        "token_ids": torch.tensor([[1, 2, 3, 4]]),
        "boundaries": torch.tensor([[0, 3]]),
    }, features)
    monkeypatch.setattr(sys, "argv", [
        "train_hierarchical_language_jepa.py",
        "--features", str(features), "--allow-unpinned-features",
        "--output", str(tmp_path / "run"),
        "--stage", "FLAT_ORACLE_TOKEN", "--vocab-size", "8",
        "--pad-id", "0",
        "--epochs", "0", "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="evaluation-only"):
        train.main()


def test_planning_depth_validation_requires_complete_cartesian_grid():
    grid = {
        "dataset_split": ["id_test"],
        "symbolic_depth": [2],
        "population_size": [64],
        "token_oracle_horizon": [4, 8, 16],
        "endpoint_kind": ["paired"],
    }
    rows = [
        {
            "dataset_split": "id_test", "symbolic_depth": 2,
            "population_size": 64, "token_oracle_horizon": horizon,
            "endpoint_kind": "paired",
        }
        for horizon in (4, 8, 16)
    ]
    evaluate.validate_planning_depth_matrix(rows, "flat_token", grid)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate.validate_planning_depth_matrix(
            rows[:-1], "flat_token", grid
        )
