import json
import sys
from types import SimpleNamespace

import torch
import pytest

import scripts.evaluate_hierarchical_language_mpc as evaluator
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
    collate_counterfactual_records,
)
from textjepa.data.provenance import sha256_file
from scripts.train_hierarchical_language_jepa import (
    _validate_grounded_planner_replay,
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
from textjepa.planning.grounded_language_worker import WorkerBank


class FakeFrozen:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, *, input_ids, **kwargs):
        self.calls["prefix_encode"] = self.calls.get("prefix_encode", 0) + 1
        hidden = input_ids.float()[..., None].repeat(1, 1, 8) / 10
        return SimpleNamespace(hidden_states=[hidden])


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return "Final answer: \\boxed{8}\n"


@pytest.mark.parametrize("mode", ["oracle", "value"])
def test_mpc_cli_executes_worker_and_exactly_reencodes_without_value_leakage(
    tmp_path, monkeypatch, mode
):
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=19, pad_id=0,
        d_token=6, d_sentence=4, d_action=2, d_task=3,
        predictor_width=8, token_layers=1, sentence_layers=1,
        n_heads=2, token_context=4, sentence_context=4,
        max_span=4, enable_macro_actions=True, enable_value=mode == "value",
    )
    model = HierarchicalLanguageJEPA(config).eval()
    learner = HierarchicalLanguageLearner(
        model, (
            ResearchStage.MACRO_ACTION
            if mode == "oracle" else ResearchStage.VALUE_DISTILLATION
        )
    ).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "config": config.__dict__, "model": model.state_dict(),
        "learner": learner.state_dict(),
        "stage": (
            ResearchStage.MACRO_ACTION.name
            if mode == "oracle" else ResearchStage.VALUE_DISTILLATION.name
        ),
    }, checkpoint)
    features = tmp_path / "features.pt"
    reference_hidden = torch.randn(1, 6, 8)
    if mode == "value":
        # A no-terminal evaluation must never read the reference solution
        # state, even though canonical feature artifacts contain it.
        reference_hidden[:, 3:] = torch.nan
    torch.save({
        "hidden_states": reference_hidden,
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "prompt_len": torch.tensor([3]),
        "solution_end": torch.tensor([6]),
        "problem_id": ["p"],
        "dataset_fingerprint": "data",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "symbolically_verified": [True],
    }, features)
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps({
        "problem_id": "p", "answer": 8, "reasoning_depth": 1,
    }) + "\n")
    calls = {"encode": 0}
    frozen = FakeFrozen(calls)

    def fake_load_reference(device, dtype):
        return FakeTokenizer(), frozen

    def fake_bank(model, frozen, tokenizer, prefix, hidden, *, population, **kwargs):
        calls["encode"] += 1
        endpoint = model.e0_to_1(model.e0(torch.ones(population, 8)))
        ids = torch.full((population, 1), 7)
        return WorkerBank(
            candidates=[(torch.tensor([7]), True) for _ in range(population)],
            predicted_coarse=endpoint, exact_coarse=endpoint,
            actions=model.a1(ids, torch.ones_like(ids, dtype=torch.bool)),
            lm_log_probability=torch.zeros(population),
            generation_seconds=0.0, exact_grounding_seconds=0.0,
            token_rollout_seconds=0.0,
        )

    monkeypatch.setattr(evaluator, "load_reference_model", fake_load_reference)
    monkeypatch.setattr(evaluator, "build_worker_bank", fake_bank)
    output = tmp_path / "evaluation.json"
    replay_output = tmp_path / "planner_replay.pt"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_hierarchical_language_mpc.py",
        "--features", str(features), "--examples", str(examples),
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--replay-output", str(replay_output),
        "--dataset-split", "id_test", "--mode", mode,
        "--metric", "euclidean", "--k0", "8", "--k1", "1",
        "--worker-population", "2", "--manager-population", "4",
        "--cem-iterations", "1", "--max-examples", "1",
        "--max-reasoning-steps", "2", "--device", "cpu",
    ])
    evaluator.main()
    result = json.loads(output.read_text())
    assert result["accuracy"] == 1.0
    assert result["rows"][0]["generated_steps"] == 1
    assert result["rows"][0]["generated_tokens"] == 1
    # Initial exact state plus mandatory exact post-execution re-encoding.
    assert calls["prefix_encode"] == 2
    assert calls["encode"] == 1
    assert result["rows"][0]["exact_reencode_seconds"] >= 0
    replay = torch.load(replay_output, map_location="cpu", weights_only=True)
    _validate_grounded_planner_replay(
        replay, dataset_fingerprint="data",
        checkpoint_sha256=sha256_file(checkpoint),
        require_exact_worker_achievement=True,
    )
    batch = collate_counterfactual_records(replay["counterfactual_records"])
    assert batch["sentence_eligible"].all()
