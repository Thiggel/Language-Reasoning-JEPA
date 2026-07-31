import argparse
import sys

import pytest
import torch

import scripts.build_hierarchical_oracle_rollouts as rollouts
import scripts.run_full_hierarchical_language_experiment as runner
from textjepa.models.hierarchical_language_jepa import (
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.planning.grounded_language_worker import WorkerBank
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def _model():
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=19, pad_id=0, d_token=6, d_sentence=4,
        d_action=2, d_task=3, predictor_width=8, token_layers=1,
        sentence_layers=1, n_heads=2, token_context=4,
        sentence_context=4, max_span=4, enable_macro_actions=True,
    )
    model = HierarchicalLanguageJEPA(config).eval()
    return model, HierarchicalLanguageLearner(
        model, ResearchStage.MACRO_ACTION
    ).eval()


def test_depth_grid_is_the_complete_cartesian_product():
    assert len(runner.DEPTH_PAIRS) == 16
    assert set(runner.DEPTH_PAIRS) == {
        (k0, k1) for k0 in (8, 16, 32, 64) for k1 in (1, 2, 4, 8)
    }


def test_missing_output_root_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_full_hierarchical_language_experiment.py",
        "--source-root", str(tmp_path),
    ])
    with pytest.raises(ValueError, match="output-root"):
        runner.main()


def test_grounded_rollout_executes_worker_and_rescores_achieved_action(
    tmp_path, monkeypatch
):
    model, learner = _model()
    features = {
        "hidden_states": torch.randn(1, 5, 8),
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "boundaries": torch.tensor([[3, 5]]),
        "prompt_len": torch.tensor([3]), "solution_end": torch.tensor([5]),
        "problem_id": ["p"], "reasoning_depth": torch.tensor([1]),
        "canonical_state_ids": torch.tensor([[1, 2]]),
        "dataset_fingerprint": "data",
    }
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"problem_id":"p","reasoning_operations":['
        '"so the number of red shells is 3 plus 4 = 7 ."],'
        '"canonical_state_ids":[1,2],"answer":7}\n'
    )
    calls = {"worker": 0, "reencode": 0}

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "so the number of red shells is 3 plus 4 = 7 ."

    monkeypatch.setattr(
        rollouts, "load_reference_model", lambda device, dtype: (Tokenizer(), object())
    )

    def bank(model, frozen, tokenizer, prefix, hidden, **kwargs):
        calls["worker"] += 1
        ids = torch.tensor([[4, 5]])
        action = model.a1(ids, torch.ones_like(ids, dtype=torch.bool))
        endpoint = model.e0_to_1(model.e0(torch.ones(1, 8)))
        return WorkerBank(
            candidates=[(torch.tensor([4, 5]), False)],
            predicted_coarse=endpoint, exact_coarse=endpoint,
            actions=action, lm_log_probability=torch.zeros(1),
            generation_seconds=0.0, exact_grounding_seconds=0.0,
            token_rollout_seconds=0.0,
        )

    def encode(frozen, prefix):
        calls["reencode"] += 1
        return torch.ones(len(prefix), 8)

    monkeypatch.setattr(rollouts, "build_worker_bank", bank)
    monkeypatch.setattr(rollouts, "encode_frozen_prefix", encode)
    args = argparse.Namespace(
        examples=examples, device="cpu", dtype="bfloat16", seed=0,
        first_actions=2, continuation_samples=1, max_prefix=1,
        worker_population=1, k0=2, worker_prior_weight=0.0,
        temperature=0.8, top_p=0.95, top_k=0,
        checkpoint=tmp_path / "checkpoint.pt", features=tmp_path / "features.pt",
    )
    args.checkpoint.write_bytes(b"checkpoint")
    args.features.write_bytes(b"features")
    payload = rollouts._build_grounded_payload(
        args, features, model, learner, [(0, 0)]
    )
    assert payload["rollouts_exactly_grounded"] is True
    assert payload["terminal_set_symbolically_verified"] is True
    assert payload["rollout_mask"].all()
    assert payload["rollout_symbolically_verified"].all()
    assert calls == {"worker": 1, "reencode": 2}
    assert torch.isfinite(payload["first_action_log_probability"]).all()
