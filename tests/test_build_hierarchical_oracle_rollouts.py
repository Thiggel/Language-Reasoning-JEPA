import sys

import torch

import scripts.build_hierarchical_oracle_rollouts as builder
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
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


def test_oracle_rollout_cli_preserves_total_prefix_and_provenance(
    tmp_path, monkeypatch
):
    torch.manual_seed(2)
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=19, pad_id=0,
        d_token=6, d_sentence=4, d_action=2, d_task=3,
        predictor_width=8, token_layers=1, sentence_layers=1,
        n_heads=2, token_context=4, sentence_context=4,
        max_span=4, enable_macro_actions=True,
    )
    model = HierarchicalLanguageJEPA(config).eval()
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.MACRO_ACTION
    ).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "config": config.__dict__,
        "model": model.state_dict(),
        "learner": learner.state_dict(),
        "stage": ResearchStage.MACRO_ACTION.name,
    }, checkpoint)
    features = tmp_path / "features.pt"
    torch.save({
        "hidden_states": torch.randn(2, 8, 8),
        "input_ids": torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 9, 10, 11, 12, 13],
        ]),
        "boundaries": torch.tensor([[3, 5, 7], [3, 6, 7]]),
        "prompt_len": torch.tensor([3, 3]),
        "solution_end": torch.tensor([7, 7]),
        "problem_id": ["a", "b"],
        "reasoning_depth": torch.tensor([1, 1]),
        "canonical_state_ids": torch.tensor([[1, 2, 2], [3, 4, 4]]),
        "symbolically_verified": [True, True],
        "dataset_fingerprint": "dataset-1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
    }, features)
    output = tmp_path / "rollouts.pt"
    monkeypatch.setattr(sys, "argv", [
        "build_hierarchical_oracle_rollouts.py",
        "--features", str(features), "--checkpoint", str(checkpoint),
        "--output", str(output), "--max-roots", "2",
        "--first-actions", "3", "--continuation-samples", "2",
        "--max-prefix", "4", "--device", "cpu", "--latent-diagnostic",
    ])
    builder.main()
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["rollout_states"].shape == (2, 3, 2, 4, 4)
    assert payload["successor_state"].shape == (2, 3, 4)
    assert payload["successor_context"].shape == (2, 3, 8)
    assert payload["first_action_log_probability"].shape == (2, 3)
    assert payload["dataset_fingerprint"] == "dataset-1"
    assert payload["sampling"]["context_history_preserved"] is True
    assert payload["terminal_set_fingerprint"]
    assert payload["oracle_rollout_fingerprint"]
    assert payload["terminal_set_symbolically_verified"] is True
    assert payload["rollouts_exactly_grounded"] is False
