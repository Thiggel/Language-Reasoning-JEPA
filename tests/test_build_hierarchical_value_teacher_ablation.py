import sys

import torch

import scripts.build_hierarchical_value_teacher as builder
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def test_value_teacher_isolates_metric_and_total_prefix_depth(
    tmp_path, monkeypatch
):
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=11, d_token=6, d_sentence=4,
        d_action=2, d_task=3, predictor_width=8,
        token_layers=1, sentence_layers=1, n_heads=2,
        token_context=4, sentence_context=4, max_span=4,
        enable_macro_actions=True,
    )
    model = HierarchicalLanguageJEPA(config)
    learner = HierarchicalLanguageLearner(model, ResearchStage.MACRO_ACTION)
    monkeypatch.setattr(builder, "load_checkpoint", lambda path, device: (
        model, learner
    ))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    rollouts = tmp_path / "rollouts.pt"
    torch.save({
        "rollout_states": torch.randn(2, 3, 2, 4, 4),
        "rollout_log_probabilities": torch.randn(2, 3, 2, 4),
        "rollout_mask": torch.ones(2, 3, 2, 4, dtype=torch.bool),
        "goals": torch.randn(2, 1, 4),
        "goal_mask": torch.ones(2, 1, dtype=torch.bool),
        "successor_state": torch.randn(2, 3, 4),
        "successor_context": torch.randn(2, 3, 8),
        "task_hidden": torch.randn(2, 8),
        "first_action_log_probability": torch.randn(2, 3),
        "action_mask": torch.ones(2, 3, dtype=torch.bool),
        "dataset_fingerprint": "data",
        "terminal_set_fingerprint": "terminal",
        "symbolically_verified": True,
    }, rollouts)
    for metric, maximum, kind in (
        ("euclidean", 1, "raw_endpoint"),
        ("mahalanobis", 4, "supported_search_quasimetric"),
    ):
        output = tmp_path / f"{metric}.pt"
        monkeypatch.setattr(sys, "argv", [
            "build_hierarchical_value_teacher.py",
            "--checkpoint", str(checkpoint),
            "--oracle-rollouts", str(rollouts), "--output", str(output),
            "--metric", metric, "--max-prefix", str(maximum),
            "--device", "cpu",
        ])
        builder.main()
        payload = torch.load(output, map_location="cpu", weights_only=True)
        assert payload["metric_name"] == metric
        assert payload["teacher_kind"] == kind
        assert payload["max_total_prefix"] == maximum
        assert payload["teacher_cost"].shape == (2, 3)
        assert payload["architecture"] == HIERARCHICAL_LANGUAGE_ARCHITECTURE

