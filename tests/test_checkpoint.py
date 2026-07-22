import torch

from textjepa.utils.checkpoint import _migrate_legacy_state_dict


def test_migrates_legacy_macro_encoder_keys_only():
    state = {
        "core.macro_encoder.cls": torch.ones(1),
        "core.macro_encoder.encoder.layers.0.weight": torch.ones(2),
        "core.predictor.weight": torch.ones(3),
    }

    migrated = _migrate_legacy_state_dict(state)

    assert "core.macro_encoder.encoder.cls" in migrated
    assert "core.macro_encoder.encoder.encoder.layers.0.weight" in migrated
    assert migrated["core.predictor.weight"] is state["core.predictor.weight"]


def test_leaves_current_macro_encoder_keys_unchanged():
    state = {"core.macro_encoder.encoder.cls": torch.ones(1)}
    assert _migrate_legacy_state_dict(state) is state
