from functools import partial

import torch
from torch.utils.data import DataLoader

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    MASK_TOKEN,
    faithful_token_edit_vocab,
)
from textjepa.models.edit_jepa import EditJEPA
from textjepa.models.ema import EMATeacher
from textjepa.models.predictor import TokenAlignedEditPredictor
from textjepa.objectives import (
    GoalAdvantageDistill, TokenAlignedCounterfactualPrediction,
)
from scripts.audit_faithful_token_edits import shuffled_action_prediction


def _dataset(mode: str):
    vocab = faithful_token_edit_vocab()
    return vocab, FaithfulTokenEditDataset(
        vocab, size=4, seed=41, min_edits=3, max_edits=4,
        corruption_mode=mode,
    )


def test_ema_teacher_stays_in_eval_mode():
    teacher = EMATeacher(torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.9)
    ))
    teacher.train(True)
    assert not teacher.training
    assert not teacher.module.training


def test_corruption_modes_are_distinct_and_exact():
    vocab, masked = _dataset("mask")
    item = masked[0]
    assert any(vocab.token_to_id[MASK_TOKEN] in step for step in item["buffers"][0])
    assert set(item["op"]) == {2}
    assert item["buffers"][-1] == masked.source[0]["steps"]

    _, replaced = _dataset("replace")
    item = replaced[0]
    assert set(item["op"]) == {2}
    assert item["resolved_n"][0] == item["resolved_n"][-1]
    assert item["buffers"][-1] == replaced.source[0]["steps"]

    _, removed = _dataset("remove")
    item = removed[0]
    assert set(item["op"]) == {1}
    assert item["resolved_n"][0] < item["resolved_n"][-1]
    assert item["buffers"][-1] == removed.source[0]["steps"]


def test_curriculum_changes_corruption_family_by_epoch():
    _, dataset = _dataset("curriculum")
    dataset.curriculum_epochs = 4
    dataset.set_epoch(0)
    assert dataset._active_corruption_mode() == "mask"
    dataset.set_epoch(1)
    assert dataset._active_corruption_mode() == "replace"
    dataset.set_epoch(2)
    assert dataset._active_corruption_mode() == "mixed"


def test_fresh_per_epoch_is_explicit_and_reproducible():
    vocab = faithful_token_edit_vocab()
    common = dict(
        vocab=vocab, size=4, seed=43, min_edits=8, max_edits=8,
        corruption_mode="mixed",
    )
    fixed = FaithfulTokenEditDataset(**common, fresh_per_epoch=False)
    before = [fixed[index]["buffers"][0] for index in range(len(fixed))]
    fixed.set_epoch(1)
    assert [fixed[index]["buffers"][0] for index in range(len(fixed))] == before

    fresh = FaithfulTokenEditDataset(**common, fresh_per_epoch=True)
    before = [fresh[index]["buffers"][0] for index in range(len(fresh))]
    fresh.set_epoch(1)
    after = [fresh[index]["buffers"][0] for index in range(len(fresh))]
    assert after != before
    repeat = FaithfulTokenEditDataset(**common, fresh_per_epoch=True)
    repeat.set_epoch(1)
    assert [repeat[index]["buffers"][0] for index in range(len(repeat))] == after


def test_pointer_action_is_invariant_to_prefix_shift():
    torch.manual_seed(0)
    predictor = TokenAlignedEditPredictor(16, 8, n_layers=1, n_heads=4)
    x, y, z, prefix, content = torch.randn(5, 16)
    state_a = torch.stack([x, y, z]).unsqueeze(0)
    state_b = torch.stack([prefix, x, y, z]).unsqueeze(0)
    code_a = predictor.encode_action(
        state_a, torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([1]), torch.tensor([2]), content.unsqueeze(0),
    )
    code_b = predictor.encode_action(
        state_b, torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([1]), torch.tensor([3]), content.unsqueeze(0),
    )
    torch.testing.assert_close(code_a, code_b)


def test_structured_model_forward_and_recursive_shapes():
    vocab, dataset = _dataset("mask")
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0,
    )
    out = model(batch)
    assert out.extras["token_predictions"].shape[:2] == out.step_mask.shape
    assert out.extras["token_rollout_predictions"].shape == out.extras["token_predictions"].shape
    assert out.actions.shape[-1] == 8
    assert torch.isfinite(out.preds).all()
    assert torch.isfinite(out.rollout).all()
    assert out.extras["gar_action_value"].shape == out.step_mask.shape
    assert not out.extras["gar_action_target"].requires_grad
    assert torch.isfinite(GoalAdvantageDistill()(out, batch))

    shuffled, reason, changed = shuffled_action_prediction(
        model, out, batch, return_output=True
    )
    assert reason is None
    assert bool(changed.any())
    assert shuffled.extras["token_predictions"].shape == out.extras[
        "token_predictions"
    ].shape


def test_structured_counterfactuals_supervise_exact_token_outcomes():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=47, min_edits=3, max_edits=4,
        corruption_mode="mixed", counterfactual_k=2,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    assert batch["alt_op"].shape == batch["alt_valid"].shape
    assert batch["alt_edit_position"].shape == batch["alt_valid"].shape
    assert batch["alt_edit_content_token"].shape == batch["alt_valid"].shape
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0,
    )
    out = model(batch)
    assert out.extras["cf_token_pred"].shape == out.extras["cf_token_tgt"].shape
    assert out.extras["cf_token_valid"].shape == batch["alt_valid"].shape
    loss = TokenAlignedCounterfactualPrediction()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.token_pred.out.weight.grad is not None
