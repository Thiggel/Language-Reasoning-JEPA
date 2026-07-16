from functools import partial

import torch
from torch.utils.data import DataLoader

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    faithful_token_edit_vocab,
)
from textjepa.models.edit_jepa import EditJEPA
from textjepa.objectives.delta_action import ObservedActionLDAD


def test_faithful_token_edits_are_text_only_and_recover_target():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=91, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=3, max_edits=4,
    )
    for index in range(2):
        item = dataset[index]
        assert len(item["actions"]) >= 3
        assert item["remaining"][-1] == 0
        action_text = vocab.decode(item["actions"][0])
        assert "token position" in action_text
        assert not any(word in action_text for word in ("ancestor", "necessary"))
        recovered = [token for sentence in item["buffers"][-1] for token in sentence]
        assert recovered == item["target_tokens"]


def test_faithful_token_edit_model_is_causal_hierarchical_and_ldad_trains():
    torch.manual_seed(4)
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=93, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=4, max_edits=4,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128,
        d_action=8, d_macro=4, macro_k=2, predictor_layers=1,
        predictor_heads=4, observed_action_ldad=True,
        dense_rollout_depth=2, high_dense_rollout_depth=2,
    )
    out = model(batch)
    assert model.core.predictor.causal_sequence
    assert model.core.hi_predictor.causal_sequence
    assert out.hi_preds is not None
    assert "dense_rollout_predictions" in out.extras
    assert "high_dense_rollout_predictions" in out.extras
    loss = ObservedActionLDAD()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.chunk_encoder.tok.weight.grad is not None


def test_faithful_token_edit_encoder_supports_full_length_sentences():
    """Regression: hard/full iGSM produced a 276-token sentence in a launch."""
    vocab = faithful_token_edit_vocab()
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=320, d_action=8, predictor_layers=1,
        predictor_heads=4,
    )
    long_sentence = torch.randint(0, len(vocab), (1, 1, 276))
    encoded = model.encode_chunks(long_sentence)
    assert encoded.shape == (1, 1, 32)
