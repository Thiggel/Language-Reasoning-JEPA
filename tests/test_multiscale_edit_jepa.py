import copy

import pytest
import torch

from textjepa.models.multiscale_edit_jepa import (
    HierarchicalBufferEncoder,
    MultiscaleEditJEPA,
)


def _batch():
    # Two replacement transitions over two persistent sentence spans.
    return {
        "prompt_tokens": torch.tensor([[[2, 3, 0]]]),
        "prompt_mask": torch.tensor([[True]]),
        "buffer_tokens": torch.tensor([[[[4, 5, 0], [6, 7, 0]],
                                          [[4, 8, 0], [6, 7, 0]],
                                          [[4, 8, 0], [9, 7, 0]]]]),
        "buffer_mask": torch.tensor([[[True, True], [True, True],
                                       [True, True]]]),
        "op": torch.tensor([[2, 2]]),
        "edit_position": torch.tensor([[1, 2]]),
        "edit_content_token": torch.tensor([[8, 9]]),
        "step_mask": torch.tensor([[True, True]]),
        "action_tokens": torch.tensor([[[10, 11, 0], [10, 12, 0]]]),
    }


def _model(variant, **kwargs):
    return MultiscaleEditJEPA(
        vocab_size=32, pad_id=0, variant=variant, d_model=16,
        d_action=4, d_macro=3, macro_k=2, token_layers=1,
        sentence_layers=1, predictor_layers=1, n_heads=4,
        max_sequence_len=32, max_sentences=4, **kwargs,
    )


def test_whole_sequence_encoder_has_cross_sentence_context_and_attention_pooling():
    torch.manual_seed(3)
    encoder = HierarchicalBufferEncoder(
        32, 0, d_model=16, token_layers=1, sentence_layers=1,
        n_heads=4, max_sequence_len=32, max_sentences=4,
    )
    prompt = torch.tensor([[[2, 3, 0]]])
    buffer = torch.tensor([[[4, 5, 0], [6, 7, 0]]])
    tokens, mask, ids, sentences, sentence_mask, attention = encoder(prompt, buffer)
    assert tokens.shape == (1, 4, 16)
    assert ids.tolist() == [[0, 0, 1, 1]]
    assert sentence_mask.tolist() == [[True, True]]
    for sentence in range(2):
        assert torch.allclose(
            (attention * ids.eq(sentence)).sum(-1), torch.ones(1), atol=1e-6
        )
    # A sentence-1 state must depend on sentence-0 input embeddings.  This
    # catches the old encode-each-sentence-then-concatenate implementation.
    # LayerNorm makes the sum of features constant, so inspect one component.
    loss = sentences[0, 1, 0]
    grad = torch.autograd.grad(loss, encoder.tok.weight)[0]
    assert grad[4].abs().sum() > 0


def test_pointer_to_sentence_mapping_and_insert_boundary_are_mechanical():
    ids = torch.tensor([[0, 0, 1, 1]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    operations = torch.tensor([1])
    positions = torch.tensor([2])
    affected = MultiscaleEditJEPA.affected_sentences(
        ids, mask, operations, positions
    )
    transitioned, transitioned_mask, _ = MultiscaleEditJEPA.transition_sentence_ids(
        ids, mask, operations, positions
    )
    assert affected.item() == 1
    # The finite test buffer truncates one token, but inserted label and order
    # remain exact: gap 2 is owned by the sentence on its right.
    assert transitioned.tolist() == [[0, 0, 1, 1]]
    assert transitioned_mask.all()


@pytest.mark.parametrize("variant", [
    "token", "sentence", "token_sentence", "token_sentence_macro"
])
def test_all_four_variants_have_explicit_non_leaking_paths(variant):
    torch.manual_seed(5)
    model = _model(variant)
    out = model(_batch())
    assert out.preds.shape == (1, 2, 16)
    assert out.step_states_tgt.requires_grad is False
    assert model.teacher.training is False
    assert model.teacher.module.training is False
    if variant == "sentence":
        assert model.token_pred is None
        assert out.extras["token_predictions"] is None
    if variant == "token":
        assert model.sentence_pred is None
        assert out.extras["sentence_predictions"] is None
    if variant == "token_sentence_macro":
        assert out.extras["macro_codes"].shape == (1, 1, 3)
        assert out.extras["macro_window_starts"].tolist() == [0]
        assert out.extras["macro_window_endpoints"].tolist() == [2]
        assert out.extras["macro_sentence_targets"].shape == (1, 1, 2, 16)


def test_hybrid_sentence_prediction_really_depends_on_lower_prediction():
    torch.manual_seed(7)
    model = _model("token_sentence")
    out = model(_batch())
    loss = out.extras["sentence_predictions"].sum()
    loss.backward()
    # Sentence supervision must train the lower transition, establishing an
    # actual hierarchy instead of two side-by-side representations.
    assert model.token_pred.out.weight.grad is not None
    assert model.token_pred.out.weight.grad.abs().sum() > 0


def test_macro_code_uses_all_actions_and_preserves_order():
    torch.manual_seed(11)
    model = _model("token_sentence_macro")
    out = model(_batch())
    reversed_batch = copy.deepcopy(_batch())
    reversed_batch["op"] = reversed_batch["op"].flip(1)
    reversed_batch["edit_position"] = reversed_batch["edit_position"].flip(1)
    reversed_batch["edit_content_token"] = reversed_batch[
        "edit_content_token"
    ].flip(1)
    reversed_out = model(reversed_batch)
    assert not torch.allclose(
        out.extras["macro_codes"], reversed_out.extras["macro_codes"]
    )


def test_sentence_ldad_uses_changed_sentence_delta_and_reaches_encoder():
    torch.manual_seed(13)
    model = _model("sentence", observed_action_ldad=True)
    out = model(_batch())
    assert out.extras["ldad_uses_changed_sentence_delta"] is True
    out.extras["observed_action_logits"].sum().backward()
    assert model.encoder.pool_score[1].weight.grad is not None


def test_dropout_and_degenerate_macro_are_rejected():
    with pytest.raises(ValueError, match="dropout=0"):
        _model("token", dropout=0.1)
    with pytest.raises(ValueError, match="macro_k"):
        MultiscaleEditJEPA(
            32, 0, "token_sentence_macro", d_model=16, d_action=4,
            d_macro=3, macro_k=1, n_heads=4,
        )
