import copy
from types import SimpleNamespace

import pytest
import torch

from textjepa.models.multiscale_edit_jepa import (
    HierarchicalBufferEncoder,
    MultiscaleEditJEPA,
)
from textjepa.objectives.prediction import (
    BaseActionValue,
    MacroOptionReconstruction,
    MacroPriorDistillation,
    MacroSentencePrediction,
    SentenceLevelPrediction,
)
from textjepa.objectives.vicreg import MultiscaleVICReg, VISReg


def test_base_action_value_uses_mse_and_same_state_pairwise_ranking():
    target = torch.tensor([[1.0]])
    alternatives = torch.tensor([[[0.0, -1.0]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    common = {
        "base_action_value_target": target,
        "base_alt_action_target": alternatives,
        "base_alt_action_valid": valid,
    }
    correct = SimpleNamespace(
        preds=torch.zeros(1),
        step_mask=torch.ones(1, 1, dtype=torch.bool),
        extras={
            **common,
            "base_action_value": torch.tensor([[1.0]]),
            "base_alt_action_value": torch.tensor([[[0.0, -1.0]]]),
        },
    )
    reversed_order = SimpleNamespace(
        preds=torch.zeros(1),
        step_mask=correct.step_mask,
        extras={
            **common,
            "base_action_value": torch.tensor([[-1.0]]),
            "base_alt_action_value": torch.tensor([[[0.0, 1.0]]]),
        },
    )
    rank_only = BaseActionValue(
        regression_weight=0.0, pairwise_weight=1.0,
        regression_kind="mse", margin=0.5,
    )
    assert rank_only(correct, {}) < rank_only(reversed_order, {})

    mse_only = BaseActionValue(
        regression_weight=0.25, pairwise_weight=0.0,
        regression_kind="mse",
    )
    assert mse_only(correct, {}) == 0
    assert mse_only(reversed_order, {}) > 0


@pytest.mark.parametrize("variant", ["token", "sentence", "token_sentence"])
def test_gar_candidate_pool_runs_for_every_original_jepa_variant(variant):
    batch = _batch()
    batch.update({
        "gar_token_edit_target": torch.tensor([[1, 1]]),
        "goal_distance": torch.tensor([[2, 1, 0]]),
        "proposal_op": torch.full((1, 2, 3), 2),
        "proposal_edit_position": torch.tensor([[
            [0, 1, 3], [0, 2, 3],
        ]]),
        "proposal_edit_content_token": torch.tensor([[
            [8, 9, 10], [7, 8, 9],
        ]]),
        "proposal_valid": torch.ones(1, 2, 3, dtype=torch.bool),
        "gar_proposal_token_edit_target": torch.tensor([[
            [0, 1, -1], [1, 0, -1],
        ]]),
    })
    model = _model(variant)
    out = model(batch)
    assert out.extras["base_action_value"].shape == (1, 2)
    assert out.extras["base_alt_action_value"].shape == (1, 2, 3)
    loss = BaseActionValue(
        regression_kind="mse", regression_weight=0.25,
        pairwise_weight=1.0,
    )(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.base_q_head[-1].weight.grad is not None


def test_sentence_gar_primitive_q_encoder_avoids_blank_pattern_candidates():
    batch = _batch()
    batch.update({
        "gar_token_edit_target": torch.tensor([[1, 1]]),
        "proposal_op": torch.full((1, 2, 3), 2),
        "proposal_edit_position": torch.tensor([[
            [0, 1, 3], [0, 2, 3],
        ]]),
        "proposal_edit_content_token": torch.tensor([[
            [8, 9, 10], [7, 8, 9],
        ]]),
        "proposal_valid": torch.ones(1, 2, 3, dtype=torch.bool),
        "gar_proposal_token_edit_target": torch.tensor([[
            [0, 1, -1], [1, 0, -1],
        ]]),
    })
    model = _model(
        "sentence", sentence_action_kind="blank_pattern",
        base_q_primitive_action=True, base_q_hidden=8,
    )
    called = {"transition": 0}
    original = model.sentence_action.forward

    def counted(*args, **kwargs):
        called["transition"] += args[1].shape[0]
        return original(*args, **kwargs)

    model.sentence_action.forward = counted
    out = model(batch)
    # Only the two executed transition actions use the expensive pattern
    # encoder; three Q alternatives per state use the primitive Q encoder.
    assert called["transition"] == 2
    assert out.extras["base_alt_action_value"].shape == (1, 2, 3)
    assert model.base_q_head[1].out_features == 8


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


def test_attention_pooling_accepts_mixed_precision_exp_weights():
    encoder = HierarchicalBufferEncoder(
        32, 0, d_model=16, token_layers=1, sentence_layers=1,
        n_heads=4, max_sequence_len=32, max_sentences=4,
    )
    token_states = torch.randn(2, 5, 16)
    token_mask = torch.ones(2, 5, dtype=torch.bool)
    sentence_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pooled, sentence_mask, attention = encoder.pool_sentences(
            token_states, token_mask, sentence_ids, 4
        )
    assert torch.isfinite(pooled).all()
    assert sentence_mask[:, :2].all()
    for sentence in range(2):
        assert torch.allclose(
            (attention * sentence_ids.eq(sentence)).sum(1),
            attention.new_ones(2), atol=1e-2,
        )


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
    "token", "sentence", "sentence_macro", "token_sentence",
    "token_sentence_macro"
])
@pytest.mark.parametrize("sequence_packing", [False, True])
def test_all_four_variants_have_explicit_non_leaking_paths(
    variant, sequence_packing,
):
    torch.manual_seed(5)
    model = _model(
        variant,
        sequence_packing=sequence_packing,
        attention_backend="auto" if sequence_packing else "torch",
    )
    out = model(_batch())
    assert out.preds.shape == (1, 2, 16)
    assert out.step_states_tgt.requires_grad is False
    assert model.teacher.training is False
    assert model.teacher.module.training is False
    if variant in {"sentence", "sentence_macro"}:
        assert model.token_pred is None
        assert out.extras["token_predictions"] is None
    if variant == "token":
        assert model.sentence_pred is None
        assert out.extras["sentence_predictions"] is None
    if variant in {"sentence_macro", "token_sentence_macro"}:
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


def test_full_packed_model_matches_dense_model_and_backpropagates():
    torch.manual_seed(8)
    dense = _model(
        "token_sentence", attention_backend="auto", sequence_packing=False
    )
    packed = _model(
        "token_sentence", attention_backend="auto", sequence_packing=True
    )
    packed.load_state_dict(dense.state_dict())
    dense.eval()
    packed.eval()
    dense_out = dense(copy.deepcopy(_batch()))
    packed_out = packed(copy.deepcopy(_batch()))
    assert torch.allclose(
        packed_out.preds, dense_out.preds, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        packed_out.extras["token_predictions"],
        dense_out.extras["token_predictions"], atol=1e-5, rtol=1e-4,
    )
    packed.train()
    train_out = packed(copy.deepcopy(_batch()))
    train_out.preds.square().sum().backward()
    assert packed.encoder.tok.weight.grad is not None
    assert packed.encoder.tok.weight.grad.abs().sum() > 0


def test_efficient_pair_encoding_halves_state_encodes_without_changing_transition():
    torch.manual_seed(9)
    full = _model("token_sentence", efficient_pair_encoding=False)
    efficient = _model("token_sentence", efficient_pair_encoding=True)
    efficient.load_state_dict(full.state_dict())
    batch = copy.deepcopy(_batch())
    batch["buffer_tokens"] = batch["buffer_tokens"][:, :2]
    batch["buffer_mask"] = batch["buffer_mask"][:, :2]
    for key in ("op", "edit_position", "edit_content_token", "step_mask",
                "action_tokens"):
        batch[key] = batch[key][:, :1]
    encoded = {"full": 0, "efficient": 0}

    def count(name):
        def hook(_module, inputs):
            encoded[name] += inputs[0].shape[0]
        return hook

    full_hook = full.encoder.token_encoder.register_forward_pre_hook(count("full"))
    full_teacher_hook = full.teacher.module.token_encoder.register_forward_pre_hook(
        count("full")
    )
    efficient_hook = efficient.encoder.token_encoder.register_forward_pre_hook(
        count("efficient")
    )
    efficient_teacher_hook = (
        efficient.teacher.module.token_encoder.register_forward_pre_hook(
            count("efficient")
        )
    )
    full_out = full(copy.deepcopy(batch))
    efficient_out = efficient(copy.deepcopy(batch))
    for hook in (full_hook, full_teacher_hook, efficient_hook,
                 efficient_teacher_hook):
        hook.remove()
    assert encoded == {"full": 4, "efficient": 2}
    assert efficient_out.extras["efficient_pair_encoding"] is True
    assert torch.allclose(
        efficient_out.extras["token_predictions"],
        full_out.extras["token_predictions"], atol=1e-6,
    )
    assert torch.allclose(
        efficient_out.extras["sentence_predictions"],
        full_out.extras["sentence_predictions"], atol=1e-6,
    )
    assert torch.allclose(
        efficient_out.step_states_tgt, full_out.step_states_tgt, atol=1e-6,
    )


@pytest.mark.parametrize(
    "mode,target_requires_grad,has_teacher",
    [
        ("ema", False, True),
        ("shared_stopgrad", False, False),
        ("shared_symmetric", True, False),
    ],
)
@pytest.mark.parametrize("sequence_packing", [False, True])
def test_target_encoder_modes_have_explicit_gradient_and_teacher_semantics(
    mode, target_requires_grad, has_teacher, sequence_packing
):
    model = _model(
        "token", efficient_pair_encoding=True, target_encoder_mode=mode,
        sequence_packing=sequence_packing,
        attention_backend="auto" if sequence_packing else "torch",
    )
    batch = copy.deepcopy(_batch())
    batch["buffer_tokens"] = batch["buffer_tokens"][:, :2]
    batch["buffer_mask"] = batch["buffer_mask"][:, :2]
    for key in (
        "op", "edit_position", "edit_content_token", "step_mask",
        "action_tokens",
    ):
        batch[key] = batch[key][:, :1]
    out = model(batch)
    assert (model.teacher is not None) is has_teacher
    assert out.step_states_tgt.requires_grad is target_requires_grad
    assert out.extras["token_targets"].requires_grad is target_requires_grad
    assert out.extras["target_encoder_mode"] == mode
    model.update_teachers(0.99)  # all modes expose the trainer protocol


def test_visreg_is_finite_and_pushes_an_exactly_collapsed_batch_apart():
    states = torch.zeros(32, 1, 16, requires_grad=True)
    out = SimpleNamespace(
        extras={
            "sigreg_states": states,
            "sigreg_state_mask": torch.ones(32, 1, dtype=torch.bool),
        },
    )
    loss = VISReg(num_projections=32)(out, {})
    assert torch.isfinite(loss)
    loss.backward()
    assert states.grad is not None
    assert states.grad.abs().sum() > 0


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


def test_sentence_macro_is_a_clean_macro_isolation_without_token_dynamics():
    model = _model("sentence_macro")
    out = model(_batch())
    assert model.token_pred is None
    assert model.sentence_pred.correction is False
    assert out.extras["token_predictions"] is None
    assert out.extras["sentence_predictions"] is not None
    assert out.extras["macro_sentence_predictions"] is not None


def test_sentence_ldad_uses_changed_sentence_delta_and_reaches_encoder():
    torch.manual_seed(13)
    model = _model("sentence", observed_action_ldad=True)
    out = model(_batch())
    assert out.extras["ldad_uses_changed_sentence_delta"] is True
    out.extras["observed_action_logits"].sum().backward()
    assert model.encoder.pool_score[1].weight.grad is not None


def test_multiscale_objectives_are_finite_and_train_both_levels():
    torch.manual_seed(17)
    model = _model("token_sentence_macro")
    batch = _batch()
    out = model(batch)
    loss = (
        SentenceLevelPrediction()(out, batch)
        + MacroSentencePrediction()(out, batch)
        + 0.1 * MacroPriorDistillation()(out, batch)
        + 0.02 * MultiscaleVICReg()(out, batch)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.pool_score[1].weight.grad is not None
    assert model.token_pred.out.weight.grad is not None
    assert model.macro_model.prior[0].weight.grad is not None


def test_macro_prior_distillation_does_not_reshape_state_encoder_by_default():
    model = _model("sentence_macro")
    batch = _batch()
    out = model(batch)
    MacroPriorDistillation()(out, batch).backward()
    assert model.macro_model.prior[0].weight.grad is not None
    assert model.encoder.pool_score[1].weight.grad is None


def test_fixed_variance_macro_prior_is_nonnegative_and_can_shape_state():
    model = MultiscaleEditJEPA(
        32, 0, "sentence_macro", d_model=16, d_action=4,
        d_macro=3, macro_k=2, n_heads=4, token_layers=1,
        sentence_layers=1, predictor_layers=1, max_sequence_len=32,
        max_sentences=4, macro_prior_detach_state=False,
    )
    batch = _batch()
    loss = MacroPriorDistillation(kind="fixed_variance_mse")(
        model(batch), batch
    )
    loss.backward()
    assert loss.item() >= 0
    assert model.encoder.pool_score[1].weight.grad is not None


def test_macro_option_decoder_reconstructs_each_executable_primitive_step():
    model = MultiscaleEditJEPA(
        32, 0, "token_sentence_macro", d_model=16, d_action=4,
        d_macro=3, macro_k=2, n_heads=4, token_layers=1,
        sentence_layers=1, predictor_layers=1, max_sequence_len=32,
        max_sentences=4, macro_decoder=True,
    )
    batch = _batch()
    out = model(batch)
    assert out.extras["macro_decoder_position_logits"].shape == (1, 1, 2, 4)
    assert out.extras["macro_decoder_content_logits"].shape == (1, 1, 2, 32)
    loss = MacroOptionReconstruction()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.macro_decoder.position[1].weight.grad is not None
    # Detached decoder training does not redefine the macro representation.
    assert model.macro_model.encoder.out[-1].weight.grad is None


def test_attached_macro_option_decoder_can_shape_macro_representation():
    model = MultiscaleEditJEPA(
        32, 0, "token_sentence_macro", d_model=16, d_action=4,
        d_macro=3, macro_k=2, n_heads=4, token_layers=1,
        sentence_layers=1, predictor_layers=1, max_sequence_len=32,
        max_sentences=4, macro_decoder=True,
        macro_decoder_detach_inputs=False,
    )
    batch = _batch()
    MacroOptionReconstruction()(model(batch), batch).backward()
    assert model.macro_model.encoder.out[-1].weight.grad is not None


def test_dropout_and_degenerate_macro_are_rejected():
    with pytest.raises(ValueError, match="dropout=0"):
        _model("token", dropout=0.1)
    with pytest.raises(ValueError, match="macro_k"):
        MultiscaleEditJEPA(
            32, 0, "token_sentence_macro", d_model=16, d_action=4,
            d_macro=3, macro_k=1, n_heads=4,
        )
    with pytest.raises(ValueError, match="pooling"):
        HierarchicalBufferEncoder(32, 0, d_model=16, n_heads=4, pooling="max")


def test_mean_pooling_control_is_uniform_within_each_sentence():
    encoder = HierarchicalBufferEncoder(
        32, 0, d_model=16, token_layers=1, sentence_layers=1,
        n_heads=4, max_sequence_len=32, max_sentences=4, pooling="mean",
    )
    result = encoder(
        torch.tensor([[[2, 3, 0]]]),
        torch.tensor([[[4, 5, 0], [6, 7, 8]]]),
    )
    ids, attention = result[2], result[-1]
    assert torch.allclose(attention[ids.eq(0)], torch.full((2,), 0.5))
    assert torch.allclose(attention[ids.eq(1)], torch.full((3,), 1 / 3))


def test_long_trajectory_is_sliced_consistently_and_keeps_macro_windows_exact():
    batch = _batch()
    # Repeat the two exact replacement transitions to make six steps.
    batch["buffer_tokens"] = torch.cat([
        batch["buffer_tokens"], batch["buffer_tokens"][:, 1:],
        batch["buffer_tokens"][:, 1:],
    ], 1)
    for key in ("op", "edit_position", "edit_content_token", "step_mask",
                "action_tokens"):
        batch[key] = batch[key].repeat(1, 3, *([1] if batch[key].ndim == 3 else []))
    model = MultiscaleEditJEPA(
        32, 0, "token_sentence_macro", d_model=16, d_action=4,
        d_macro=3, macro_k=2, n_heads=4, token_layers=1,
        sentence_layers=1, predictor_layers=1, max_sequence_len=32,
        max_sentences=4, max_transitions_per_forward=3,
    ).eval()
    out = model(batch)
    assert out.step_mask.shape == (1, 3)
    assert batch["op"].shape == (1, 3)
    assert batch["buffer_tokens"].shape[1] == 4
    assert out.extras["observed_action_targets"].shape[1] == 3
    assert out.extras["macro_window_starts"].tolist() == [0, 1]
    assert out.extras["macro_window_endpoints"].tolist() == [2, 3]
