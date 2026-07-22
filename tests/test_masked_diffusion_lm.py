from types import MethodType

import torch

from textjepa.models.masked_diffusion_lm import (
    MaskedDiffusionLM,
    select_terminal_buffers,
)


def _model():
    return MaskedDiffusionLM(
        32, pad_id=0, mask_id=1, d_model=16, n_layers=1,
        n_heads=4, max_sequence_len=32,
    )


def test_pack_keeps_prompt_out_of_diffusion_domain():
    model = _model()
    clean, valid, response = model.pack_clean(
        torch.tensor([[[2, 3, 0]]]),
        torch.tensor([[[4, 5, 0], [6, 0, 0]]]),
    )
    assert clean.tolist() == [[2, 3, 4, 5, 6]]
    assert valid.all()
    assert response.tolist() == [[False, False, True, True, True]]


def test_vectorized_pack_handles_empty_sentences_and_visible_boundaries():
    model = MaskedDiffusionLM(
        32, pad_id=0, mask_id=1, boundary_id=31, d_model=16,
        n_layers=1, n_heads=4, max_sequence_len=32,
    )
    clean, valid, response = model.pack_clean(
        torch.tensor([[[2, 3, 0]], [[7, 0, 0]]]),
        torch.tensor([
            [[4, 5, 0], [0, 0, 0], [6, 0, 0]],
            [[8, 0, 0], [9, 10, 0], [0, 0, 0]],
        ]),
    )
    assert clean.tolist() == [
        [2, 3, 4, 5, 31, 6],
        [7, 8, 31, 9, 10, 0],
    ]
    assert valid.tolist() == [
        [True, True, True, True, True, True],
        [True, True, True, True, True, False],
    ]
    assert response.tolist() == [
        [False, False, True, True, False, True],
        [False, True, False, True, True, False],
    ]


def test_absorbing_corruption_never_changes_prompt_or_uses_random_tokens():
    model = _model()
    clean = torch.tensor([[2, 3, 4, 5]])
    response = torch.tensor([[False, False, True, True]])
    noised, masked = model.corrupt(
        clean, response, torch.tensor([0.5]),
        random=torch.tensor([[0.0, 0.0, 0.1, 0.9]]),
    )
    assert noised.tolist() == [[2, 3, 1, 5]]
    assert masked.tolist() == [[False, False, True, False]]


def test_mdlm_elbo_is_finite_and_backpropagates():
    torch.manual_seed(3)
    model = _model()
    clean, valid, response = model.pack_clean(
        torch.tensor([[[2, 3, 0]], [[7, 0, 0]]]),
        torch.tensor([[[4, 5, 0]], [[8, 9, 10]]]),
    )
    loss, extra = model.mdlm_loss(
        clean, valid, response, torch.tensor([1.0, 1.0])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert extra["masked"].equal(response)
    assert model.output.weight.grad.abs().sum() > 0


def test_sparse_mdlm_head_matches_dense_reference_loss_and_gradient():
    torch.manual_seed(19)
    sparse = _model()
    dense = _model()
    dense.load_state_dict(sparse.state_dict())
    clean, valid, response = sparse.pack_clean(
        torch.tensor([[[2, 3, 0]], [[7, 0, 0]]]),
        torch.tensor([[[4, 5, 0]], [[8, 9, 10]]]),
    )
    noise = torch.tensor([0.4, 0.8])
    draw = torch.tensor([
        [0.9, 0.9, 0.1, 0.7],
        [0.9, 0.2, 0.1, 0.7],
    ])
    noised, masked = sparse.corrupt(clean, response, noise, random=draw)
    states = dense.states(noised, valid, response)
    logits = dense.output(states)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), clean.reshape(-1),
        reduction="none",
    ).reshape_as(clean)
    expected = (
        ce * masked.to(ce.dtype) / noise[:, None]
    ).sum() / response.sum()

    # Reproduce the identical corruption inside mdlm_loss.
    original_corrupt = sparse.corrupt
    sparse.corrupt = lambda clean, response, noise: (noised, masked)
    actual, extra = sparse.mdlm_loss(clean, valid, response, noise)
    sparse.corrupt = original_corrupt
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert extra["masked_logits"].shape == (int(masked.sum()), sparse.vocab_size)

    actual.backward()
    expected.backward()
    assert torch.allclose(
        sparse.token.weight.grad, dense.token.weight.grad,
        atol=2e-6, rtol=2e-5,
    )


def test_sparse_mdlm_loss_handles_zero_sampled_masks():
    model = _model()
    clean, valid, response = model.pack_clean(
        torch.tensor([[[2, 3, 0]]]), torch.tensor([[[4, 5, 0]]])
    )
    original_corrupt = model.corrupt
    model.corrupt = lambda clean, response, noise: (
        clean, torch.zeros_like(response)
    )
    loss, extra = model.mdlm_loss(clean, valid, response, torch.tensor([0.1]))
    model.corrupt = original_corrupt
    assert loss.item() == 0.0
    assert extra["masked_logits"].shape == (0, model.vocab_size)
    loss.backward()


def test_packed_mdlm_matches_dense_states_loss_and_gradients():
    torch.manual_seed(29)
    dense = MaskedDiffusionLM(
        32, 0, 1, d_model=16, n_layers=2, n_heads=4,
        max_sequence_len=32, attention_backend="auto",
        sequence_packing=False,
    )
    packed = MaskedDiffusionLM(
        32, 0, 1, d_model=16, n_layers=2, n_heads=4,
        max_sequence_len=32, attention_backend="auto",
        sequence_packing=True,
    )
    packed.load_state_dict(dense.state_dict())
    clean = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 0]])
    valid = clean.ne(0)
    response = torch.tensor([
        [False, False, True, True],
        [False, True, True, False],
    ])
    noised = clean.masked_fill(response, 1)
    dense_states = dense.states(noised, valid, response)
    packed_states = packed.states(noised, valid, response)
    assert torch.allclose(
        packed_states[valid], dense_states[valid], atol=2e-6, rtol=2e-5
    )

    masked = response.clone()
    dense.corrupt = lambda clean, response, noise: (noised, masked)
    packed.corrupt = lambda clean, response, noise: (noised, masked)
    noise = torch.tensor([1.0, 1.0])
    dense_loss, _ = dense.mdlm_loss(clean, valid, response, noise)
    packed_loss, _ = packed.mdlm_loss(clean, valid, response, noise)
    assert torch.allclose(packed_loss, dense_loss, atol=2e-6, rtol=2e-5)
    dense_loss.backward()
    packed_loss.backward()
    assert torch.allclose(
        packed.token.weight.grad, dense.token.weight.grad,
        atol=3e-6, rtol=3e-5,
    )


def test_subs_sampler_never_modifies_prompt_and_resolves_all_masks():
    torch.manual_seed(5)
    model = _model().eval()
    prompt = torch.tensor([[[2, 3, 0]]])
    shape = torch.tensor([[[4, 5, 0], [6, 0, 0]]])
    sampled, valid, response = model.sample(prompt, shape, steps=4)
    assert sampled[0, :2].tolist() == [2, 3]
    assert not sampled[response].eq(model.mask_id).any()
    assert sampled[valid].ne(model.pad_id).all()


def test_confidence_sampler_reveals_exactly_highest_confidence_position():
    model = _model().eval()

    def controlled_logits(self, tokens, valid, response):
        logits = torch.zeros(*tokens.shape, self.vocab_size)
        logits[:, 2, 7] = 8.0
        logits[:, 3, 9] = 4.0
        return logits, torch.zeros(*tokens.shape, 16)

    model.logits = MethodType(controlled_logits, model)
    sampled, _, response = model.sample(
        torch.tensor([[[2, 3, 0]]]),
        torch.tensor([[[4, 5, 0]]]),
        steps=1,
        schedule="confidence",
    )
    assert sampled[0, :2].tolist() == [2, 3]
    assert sampled[0, 2].item() == 7
    assert sampled[0, 3].item() == model.mask_id
    assert sampled[response].ne(model.mask_id).sum().item() == 1


def test_sampler_rejects_unknown_schedule():
    model = _model().eval()
    try:
        model.sample(
            torch.tensor([[[2, 3, 0]]]),
            torch.tensor([[[4, 5, 0]]]),
            schedule="not-a-schedule",
        )
    except ValueError as error:
        assert "unknown unmasking schedule" in str(error)
    else:
        raise AssertionError("unknown schedules must be rejected")


def test_dropout_is_rejected_for_matched_comparison():
    try:
        MaskedDiffusionLM(32, 0, 1, dropout=0.1)
    except ValueError as error:
        assert "dropout=0" in str(error)
    else:
        raise AssertionError("dropout must be rejected")


def test_terminal_buffer_selection_uses_each_examples_step_count():
    buffers = torch.tensor([
        [[[10]], [[11]], [[12]], [[0]]],
        [[[20]], [[21]], [[22]], [[23]]],
    ])
    step_mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    terminal = select_terminal_buffers(buffers, step_mask)
    assert terminal[:, 0, 0].tolist() == [12, 23]
