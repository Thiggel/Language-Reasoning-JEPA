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


def test_subs_sampler_never_modifies_prompt_and_resolves_all_masks():
    torch.manual_seed(5)
    model = _model().eval()
    prompt = torch.tensor([[[2, 3, 0]]])
    shape = torch.tensor([[[4, 5, 0], [6, 0, 0]]])
    sampled, valid, response = model.sample(prompt, shape, steps=4)
    assert sampled[0, :2].tolist() == [2, 3]
    assert not sampled[response].eq(model.mask_id).any()
    assert sampled[valid].ne(model.pad_id).all()


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
