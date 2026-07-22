import copy

import pytest
import torch

from textjepa.models.layers import encoder_stack, packed_encoder_forward


def _prefix_mask(lengths, width):
    return torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


@pytest.mark.parametrize(
    "dimension,heads,lengths",
    [
        (16, 4, [1]),
        (32, 4, [9, 5, 2]),
        (40, 4, [1, 7, 3, 11]),  # non-power-of-two head width 10
        (56, 8, [13, 1, 8]),     # non-power-of-two head width 7
    ],
)
def test_packed_encoder_matches_dense_forward_input_and_parameter_gradients(
    dimension, heads, lengths,
):
    torch.manual_seed(101 + dimension)
    width = max(lengths)
    valid = _prefix_mask(lengths, width)
    dense_model = encoder_stack(
        dimension, 2, heads, 2.25, 0.0, attention_backend="auto"
    )
    packed_model = copy.deepcopy(dense_model)
    dense_input = torch.randn(len(lengths), width, dimension, requires_grad=True)
    packed_input = dense_input.detach().clone().requires_grad_(True)
    dense = dense_model(dense_input, src_key_padding_mask=~valid)
    packed = packed_encoder_forward(packed_model, packed_input, valid)
    assert torch.allclose(packed[valid], dense[valid], atol=8e-6, rtol=8e-5)
    assert packed[~valid].eq(0).all()

    weight = torch.randn_like(dense) * valid.unsqueeze(-1)
    (dense * weight).sum().backward()
    (packed * weight).sum().backward()
    assert torch.allclose(
        packed_input.grad[valid], dense_input.grad[valid], atol=1e-5, rtol=1e-4
    )
    for (dense_name, dense_parameter), (packed_name, packed_parameter) in zip(
        dense_model.named_parameters(), packed_model.named_parameters()
    ):
        assert dense_name == packed_name
        assert dense_parameter.grad is not None, dense_name
        assert packed_parameter.grad is not None, packed_name
        assert torch.allclose(
            packed_parameter.grad, dense_parameter.grad,
            atol=2e-5, rtol=2e-4,
        ), dense_name


def test_packed_encoder_matches_running_every_example_independently():
    torch.manual_seed(113)
    encoder = encoder_stack(32, 3, 4, 2, 0.0, attention_backend="auto")
    lengths = [12, 1, 7, 3]
    value = torch.randn(4, 12, 32)
    valid = _prefix_mask(lengths, 12)
    packed = packed_encoder_forward(encoder, value, valid)
    for row, length in enumerate(lengths):
        independent = encoder(value[row:row + 1, :length])
        assert torch.allclose(
            packed[row, :length], independent[0], atol=8e-6, rtol=8e-5
        )


def test_packed_encoder_supports_noncontiguous_valid_tokens_exactly():
    torch.manual_seed(127)
    encoder = encoder_stack(24, 2, 4, 2, 0.0, attention_backend="auto")
    value = torch.randn(3, 8, 24)
    valid = torch.tensor([
        [True, False, True, True, False, True, False, False],
        [False, True, False, False, True, False, True, True],
        [True, False, False, False, False, False, False, False],
    ])
    dense = encoder(value, src_key_padding_mask=~valid)
    packed = packed_encoder_forward(encoder, value, valid)
    assert torch.allclose(packed[valid], dense[valid], atol=8e-6, rtol=8e-5)
    assert packed[~valid].eq(0).all()


def test_packed_batch_permutation_and_position_reset_are_exact():
    torch.manual_seed(131)
    encoder = encoder_stack(32, 2, 4, 2, 0.0, attention_backend="auto")
    value = torch.randn(4, 9, 32)
    value[3, :5] = value[1, :5]
    valid = _prefix_mask([9, 5, 2, 5], 9)
    original = packed_encoder_forward(encoder, value, valid)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = packed_encoder_forward(
        encoder, value[permutation], valid[permutation]
    )
    inverse = permutation.argsort()
    assert torch.equal(permuted[inverse], original)
    assert torch.equal(original[1, :5], original[3, :5])


def test_packed_outputs_have_zero_cross_example_input_gradient():
    torch.manual_seed(137)
    encoder = encoder_stack(32, 2, 4, 2, 0.0, attention_backend="auto")
    value = torch.randn(3, 7, 32, requires_grad=True)
    valid = _prefix_mask([7, 4, 2], 7)
    packed = packed_encoder_forward(encoder, value, valid)
    gradient = torch.autograd.grad(packed[1, :4].square().sum(), value)[0]
    assert gradient[0].eq(0).all()
    assert gradient[2].eq(0).all()
    assert gradient[1, 4:].eq(0).all()
    assert gradient[1, :4].abs().sum() > 0


def test_production_head_width_matches_dense_on_cpu_reference():
    torch.manual_seed(149)
    encoder = encoder_stack(832, 1, 8, 1.0, 0.0, attention_backend="auto")
    value = torch.randn(3, 6, 832)
    valid = _prefix_mask([6, 3, 1], 6)
    dense = encoder(value, src_key_padding_mask=~valid)
    packed = packed_encoder_forward(encoder, value, valid)
    assert torch.allclose(packed[valid], dense[valid], atol=1e-5, rtol=1e-4)


def test_packed_encoder_bfloat16_autocast_is_finite_and_close():
    torch.manual_seed(151)
    encoder = encoder_stack(32, 2, 4, 2, 0.0, attention_backend="auto")
    value = torch.randn(3, 9, 32)
    valid = _prefix_mask([9, 4, 1], 9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dense = encoder(value, src_key_padding_mask=~valid)
        packed = packed_encoder_forward(encoder, value, valid)
    assert torch.isfinite(packed).all()
    assert torch.allclose(packed[valid], dense[valid], atol=2e-2, rtol=2e-2)


def test_packed_encoder_rejects_unsafe_or_malformed_inputs():
    encoder = encoder_stack(16, 1, 4, 2, 0.0, attention_backend="auto")
    value = torch.randn(2, 4, 16)
    with pytest.raises(ValueError, match="empty sequences"):
        packed_encoder_forward(
            encoder, value, torch.tensor([
                [True, False, False, False],
                [False, False, False, False],
            ])
        )
    with pytest.raises(ValueError, match="expects"):
        packed_encoder_forward(encoder, value, torch.ones(2, 3, dtype=torch.bool))

    post_norm = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            16, 4, 32, dropout=0, batch_first=True, norm_first=False
        ),
        1,
    )
    with pytest.raises(ValueError, match="norm_first"):
        packed_encoder_forward(
            post_norm, value, torch.ones(2, 4, dtype=torch.bool)
        )


def test_packing_keeps_parameter_names_and_checkpoint_values_unchanged():
    dense = encoder_stack(32, 2, 4, 2, 0.0, attention_backend="auto")
    packed = copy.deepcopy(dense)
    assert dense.state_dict().keys() == packed.state_dict().keys()
    for name, value in dense.state_dict().items():
        assert torch.equal(value, packed.state_dict()[name]), name


def test_randomized_masks_and_shapes_match_dense_reference():
    # Deterministic property-style coverage of many padding patterns, including
    # interior holes that production does not normally generate.
    for seed in range(20):
        generator = torch.Generator().manual_seed(2000 + seed)
        batch = 1 + seed % 6
        width = 1 + (seed * 7) % 19
        dimension = (16, 24, 32, 40)[seed % 4]
        heads = 4
        encoder = encoder_stack(
            dimension, 1 + seed % 3, heads, 1.5, 0.0,
            attention_backend="auto",
        )
        value = torch.randn(batch, width, dimension, generator=generator)
        valid = torch.rand(batch, width, generator=generator).gt(0.35)
        valid[:, 0] = True
        dense = encoder(value, src_key_padding_mask=~valid)
        packed = packed_encoder_forward(encoder, value, valid)
        assert torch.allclose(
            packed[valid], dense[valid], atol=1e-5, rtol=1e-4
        ), seed
        assert packed[~valid].eq(0).all(), seed
