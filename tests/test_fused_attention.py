import torch

from textjepa.models.layers import (
    FlashMultiheadAttention, encoder_stack, packed_encoder_forward,
)


def test_torch_sdpa_attention_matches_multihead_attention_with_padding():
    torch.manual_seed(41)
    reference = torch.nn.MultiheadAttention(32, 4, batch_first=True, dropout=0)
    fused = FlashMultiheadAttention(
        32, 4, batch_first=True, dropout=0, attention_backend="torch"
    )
    fused.load_state_dict(reference.state_dict())
    value = torch.randn(3, 7, 32, requires_grad=True)
    other = value.detach().clone().requires_grad_(True)
    padding = torch.tensor([
        [False, False, False, False, False, False, False],
        [False, False, False, False, True, True, True],
        [False, False, True, True, True, True, True],
    ])
    expected = reference(
        value, value, value, key_padding_mask=padding, need_weights=False
    )[0]
    actual = fused(
        other, other, other, key_padding_mask=padding, need_weights=False
    )[0]
    # Padded query outputs are deliberately zeroed by the fused path and are
    # never consumed by TextJEPA. Valid-token values and gradients must match.
    valid = ~padding
    assert torch.allclose(actual[valid], expected[valid], atol=2e-6, rtol=2e-5)
    expected[valid].square().sum().backward()
    actual[valid].square().sum().backward()
    assert torch.allclose(other.grad[valid], value.grad[valid], atol=2e-6, rtol=2e-5)


def test_fused_attention_accepts_transformer_canonical_float_padding_mask():
    attention = FlashMultiheadAttention(
        32, 4, batch_first=True, dropout=0, attention_backend="torch"
    )
    value = torch.randn(2, 5, 32)
    padding = torch.tensor([
        [0.0, 0.0, 0.0, float("-inf"), float("-inf")],
        [0.0, 0.0, 0.0, 0.0, float("-inf")],
    ])
    output = attention(
        value, value, value, key_padding_mask=padding, need_weights=False
    )[0]
    assert torch.isfinite(output).all()
    assert output[0, 3:].eq(0).all()
    assert output[1, 4:].eq(0).all()


def test_packed_attention_matches_independent_padded_attention_and_gradients():
    torch.manual_seed(43)
    attention = FlashMultiheadAttention(
        32, 4, batch_first=True, dropout=0, attention_backend="torch"
    )
    dense = torch.randn(3, 7, 32, requires_grad=True)
    packed_source = dense.detach().clone().requires_grad_(True)
    valid = torch.tensor([
        [True] * 7,
        [True] * 4 + [False] * 3,
        [True] * 2 + [False] * 5,
    ])
    expected = attention(
        dense, dense, dense, key_padding_mask=~valid, need_weights=False
    )[0]
    lengths = valid.sum(1, dtype=torch.int32)
    cu = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
    actual_flat = attention.forward_packed(
        packed_source[valid], cu, int(lengths.max())
    )
    assert torch.allclose(actual_flat, expected[valid], atol=2e-6, rtol=2e-5)
    expected[valid].square().sum().backward()
    actual_flat.square().sum().backward()
    assert torch.allclose(
        packed_source.grad[valid], dense.grad[valid], atol=3e-6, rtol=3e-5
    )


def test_packed_encoder_matches_dense_encoder_without_cross_example_leakage():
    torch.manual_seed(47)
    encoder = encoder_stack(
        32, 2, 4, 2, 0.0, attention_backend="auto"
    )
    encoder.eval()
    value = torch.randn(3, 9, 32)
    valid = torch.tensor([
        [True] * 9,
        [True] * 5 + [False] * 4,
        [True] * 2 + [False] * 7,
    ])
    expected = encoder(value, src_key_padding_mask=~valid)
    actual = packed_encoder_forward(encoder, value, valid)
    assert torch.allclose(actual[valid], expected[valid], atol=5e-6, rtol=5e-5)
    assert actual[~valid].eq(0).all()

    changed = value.clone()
    changed[0] += 1000
    changed_actual = packed_encoder_forward(encoder, changed, valid)
    assert torch.allclose(
        changed_actual[1:][valid[1:]], actual[1:][valid[1:]],
        atol=5e-6, rtol=5e-5,
    )
