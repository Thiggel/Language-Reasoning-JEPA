import torch

from textjepa.models.layers import FlashMultiheadAttention


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
