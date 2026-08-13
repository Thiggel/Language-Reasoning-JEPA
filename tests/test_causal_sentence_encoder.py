import pytest
import torch

from textjepa.models.causal_sentence_encoder import CausalSentenceEncoder


def encoder(**kwargs):
    torch.manual_seed(0)
    return CausalSentenceEncoder(
        d_token=8, d_sentence=6, n_layers=1, n_heads=2, max_len=16, **kwargs
    ).eval()


def test_shapes():
    out = encoder()(torch.randn(3, 5, 8))
    assert out.shape == (3, 5, 6)


def test_output_is_causal_in_the_token_latent_history():
    """Position t must not see t+1: planning reads boundaries autoregressively."""
    model = encoder()
    states = torch.randn(1, 6, 8)
    before = model(states)
    perturbed = states.clone()
    perturbed[0, 4:] += 5.0
    after = model(perturbed)
    assert torch.allclose(before[0, :4], after[0, :4], atol=1e-5)
    assert not torch.allclose(before[0, 4], after[0, 4], atol=1e-5)


def test_state_depends_on_the_whole_step_not_only_the_boundary_token():
    """The reason this module exists: the MLP it replaces could not do this."""
    model = encoder()
    states = torch.randn(1, 5, 8)
    changed = states.clone()
    changed[0, 1] += 3.0  # an interior token, not the readout position
    assert not torch.allclose(model(states)[0, -1], model(changed)[0, -1], atol=1e-5)


def test_pointwise_call_is_rejected_rather_than_silently_mixing_items():
    with pytest.raises(ValueError, match="silently mix"):
        encoder()(torch.randn(4, 8))


def test_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match="width does not match"):
        encoder()(torch.randn(2, 3, 7))


def test_longer_than_trained_sequences_interpolate_positions():
    out = encoder()(torch.randn(1, 40, 8))  # max_len=16
    assert out.shape == (1, 40, 6)
    assert torch.isfinite(out).all()


def test_encode_last_matches_the_final_position():
    model = encoder()
    states = torch.randn(2, 7, 8)
    assert torch.allclose(model.encode_last(states), model(states)[:, -1], atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_layers": 0}, "sizes must be positive"),
        ({"n_heads": 3}, "divisible"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CausalSentenceEncoder(d_token=8, d_sentence=6, **kwargs)
