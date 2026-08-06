import torch

from textjepa.models.state_model import (
    CausalSentenceStateModel,
    DiscourseStateModel,
)


def test_discourse_state_model_interpolates_positions_beyond_training_length():
    model = DiscourseStateModel(
        d_model=8, n_layers=1, n_heads=2, ff_mult=2, max_chunks=4
    )
    prompt = torch.randn(2, 2, 8)
    steps = torch.randn(2, 3, 8)
    prompt_mask = torch.ones(2, 2, dtype=torch.bool)
    step_mask = torch.ones(2, 3, dtype=torch.bool)

    initial, states = model(prompt, prompt_mask, steps, step_mask)

    assert initial.shape == (2, 8)
    assert states.shape == (2, 3, 8)
    assert torch.isfinite(initial).all()
    assert torch.isfinite(states).all()


def test_sentence_state_model_interpolates_positions_beyond_training_length():
    model = CausalSentenceStateModel(
        d_model=8, n_layers=1, n_heads=2, ff_mult=2, max_chunks=4
    )
    sentences = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    states = model(sentences, mask)

    assert states.shape == (2, 5, 8)
    assert torch.isfinite(states).all()
