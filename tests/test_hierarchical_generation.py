from types import SimpleNamespace

import torch

from textjepa.utils.hierarchical_generation import (
    exact_ground_sentence_candidates,
    trim_reasoning_candidate,
)


def test_trim_accepts_a_short_genuine_newline_boundary():
    tokens, complete, terminal = trim_reasoning_candidate(
        [7, 10, 8], [10], maximum=64
    )
    assert tokens == [7, 10]
    assert complete is True
    assert terminal is False


def test_trim_excludes_eos_from_the_jepa_action():
    tokens, complete, terminal = trim_reasoning_candidate(
        [7, 248046, 9], [10], maximum=64
    )
    assert tokens == [7]
    assert complete is True
    assert terminal is True


class ExactFakeLM:
    def __call__(self, *, input_ids, attention_mask, **kwargs):
        batch, length = input_ids.shape
        vocab = 32
        hidden = input_ids.float()[..., None].repeat(1, 1, 3)
        logits = torch.zeros(batch, length, vocab)
        # Each position predicts the next integer token perfectly.
        predicted = (input_ids + 1).clamp_max(vocab - 1)
        logits.scatter_(-1, predicted[..., None], 10.0)
        return SimpleNamespace(hidden_states=[hidden], logits=logits)


def test_exact_grounding_uses_each_variable_length_endpoint():
    grounded = exact_ground_sentence_candidates(
        ExactFakeLM(), torch.tensor([1, 2]),
        [(torch.tensor([3, 4]), False), (torch.tensor([3]), True)],
    )
    assert grounded.lengths.tolist() == [2, 1]
    assert grounded.endpoint_hidden[:, 0].tolist() == [4.0, 3.0]
    assert grounded.mask.tolist() == [[True, True], [True, False]]
    assert grounded.terminal.tolist() == [False, True]
    assert bool((grounded.log_probabilities > -0.1).all())

