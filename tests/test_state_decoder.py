"""Tests for the frozen-state sentence read-out (decoder + answer head)."""

from __future__ import annotations

import torch

from scripts.eval_state_decoder import Accumulator, parse_value, strip
from scripts.train_state_decoder import batch_loss
from textjepa.models.state_decoder import FrozenStateSentenceDecoder


def _decoder(vocab_size: int = 12, d_state: int = 16, max_len: int = 6):
    torch.manual_seed(0)
    return FrozenStateSentenceDecoder(
        d_state=d_state, vocab_size=vocab_size, max_len=max_len,
        n_layers=2, n_heads=4, n_answers=5,
    )


def test_forward_and_backward_are_finite():
    decoder = _decoder()
    state = torch.randn(3, 2, 16)
    tokens = torch.randint(1, 12, (3, 2, 6))
    logits = decoder(state, tokens)
    assert logits.shape == (3, 2, 6, 12)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 12), tokens.reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_answer_head_shapes():
    decoder = _decoder()
    assert decoder.answer_logits(torch.randn(4, 16)).shape == (4, 5)


def test_no_gradient_reaches_a_frozen_backbone():
    """A frozen "backbone" producing the states must receive no gradient."""
    backbone = torch.nn.Linear(16, 16)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    decoder = _decoder()
    with torch.no_grad():
        states = backbone(torch.randn(4, 16))
    assert not states.requires_grad
    tokens = torch.randint(1, 12, (4, 6))
    loss = torch.nn.functional.cross_entropy(
        decoder(states, tokens).reshape(-1, 12), tokens.reshape(-1)
    ) + torch.nn.functional.cross_entropy(
        decoder.answer_logits(states), torch.zeros(4, dtype=torch.long)
    )
    loss.backward()
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None for p in decoder.parameters())


def test_generation_returns_valid_ids_and_stops_at_pad():
    decoder = _decoder()
    # Force PAD (id 0) to win at every position: generation must stop at once.
    with torch.no_grad():
        decoder.token_head.bias.zero_()
        decoder.token_head.bias[0] = 50.0
    out = decoder.generate(torch.randn(3, 16))
    assert out == [[], [], []]

    decoder = _decoder()
    out = decoder.generate(torch.randn(3, 16))
    assert len(out) == 3
    for row in out:
        assert len(row) <= decoder.max_len
        assert all(0 < token < 12 for token in row)  # PAD stripped, in range

    sampled = decoder.generate(
        torch.randn(2, 16), greedy=False, top_p=0.9,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(sampled) == 2


def test_training_smoke_loss_decreases_on_frozen_states():
    """Tiny CPU fit: the read-out must be able to learn from fixed states."""
    torch.manual_seed(0)
    pad_id = 0
    B, T, L, D = 4, 3, 6, 16
    states = torch.randn(B, T, D)
    assert not states.requires_grad
    tokens = torch.randint(1, 12, (B, T, L))
    tokens[..., -1] = pad_id
    mask = torch.ones(B, T, dtype=torch.bool)
    batch = {
        "prompt_tokens": torch.randint(1, 12, (B, 2, L)),
        "prompt_mask": torch.ones(B, 2, dtype=torch.bool),
        "step_tokens": tokens,
        "step_mask": mask,
        "answer": torch.randint(0, 5, (B,)),
    }

    class FixedBackbone(torch.nn.Module):
        """Stand-in for the frozen JEPA: returns pre-computed states."""

        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(1))
            self.unused.requires_grad_(False)

        def encode_states(self, *_args, **_kwargs):
            return states[:, -1], states

    model = FixedBackbone().eval()
    decoder = _decoder(max_len=L)
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        s_loss, a_loss, _, _ = batch_loss(model, decoder, batch, pad_id)
        loss = s_loss + a_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_value_parsing_and_metrics():
    assert parse_value("so the number of x is 3 plus 4 = 7 .") == "7"
    assert parse_value("so the number of x is 5 .") == "5"
    assert parse_value(".") is None

    class FakeVocab:
        pad_id = 0

        def decode(self, ids):
            table = {1: "so", 2: "=", 3: "7", 4: "9"}
            return " ".join(table[i] for i in ids)

    acc = Accumulator()
    vocab = FakeVocab()
    acc.add([1, 2, 3], [1, 2, 3], vocab)
    acc.add([1, 2, 4], [1, 2, 3], vocab)
    metrics = acc.as_dict()
    assert metrics["n"] == 2
    assert metrics["exact_match"] == 0.5
    assert metrics["value_acc"] == 0.5
    assert metrics["token_acc"] > 0.5


def test_strip_cuts_at_pad():
    assert strip(torch.tensor([4, 5, 0, 6]), 0) == [4, 5]
