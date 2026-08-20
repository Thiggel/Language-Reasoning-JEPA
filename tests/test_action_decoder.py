"""Contract tests for the context-conditioned detached action decoder."""

import torch

from textjepa.planning.action_decoder import ContextActionDecoder


def _mk(**kw):
    torch.manual_seed(0)
    return ContextActionDecoder(
        d_state=32, vocab_size=17, max_len=6, d_model=24, n_layers=2,
        n_heads=4, **kw
    )


def _inputs(B=3, L=7):
    torch.manual_seed(1)
    return (torch.randn(B, 32), torch.randn(B, L, 32),
            torch.zeros(B, L, dtype=torch.bool), torch.randint(1, 17, (B, 6)))


def test_forward_shapes():
    dec = _mk()
    a, c, m, t = _inputs()
    assert dec(a, c, m, t).shape == (3, 6, 17)


def test_action_actually_changes_the_output():
    """The gate is meaningless if the action vector is inert."""
    dec = _mk()
    a, c, m, t = _inputs()
    assert not torch.allclose(dec(a, c, m, t), dec(a.flip(0), c, m, t))


def test_no_action_arm_is_exactly_invariant_to_the_action():
    dec = _mk(use_action=False)
    a, c, m, t = _inputs()
    assert torch.equal(dec(a, c, m, t), dec(torch.randn_like(a), c, m, t))


def test_no_context_arm_is_exactly_invariant_to_the_context():
    dec = _mk(use_context=False)
    a, c, m, t = _inputs()
    assert torch.equal(dec(a, c, m, t), dec(a, torch.randn_like(c), m, t))


def test_context_actually_changes_the_output():
    dec = _mk()
    a, c, m, t = _inputs()
    assert not torch.allclose(dec(a, c, m, t), dec(a, torch.randn_like(c), m, t))


def test_padding_mask_hides_padded_context():
    dec = _mk()
    a, c, m, t = _inputs()
    m2 = m.clone()
    m2[:, 4:] = True
    c2 = c.clone()
    c2[:, 4:] = 99.0  # garbage behind the mask must not leak
    assert torch.allclose(dec(a, c2, m2, t), dec(a, c, m2, t), atol=1e-5)


def test_generate_stops_at_eos_and_respects_cap():
    dec = _mk()
    a, c, m, _ = _inputs()
    out = dec.generate(a, c, m, eos_id=0)
    assert len(out) == 3
    for seq in out:
        assert len(seq) <= dec.max_len
        assert 0 not in seq  # EOS terminates and is never emitted


def test_no_weight_tying_between_embedding_and_output_head():
    """The intent_prior_lm detach round's trap: a tied output matrix would
    put the decoding gradient back into a shared table."""
    dec = _mk()
    assert dec.tok.weight.data_ptr() != dec.head.weight.data_ptr()
    ptrs = [p.data_ptr() for p in dec.parameters()]
    assert len(ptrs) == len(set(ptrs))


def test_save_load_roundtrip(tmp_path):
    dec = _mk()
    a, c, m, t = _inputs()
    path = tmp_path / "dec.pt"
    dec.save(str(path))
    dec2 = ContextActionDecoder.load(str(path))
    assert torch.equal(dec(a, c, m, t), dec2(a, c, m, t))
    assert dec2.use_action and dec2.use_context
