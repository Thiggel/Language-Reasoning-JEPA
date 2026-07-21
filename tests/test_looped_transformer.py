import pytest
import torch

from textjepa.models.layers import LoopedTransformerEncoder
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM


def test_looped_encoder_reuses_one_block_and_eval_is_deterministic():
    encoder = LoopedTransformerEncoder(
        16, 4, 2, train_loop_mean=3, train_loop_min=1,
        train_loop_max=6, eval_loops=5,
    ).eval()
    x = torch.randn(2, 4, 16)
    first = encoder(x)
    second = encoder(x)
    assert encoder.last_num_loops == 5
    assert torch.equal(first, second)
    assert sum(p.numel() for p in encoder.parameters()) == sum(
        p.numel() for p in encoder.block.parameters()
    )


def test_training_loop_samples_are_bounded_and_seeded():
    encoder = LoopedTransformerEncoder(
        8, 2, 2, train_loop_mean=4, train_loop_min=2,
        train_loop_max=5, eval_loops=3,
    ).train()
    torch.manual_seed(7)
    first = [encoder.sample_num_loops() for _ in range(20)]
    torch.manual_seed(7)
    second = [encoder.sample_num_loops() for _ in range(20)]
    assert first == second
    assert all(2 <= value <= 5 for value in first)
    assert len(set(first)) > 1


def test_looped_encoder_rejects_dropout_and_bad_bounds():
    with pytest.raises(ValueError, match="dropout=0"):
        LoopedTransformerEncoder(8, 2, 2, dropout=0.1)
    with pytest.raises(ValueError, match="bounds"):
        LoopedTransformerEncoder(8, 2, 2, train_loop_min=0)


def test_token_lm_supports_train_and_test_loop_depths():
    model = DecoderLM(
        32, 0, d_model=16, n_layers=2, n_heads=4, max_len=12,
        recurrent=True, train_loop_mean=2, train_loop_max=4, eval_loops=2,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 0]])
    shallow = model(tokens, num_loops=1)
    deep = model(tokens, num_loops=4)
    assert shallow.shape == deep.shape == (1, 4, 32)
    assert not torch.allclose(shallow, deep)
    with pytest.raises(ValueError, match="positive"):
        model(tokens, num_loops=0)


def test_nonrecurrent_token_lm_rejects_loop_override():
    model = DecoderLM(
        16, 0, d_model=8, n_layers=1, n_heads=2, max_len=8
    )
    with pytest.raises(ValueError, match="recurrent"):
        model(torch.tensor([[1, 2]]), num_loops=2)


def test_sentence_lm_loops_only_the_reasoning_state_backbone():
    model = SentenceLM(
        24, 0, d_model=16, chunk_layers=1, chunk_heads=4,
        state_layers=3, state_heads=4, dec_layers=1, dec_heads=4,
        max_chunk_len=8, max_chunks=8, recurrent=True,
        train_loop_mean=2, train_loop_max=4, eval_loops=3,
    )
    assert isinstance(model.state_model.encoder, LoopedTransformerEncoder)
    assert not isinstance(model.chunk_encoder.encoder, LoopedTransformerEncoder)
    assert model.state_model.encoder.last_num_loops == 3
