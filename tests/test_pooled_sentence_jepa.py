from functools import partial

import torch
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.pooled_sentence_jepa import (
    CausalAttentionPooler, PooledSentenceJEPA,
)


def _batch(size=2):
    vocab = build_vocab(23)
    ds = SemanticBoundaryLMDataset(
        vocab, size=size, seed=71, boundary_mode="semantic", modulus=23,
        n_vars_range=(8, 10), leaf_prob=0.35, steps_range=(4, 6),
        distractor_prob=0.0, max_distractors=0,
    )
    return next(iter(DataLoader(
        ds, batch_size=size,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    ))), vocab


def _model(vocab, scope="sentence", decoder=True):
    return PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], d_state=32, encoder_layers=1,
        pool_heads=4, predictor_layers=1, n_heads=4, ff_mult=2,
        max_len=768, d_action=8, dense_depth=2, pooling_scope=scope,
        use_token_prior=True, use_prefix_decoder=decoder,
        decoder_dim=24, decoder_layers=1, decoder_heads=4,
        decoder_max_len=64, decoder_prefixes_per_sequence=3,
    )


def test_pooler_sentence_mask_excludes_previous_segment_but_global_does_not():
    torch.manual_seed(1)
    hidden = torch.randn(1, 6, 16)
    tokens = torch.tensor([[2, 3, 10, 4, 5, 6]])
    changed = hidden.clone()
    changed[:, :3] += 100
    sentence = CausalAttentionPooler(16, 4, "sentence", (10, 15)).eval()
    global_pool = CausalAttentionPooler(16, 4, "global", (10, 15)).eval()
    with torch.no_grad():
        local_a = sentence(hidden, tokens, pad_id=0)
        local_b = sentence(changed, tokens, pad_id=0)
        global_a = global_pool(hidden, tokens, pad_id=0)
        global_b = global_pool(changed, tokens, pad_id=0)
    assert torch.allclose(local_a[:, 3:], local_b[:, 3:], atol=1e-5)
    assert not torch.allclose(global_a[:, 3:], global_b[:, 3:])


def test_pooled_states_and_targets_are_strictly_causal():
    batch, vocab = _batch()
    model = _model(vocab).eval()
    with torch.no_grad():
        original = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    changed = batch["tokens"].clone()
    cut = int(batch["prompt_len"][0]) + 3
    changed[0, cut:] = torch.randint(1, len(vocab), changed[0, cut:].shape)
    with torch.no_grad():
        other = model(changed, batch["prompt_len"], batch["sentence_ends"])
    assert torch.allclose(original["states"][0, :cut], other["states"][0, :cut], atol=1e-5)
    assert torch.allclose(original["target"][0, :3], other["target"][0, :3], atol=1e-5)


def test_next_token_action_predicts_next_pooled_prefix_state():
    batch, vocab = _batch()
    model = _model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    for row in range(len(batch["tokens"])):
        prompt = int(batch["prompt_len"][row])
        count = int(out["lengths"][row])
        assert torch.allclose(out["prev"][row, 0], out["states"][row, prompt - 1])
        assert torch.allclose(
            out["target"][row, :count], out["targets"][row, prompt:prompt + count]
        )
    assert model.predictor.causal_sequence
    assert model.predictor.residual


def test_prefix_decoder_is_causal_conditioned_and_reaches_pooler():
    batch, vocab = _batch()
    model = _model(vocab, decoder=True).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    decoded = model.prefix_decoder_batch(
        out, batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
    )
    assert decoded["logits"].shape[:2] == decoded["targets"].shape
    assert decoded["valid"].any()
    assert not torch.allclose(decoded["logits"], decoded["shuffled_logits"])
    loss = torch.nn.functional.cross_entropy(
        decoded["logits"][decoded["valid"]], decoded["targets"][decoded["valid"]]
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.state_encoder.pooler.parameters()
    )


def test_prefix_decoder_cannot_see_future_teacher_forced_tokens():
    batch, vocab = _batch()
    model = _model(vocab, decoder=True).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
        decoded = model.prefix_decoder_batch(
            out, batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
        )
        changed = decoded["targets"].clone()
        changed[:, -1] = (changed[:, -1] + 1) % len(vocab)
        other = model.prefix_decoder(
            decoded["states"], changed, decoded["valid"]
        )
    # The last target is shifted into no earlier decoder input.
    assert torch.allclose(decoded["logits"][:, :-1], other[:, :-1], atol=1e-6)


def test_all_trainable_transformers_are_dropout_free():
    _, vocab = _batch(1)
    model = _model(vocab, decoder=True)
    dropouts = [module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropouts and max(dropouts) == 0.0


def test_default_model_is_approximately_fifty_million_parameters():
    _, vocab = _batch(1)
    for decoder in (False, True):
        model = PooledSentenceJEPA(
            len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
            question_id=vocab.token_to_id["?"], use_prefix_decoder=decoder,
        )
        # The frozen exponential-moving-average target encoder is a training
        # target, not extra trainable/inference capacity.
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 40_000_000 <= parameters <= 60_000_000
