from functools import partial

import torch
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import (
    SemanticBoundaryLMDataset,
    collate_semantic_lm,
    random_matched_phrase_ends,
    semantic_phrase_ends,
)
from textjepa.models.semantic_token_hierarchy import SemanticBoundaryTokenHierarchyJEPA


def dataset(mode="semantic", size=3):
    vocab = build_vocab(23)
    return vocab, SemanticBoundaryLMDataset(
        vocab, size=size, seed=5, boundary_mode=mode, modulus=23,
        n_vars_range=(8, 12), leaf_prob=0.35, steps_range=(4, 8),
        distractor_prob=0.15, max_distractors=2,
    )


def tiny_model(vocab):
    return SemanticBoundaryTokenHierarchyJEPA(
        len(vocab), vocab.pad_id, d_model=24, encoder_layers=1,
        predictor_layers=1, n_heads=4, ff_mult=2, max_len=768,
        d_action=8, level_dims=(6, 4), low_dense_depth=2,
        high_dense_depth=2,
    )


def test_semantic_and_random_boundaries_have_matched_counts():
    vocab, semantic = dataset("semantic", 4)
    _, random = dataset("random_matched", 4)
    for index in range(4):
        left, right = semantic[index], random[index]
        assert len(left["phrase_ends"]) == len(right["phrase_ends"])
        assert left["sentence_ends"] == right["sentence_ends"]
        assert left["phrase_ends"][-1] == left["sentence_ends"][-1]


def test_phrase_split_does_not_use_symbolic_state():
    vocab = build_vocab(23)
    ids = vocab.encode("so the number of red boxes is 4 plus 2 = 6 .")
    ends = semantic_phrase_ends(ids, vocab.id_to_token)
    text = [vocab.id_to_token[index] for index in ids]
    assert ends == [text.index("is"), text.index("="), len(ids)]


def test_semantic_hierarchy_targets_exact_boundaries_and_is_causal():
    torch.manual_seed(11)
    vocab, source = dataset("semantic", 2)
    batch = next(iter(DataLoader(
        source, batch_size=2,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )))
    model = tiny_model(vocab).eval()
    with torch.no_grad():
        out = model(**batch)
    phrase, sentence = out["levels"]
    for row in range(2):
        phrase_ends = batch["phrase_ends"][row]
        phrase_ends = phrase_ends[phrase_ends > 0]
        sentence_ends = batch["sentence_ends"][row]
        sentence_ends = sentence_ends[sentence_ends > 0]
        assert torch.equal(phrase["end_positions"][row, :len(phrase_ends)], phrase_ends)
        assert torch.equal(sentence["end_positions"][row, :len(sentence_ends)], sentence_ends)
        for column, end in enumerate(phrase_ends):
            assert torch.allclose(
                phrase["target"][row, column], out["target"][row, end - 1]
            )

    # Changing text strictly after the first phrase cannot change its causal
    # start/target state or its observed macro-action code.
    changed = {name: value.clone() for name, value in batch.items()}
    prompt = int(batch["prompt_len"][0])
    first_end = int(batch["phrase_ends"][0, 0])
    changed["tokens"][0, prompt + first_end:] = torch.randint(
        1, len(vocab), changed["tokens"][0, prompt + first_end:].shape
    )
    with torch.no_grad():
        other = model(**changed)
    assert torch.allclose(phrase["prev"][0, 0], other["levels"][0]["prev"][0, 0])
    assert torch.allclose(phrase["target"][0, 0], other["levels"][0]["target"][0, 0])
    assert torch.allclose(phrase["codes"][0, 0], other["levels"][0]["codes"][0, 0])
