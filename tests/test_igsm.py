import random

from textjepa.data.igsm.dataset import (
    DEFAULT_ADJECTIVES,
    DEFAULT_NOUNS,
    IGSMDataset,
    build_vocab,
    collate,
    rollout_trace,
)
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import sample_problem
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence


def _problem(seed=0):
    rng = random.Random(seed)
    return sample_problem(rng, DEFAULT_ADJECTIVES, DEFAULT_NOUNS), rng


def test_values_and_ancestors():
    p, _ = _problem()
    assert all(0 <= v < p.modulus for v in p.values)
    assert p.query in p.query_ancestors
    for i in p.query_ancestors:
        assert all(pa in p.query_ancestors for pa in p.vars[i].parents)


def test_env_trace_solves():
    p, rng = _problem(1)
    trace = rollout_trace(p, rng, distractor_prob=0.3, max_distractors=2)
    env = SymbolicEnv(p)
    for a in trace:
        env.step(a)
    assert env.solved
    assert env.remaining_necessary() == 0


def test_rendering_is_tokenizable():
    p, rng = _problem(2)
    vocab = build_vocab(p.modulus)
    unk = vocab.token_to_id[vocab.UNK]
    texts = prompt_sentences(p, rng)
    texts += [step_sentence(p, i) for i in p.query_ancestors]
    texts += [action_phrase(p, i) for i in p.query_ancestors]
    for t in texts:
        assert unk not in vocab.encode(t), f"UNK in: {t}"


def test_dataset_collate_shapes():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=8, seed=0)
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    B, T = batch["step_mask"].shape
    assert B == 8
    assert batch["step_tokens"].shape[:2] == (B, T)
    assert batch["op"].shape == (B, T)
    # remaining hits zero exactly at the last valid step
    for b in range(B):
        last = int(batch["step_mask"][b].sum()) - 1
        assert batch["remaining"][b, last] == 0


def test_determinism():
    vocab = build_vocab(23)
    a = IGSMDataset(vocab, size=4, seed=7)[2]
    b = IGSMDataset(vocab, size=4, seed=7)[2]
    assert a["steps"] == b["steps"] and a["answer"] == b["answer"]
