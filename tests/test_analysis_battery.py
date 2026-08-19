"""Tests for the representation-analysis battery (scripts/analysis)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import pairs as P                                   # noqa: E402
from analysis.adapters import (                                   # noqa: E402
    DiscourseJEPAAdapter, SentenceLMAdapter, TokenLMAdapter, pad_stack,
)
from analysis.common import (                                     # noqa: E402
    auc, binary_summary, cosine_distance, l2_distance, multiclass_summary,
    normalized_l2, regression_summary, separation, standardize,
    train_binary_probe,
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_auc_perfect_and_chance():
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert auc(scores, labels) == pytest.approx(1.0)
    assert auc(-scores, labels) == pytest.approx(0.0)
    # all ties -> .5
    assert auc(torch.zeros(4), labels) == pytest.approx(0.5)


def test_auc_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, 5, (200,), generator=g).double()
    y = (torch.rand(200, generator=g) > 0.5).double()
    pos, neg = s[y > 0.5], s[y <= 0.5]
    brute = float(
        ((pos.unsqueeze(1) > neg.unsqueeze(0)).double()
         + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).double()).mean()
    )
    assert auc(s, y) == pytest.approx(brute, abs=1e-9)


def test_distances_basic():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[2.0, 0.0]])  # same direction, different scale
    c = torch.tensor([[0.0, 1.0]])
    assert float(cosine_distance(a, b)) == pytest.approx(0.0, abs=1e-9)
    assert float(cosine_distance(a, c)) == pytest.approx(1.0, abs=1e-9)
    assert float(l2_distance(a, b)) == pytest.approx(1.0)
    assert float(normalized_l2(a, b)) == pytest.approx(0.0, abs=1e-6)


def test_separation_direction_and_ratio():
    same = torch.full((50,), 0.1)
    diff = torch.full((50,), 0.5)
    r = separation(same, diff)
    assert r.ratio == pytest.approx(5.0)
    assert r.auc == pytest.approx(1.0)
    flipped = separation(diff, same)
    assert flipped.auc == pytest.approx(0.0)


def test_summaries_report_floors_and_depth():
    scores = torch.tensor([-1.0, 1.0, -1.0, 1.0] * 30)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0] * 30)
    depth = torch.tensor([0, 0, 1, 1] * 30)
    out = binary_summary(scores, labels, depth)
    assert out["acc"] == pytest.approx(1.0)
    assert out["majority_acc"] == pytest.approx(0.5)
    assert set(out["by_depth"]) == {0, 1}
    reg = regression_summary(torch.arange(100.0), torch.arange(100.0))
    assert reg["r2"] == pytest.approx(1.0)
    logits = torch.eye(3).repeat(40, 1)
    mc = multiclass_summary(logits, torch.arange(3).repeat(40))
    assert mc["acc"] == pytest.approx(1.0)


def test_standardize_uses_train_statistics_only():
    tr = torch.randn(100, 4) * 3 + 7
    va = torch.randn(50, 4)
    a, b = standardize(tr, va)
    assert torch.allclose(a.mean(0), torch.zeros(4), atol=1e-5)
    assert not torch.allclose(b.mean(0), torch.zeros(4), atol=1e-3)


def test_binary_probe_learns_a_separable_task():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 8, generator=g)
    y = (X[:, 0] > 0).float()
    s = train_binary_probe("linear", X[:1500], y[:1500], X[1500:], y[1500:],
                           epochs=200, bs=256)
    assert auc(s, y[1500:]) > 0.95


# --------------------------------------------------------------------------- #
# pair construction
# --------------------------------------------------------------------------- #
def test_token_edit_distance():
    assert P.token_edit_distance("a b c", "a b c") == 0
    assert P.token_edit_distance("a b c", "a x c") == 1
    assert P.token_edit_distance("x plus y", "y plus x") == 2


def _stylized_problem():
    from textjepa.data.igsm.dataset import DEFAULT_ADJECTIVES, DEFAULT_NOUNS
    from textjepa.data.igsm.graph import sample_problem
    return sample_problem(
        random.Random(3), DEFAULT_ADJECTIVES, DEFAULT_NOUNS, modulus=23
    )


def test_stylized_pairs_use_the_environment_renderer():
    """Every generated phrase must be exactly what ``action_phrase`` emits
    for SOME (target, operands, operator) triple -- nothing invented."""
    from textjepa.data.igsm.render import action_phrase
    p = _stylized_problem()
    specs = P.stylized_action_pairs(p, [], random.Random(0))
    assert specs
    rendered = {action_phrase(p, v.idx) for v in p.vars}
    for s in specs:
        assert s.base in rendered  # base is always a real catalogue phrase
        assert s.alt.startswith("derive ") and s.alt.endswith(" .")
        assert s.alt.count(" from ") == 1


def test_stylized_paraphrases_preserve_the_value_and_controls_do_not():
    p = _stylized_problem()
    specs = P.stylized_action_pairs(p, [], random.Random(1))
    kinds = {s.subkind for s in specs}
    assert "operator_flip" in kinds
    for s in specs:
        if s.subkind == "commuted_operands":
            assert s.relation == "same_consequence"
            assert s.meta["op"] in {"add", "mul"}
            # the commuted phrase differs in text but not in outcome
            assert s.alt != s.base and s.edit_distance > 0
        elif s.subkind == "operator_flip":
            assert s.relation == "different_consequence"
            assert s.meta["value"] != s.meta["alt_value"]
            assert s.edit_distance == 1  # minimal edit, maximal consequence
        else:
            assert s.relation == "different_consequence"


def test_paraphrase_pairs_are_surface_farther_than_negation_pairs():
    """The double dissociation the report relies on: text distance points the
    WRONG way, so any correct separation must come from the representation."""
    p = _stylized_problem()
    specs = P.stylized_action_pairs(p, [], random.Random(2))
    para = [s.edit_distance for s in specs if s.subkind == "commuted_operands"]
    flip = [s.edit_distance for s in specs if s.subkind == "operator_flip"]
    assert para and flip
    assert min(para) > max(flip)


def test_context_pairs_index_mapping_is_consistent():
    p = _stylized_problem()
    specs = P.stylized_action_pairs(p, [], random.Random(4))
    ctx = P.ContextPairs("x", ["prompt ."], [], specs)
    phrases = ctx.phrases
    assert len(phrases) == len(set(phrases))
    for i, j, spec in ctx.index_pairs():
        assert phrases[i] == spec.base
        assert phrases[j] == spec.alt


def test_proofwriter_polarity_flip_is_the_renderer_inverse():
    from textjepa.data.proofwriter import render_fact
    fact = ("Anne", "is", "quiet", "+")
    assert render_fact(fact) == "Anne is quiet."
    assert render_fact(P._flip(fact)) == "Anne is not quiet."
    verb = ("Anne", "likes", "the dog", "+")
    assert render_fact(verb) == "Anne does likes the dog."
    assert render_fact(P._flip(verb)) == "Anne does not likes the dog."
    assert P._flip(P._flip(fact)) == fact


def test_faithful_has_no_action_paraphrase_and_says_so():
    assert "no natural same-consequence paraphrase" in \
        P.FAITHFUL_ACTION_PARAPHRASE_NOTE.lower().replace("NO ", "no ")


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_vocab():
    from textjepa.data.igsm.dataset import build_vocab
    return build_vocab(23)


def _texts():
    return ["the number of red apples is 3 .", "how many blue keys are there ?"]


def test_pad_stack_shapes():
    out = pad_stack([[1, 2], [3]], pad=0)
    assert out.shape == (2, 2)
    assert out[1, 1] == 0


def test_token_lm_adapter_positions(tiny_vocab):
    from textjepa.models.lm_baseline import DecoderLM
    torch.manual_seed(0)
    m = DecoderLM(len(tiny_vocab), tiny_vocab.pad_id, d_model=32, n_layers=2,
                  n_heads=4, max_len=128).eval()
    ad = TokenLMAdapter(m, tiny_vocab, "tok")
    prompt, steps = _texts(), ["so the number of red apples is 3 ."]
    s = ad.encode_states(prompt, steps)
    assert s.shape == (len(steps) + 1, 32)
    # the state before step 0 must not depend on later steps (causality)
    s2 = ad.encode_states(prompt, [])
    assert torch.allclose(s[0], s2[0], atol=1e-5)
    u = ad.encode_actions_ctx(prompt, steps, ["derive a b from c d plus e f ."])
    assert u.shape == (1, 32)
    assert ad.encode_actions_free(["derive a b from c d plus e f ."]).shape == (1, 32)
    assert ad.predict(torch.zeros(1, 32), torch.zeros(1, 32)) is None


def test_sentence_lm_adapter(tiny_vocab):
    from textjepa.models.sent_lm import SentenceLM
    torch.manual_seed(0)
    m = SentenceLM(len(tiny_vocab), tiny_vocab.pad_id, d_model=32,
                   chunk_layers=1, chunk_heads=2, state_layers=1,
                   state_heads=2, dec_layers=1, dec_heads=2).eval()
    ad = SentenceLMAdapter(m, tiny_vocab, "sent")
    s = ad.encode_states(_texts(), ["so the number of red apples is 3 ."])
    assert s.shape == (2, 32)
    assert ad.encode_actions_free(["look up the number of red apples ."]).shape \
        == (1, 32)
    assert ad.predict(torch.zeros(1, 32), torch.zeros(1, 32)) is None


def test_discourse_jepa_adapter_has_a_predictor(tiny_vocab):
    from textjepa.models.discourse_jepa import DiscourseJEPA
    torch.manual_seed(0)
    m = DiscourseJEPA(len(tiny_vocab), tiny_vocab.pad_id, d_model=32,
                      chunk_layers=1, chunk_heads=2, state_layers=1,
                      state_heads=2, d_action=8).eval()
    ad = DiscourseJEPAAdapter(m, tiny_vocab, "jepa")
    assert ad.d_action == 8
    s = ad.encode_states(_texts(), ["so the number of red apples is 3 ."])
    assert s.shape[0] == 2
    u = ad.encode_actions_free(["look up the number of red apples ."])
    assert u.shape == (1, 8)
    nxt = ad.predict(s[-1:], u)
    assert nxt is not None and nxt.shape == s[-1:].shape
    # a wide (chunk-width) code is not a valid predictor input
    assert ad.predict(s[-1:], torch.zeros(1, 32)) is None


def test_adapters_are_deterministic_and_do_not_mutate_weights(tiny_vocab):
    from textjepa.models.lm_baseline import DecoderLM
    torch.manual_seed(0)
    m = DecoderLM(len(tiny_vocab), tiny_vocab.pad_id, d_model=32, n_layers=1,
                  n_heads=4, max_len=128).eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ad = TokenLMAdapter(m, tiny_vocab, "tok")
    a = ad.encode_states(_texts(), [])
    b = ad.encode_states(_texts(), [])
    assert torch.allclose(a, b)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])


def test_token_lm_batched_context_matches_single(tiny_vocab):
    """Batching the candidate pass must be numerically inert."""
    from textjepa.models.lm_baseline import DecoderLM
    torch.manual_seed(0)
    m = DecoderLM(len(tiny_vocab), tiny_vocab.pad_id, d_model=32, n_layers=2,
                  n_heads=4, max_len=128).eval()
    ad = TokenLMAdapter(m, tiny_vocab, "tok")
    prompt = _texts()
    phrases = [
        "look up the number of red apples .",
        "derive blue keys from red apples plus green pens .",
        "look up the number of golden coins .",
    ]
    batched = ad.encode_actions_ctx(prompt, [], phrases)
    one_by_one = torch.cat(
        [ad.encode_actions_ctx(prompt, [], [p], batch=1) for p in phrases]
    )
    assert torch.allclose(batched, one_by_one, atol=1e-4)


# --------------------------------------------------------------------------- #
# report tables
# --------------------------------------------------------------------------- #
def test_normalized_table_removes_encoder_scale(tmp_path):
    """Two encoders with identical STRUCTURE but different overall scale must
    produce identical normalized rows -- that is the point of the statistic."""
    import json
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "summarize_battery",
        Path(__file__).resolve().parents[1]
        / "scripts" / "analysis" / "summarize_battery.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def block(scale):
        return {"pair_geometry": {"action_ctx": {"cosine": {"per_subkind": {
            "premise_negation": {"mean": 0.2 * scale},
            "premise_reorder": {"mean": 0.01 * scale},
            "other_legal_application": {"mean": 1.0 * scale},
        }}}}}

    path = tmp_path / "geo.json"
    path.write_text(json.dumps(
        {"models": {"big": block(1.0), "collapsed": block(0.001)}}
    ))
    out = mod.normalized_table(path, "other_legal_application")
    assert "0.2000" in out and "0.0100" in out
    # both models produce the same two normalized values
    assert out.count("0.2000") == 2
    assert out.count("0.0100") == 2
    # the reference family itself is not repeated as a column
    assert "other_legal_application |" not in out.split("\n")[2]
