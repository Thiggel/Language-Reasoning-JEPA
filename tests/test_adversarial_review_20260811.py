"""Adversarial review of the open-ended proposers and observed-action planner.

Each test in this file names a concrete defect found by reviewing
``5f89e47..f0fae8c`` (state-conditioned generator head + ``generator_cycle``,
CEM-in-action-embedding-space ``cem_cycle``, and ``ObservedActionPlanner`` with
single-pass slack curves).  Every test failed before the accompanying fix.
"""

from __future__ import annotations

import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.data.igsm.render import action_phrase
from textjepa.models import DiscourseJEPA
from textjepa.objectives import ActionGeneration
from textjepa.planning import LatentPlanner


def _tiny_model(vocab, **kwargs) -> DiscourseJEPA:
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, d_macro=4,
        **kwargs,
    )


# ------------------------------------------------------------------ #
# 1. the generator head is never taught to emit its end marker
# ------------------------------------------------------------------ #
def test_action_generation_loss_supervises_the_phrase_end_marker():
    """Sampling stops at the first PAD, so PAD must be a training target.

    ``ActionGeneration`` masked out every PAD position, so the position right
    after a phrase carried no gradient: the head could never learn to
    terminate, every sample ran to ``max_len`` and no proposal ever parsed.
    """
    torch.manual_seed(0)
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)
    model = _tiny_model(vocab, action_generator=True)
    out = model(batch)
    tokens = batch["action_tokens"]
    valid = out.extras["action_generator_valid"]
    # Find a valid step whose phrase is shorter than the padded width, i.e.
    # one that has an end marker to predict at all.
    lengths = (tokens != vocab.pad_id).sum(-1)
    positions = (valid & (lengths < tokens.shape[-1])).nonzero()
    assert len(positions) > 0, "no padded phrase in this batch"
    b, t = positions[0].tolist()
    end = int(lengths[b, t])

    base = float(ActionGeneration()(out, batch))
    logits = out.extras["action_generator_logits"]
    perturbed = logits.clone()
    # Make the end-marker position confidently predict a non-PAD token.
    perturbed[b, t, end] = -20.0
    perturbed[b, t, end, (vocab.pad_id + 1) % logits.shape[-1]] = 20.0
    out.extras["action_generator_logits"] = perturbed
    assert float(ActionGeneration()(out, batch)) > base, (
        "the loss ignores the end-marker position: the generator can never "
        "learn to terminate a phrase"
    )


# ------------------------------------------------------------------ #
# 2. parse_rate is 1.0 when the generator produced nothing at all
# ------------------------------------------------------------------ #
def test_generator_parse_rate_is_zero_when_no_phrase_is_produced():
    """``1 - unparseable/max(n, 1)`` reported a perfect parse rate for n=0."""
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=3).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
    )
    planner.model.generate_action_phrases = lambda *a, **kw: []
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert result.steps == 0 and not result.solved
    assert result.n_no_proposal == 1
    assert result.parse_rate == 0.0, (
        "an empty generator output was scored as a 100% parse rate"
    )


# ------------------------------------------------------------------ #
# 3. generator sampling ignores the planner's deterministic RNG
# ------------------------------------------------------------------ #
def test_generator_sampling_accepts_an_explicit_rng_generator():
    """cem_cycle seeds every proposal explicitly; generator_cycle did not.

    Without a ``torch.Generator`` the head samples from the ambient global
    RNG, so a ``generator_cycle`` run is not reproducible across processes
    (or across any change in unrelated RNG consumption).
    """
    vocab = build_vocab(23)
    model = _tiny_model(vocab, action_generator=True).eval()
    state = torch.randn(1, 64)

    def sample_with(seed: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(7)
        torch.manual_seed(seed)  # ambient RNG must not matter
        return model.generate_action_phrases(
            state, k=8, top_p=0.9, max_len=12, generator=generator
        )

    assert sample_with(0) == sample_with(1234)


# ------------------------------------------------------------------ #
# 4. beam roots silently fell back to the oracle feasible menu
# ------------------------------------------------------------------ #
def test_beam_roots_never_fall_back_to_the_feasible_menu_for_generator_cycle():
    """``_beam_search`` had no ``generator_cycle`` root branch.

    With ``root_candidates=None`` the chain fell through to ``_feasible``,
    i.e. the oracle menu the open-ended interface is defined to never see.
    """
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=3).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
    )
    # The head proposes nothing parseable, so an honest interface has no
    # candidate at all; only a menu fallback could produce one.
    planner.model.generate_action_phrases = (
        lambda *a, **kw: [vocab.encode("derive nothing at all .")]
    )
    prompt = planner._tokens(["the answer is 1 ."])
    mask = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        state = planner._current_state(prompt, mask, [])
        s0 = planner._s0(prompt, mask)
        history, codes = planner._causal_history(
            prompt, mask, [], problem, []
        )
        best = planner._beam_search(
            state, s0, problem, frozenset(), None, history, codes,
            score_seed="0:0:scores",
        )
    assert best == [None], (
        "generator_cycle beam roots came from the oracle feasible menu"
    )


# ------------------------------------------------------------------ #
# 5. cem_cycle beam roots must not fall back either
# ------------------------------------------------------------------ #
def test_generator_cycle_expansion_uses_the_planner_rng_not_global_rng():
    """Beam expansions resample phrases; they must be seeded deterministically."""
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=5).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    seen: list[bool] = []
    original = model.generate_action_phrases

    def spy(state, k=8, top_p=1.0, max_len=None, temperature=1.0,
            generator=None):
        seen.append(generator is not None)
        return [vocab.encode(action_phrase(problem, a))
                for a in range(len(problem.vars))]

    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2, max_expand=4,
        candidate_interface="generator_cycle", prior_top_k=2,
        search_algorithm="beam",
    )
    planner.model.generate_action_phrases = spy
    with torch.no_grad():
        planner.plan_episode(problem, slack=1, seed=0)
    del original
    assert seen and all(seen), (
        "some generator proposals were drawn from the global RNG"
    )
