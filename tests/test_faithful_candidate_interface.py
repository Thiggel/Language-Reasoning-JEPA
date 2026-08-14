"""Candidate interfaces on the FAITHFUL iGSM planner.

Regression cover for the 2026-08-13 defect: ``candidate_interface`` was
accepted by the config and never read on the faithful path, so every
"full_catalogue" evaluation was silently a feasible-menu evaluation.
"""

import random

import pytest
import torch

from textjepa.data.faithful import (
    INVALID_DEFINITION_OUTCOME,
    FaithfulDataset,
    FaithfulEnv,
    cached_faithful_vocab,
)
from textjepa.models import DiscourseJEPA
from textjepa.planning.faithful_search import (
    FaithfulPlanner,
    evaluate_faithful_planning,
    faithful_catalogue,
)


@pytest.fixture(scope="module")
def vocab():
    return cached_faithful_vocab()


@pytest.fixture(scope="module")
def dataset(vocab):
    return FaithfulDataset(vocab, size=4, seed=3, max_op=15, max_edge=20,
                           op_range=(3, 10))


@pytest.fixture(scope="module")
def model(vocab):
    torch.manual_seed(0)
    m = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        d_action=8, d_macro=4, predictor_kind="concat", macro_k=0,
    )
    return m.eval()


def _planner(model, vocab, **kw):
    return FaithfulPlanner(model, vocab, torch.device("cpu"), **kw)


def test_unknown_interface_raises(model, vocab):
    with pytest.raises(ValueError, match="unknown candidate interface"):
        _planner(model, vocab, candidate_interface="generator_cycle")
    with pytest.raises(ValueError, match="unknown invalid action mode"):
        _planner(model, vocab, invalid_action_mode="ignore")


def test_feasible_menu_bit_identical(model, vocab, dataset):
    """Default behaviour must be unchanged by the new plumbing."""
    explicit = evaluate_faithful_planning(
        _planner(model, vocab, candidate_interface="feasible_menu"),
        dataset, 4, slack=2, seed=1,
    )
    implicit = evaluate_faithful_planning(
        _planner(model, vocab), dataset, 4, slack=2, seed=1,
    )
    assert explicit == implicit
    assert explicit["latent_planner"]["invalid_action_rate"] == 0.0
    # A feasible-menu run never proposes an infeasible action.
    fp, _ = dataset.problem(0)
    result = _planner(model, vocab).plan_episode(fp, slack=2, seed=1)
    assert result.n_invalid == 0
    # Golden values recorded from the pre-fix implementation (commit 3ddac49)
    # on this exact tiny model/dataset: the feasible-menu path must stay
    # numerically frozen.
    assert explicit["latent_planner"] == {
        "success": 0.5,
        "mean_steps": 8.5,
        "mean_necessary": 7.0,
        "distractor_rate": 0.3235294117647059,
        "invalid_action_rate": 0.0,
    }
    assert explicit["random_policy"] == {
        "success": 0.5,
        "mean_steps": 8.25,
        "mean_necessary": 7.0,
        "distractor_rate": 0.24242424242424243,
        "invalid_action_rate": 0.0,
    }


def test_full_catalogue_is_strictly_larger(dataset):
    fp, _ = dataset.problem(0)
    env = FaithfulEnv(fp)
    catalogue = faithful_catalogue(env)
    assert set(env.feasible_actions()) < set(catalogue)
    assert set(catalogue) == set(fp.action_order)
    # and it does not shrink as the episode progresses
    env.step(env.feasible_actions()[0])
    assert faithful_catalogue(env) == list(fp.action_order)


def test_invalid_actions_are_noops_and_counted(dataset):
    fp, _ = dataset.problem(0)
    env = FaithfulEnv(fp)
    infeasible = [
        q for q in faithful_catalogue(env) if q not in env.feasible_actions()
    ]
    assert infeasible
    before = list(env.resolved)
    text = env.step_or_invalid(infeasible[0])
    assert text == INVALID_DEFINITION_OUTCOME
    assert env.resolved == before


def test_full_catalogue_planner_counts_invalid(model, vocab, dataset):
    planner = _planner(model, vocab, candidate_interface="full_catalogue")
    fp, _ = dataset.problem(0)
    result = planner.plan_episode(fp, slack=4, seed=0)
    assert result.n_invalid > 0
    assert result.n_invalid <= result.steps


def test_full_catalogue_differs_from_feasible_menu(model, vocab, dataset):
    menu = evaluate_faithful_planning(
        _planner(model, vocab), dataset, 4, slack=3, seed=0,
    )
    full = evaluate_faithful_planning(
        _planner(model, vocab, candidate_interface="full_catalogue"),
        dataset, 4, slack=3, seed=0,
    )
    assert full != menu
    assert full["latent_planner"]["invalid_action_rate"] > 0.0
    assert full["random_policy"]["invalid_action_rate"] > 0.0


def test_full_catalogue_allows_deep_lookahead_without_oracle(model, vocab):
    # Oracle guard applies to the feasible menu only: catalogue expansions
    # consult no reference environment.
    _planner(model, vocab, candidate_interface="full_catalogue", lookahead=3)
    with pytest.raises(ValueError, match="allow_oracle_future_actions"):
        _planner(model, vocab, lookahead=3)


def test_catalogue_masks_attempted_actions(dataset):
    fp, _ = dataset.problem(0)
    env = FaithfulEnv(fp)
    full = faithful_catalogue(env)
    tried = {full[0], full[1]}
    masked = faithful_catalogue(env, frozenset(tried))
    assert set(masked) == set(full) - tried
    assert len(masked) == len(full) - 2


def test_planner_never_repeats_an_attempted_action(model, vocab, dataset):
    """Masked full_catalogue must not re-propose a known-dead action."""
    fp, _ = dataset.problem(0)
    planner = _planner(
        model, vocab, candidate_interface="full_catalogue",
        mask_attempted=True,
    )
    seen = []
    original = planner._sequences

    def spy(env, rng, attempted=frozenset(), _o=original):
        seqs = _o(env, rng, attempted)
        roots = {s[0] for s in seqs if s[0] is not None}
        seen.append((set(attempted), roots))
        return seqs

    planner._sequences = spy
    try:
        result = planner.plan_episode(fp, slack=6, seed=2)
    finally:
        planner._sequences = original
    assert len(seen) >= 2
    for step, (attempted, roots) in enumerate(seen):
        # the mask never leaks back into the offered candidates; it holds
        # only actions dead in the CURRENT state (resolved, or invalid since
        # the last progress), so it can shrink after a feasible step
        assert len(attempted) <= step
        assert not (attempted & roots)
    # every step consumed one call; a final extra call happens only when the
    # mask exhausted the catalogue and the episode stalled
    assert result.steps in (len(seen), len(seen) - 1)
    if result.steps == len(seen) - 1:
        assert not seen[-1][1]


def test_unmasked_locks_in_and_masked_does_not(model, vocab, dataset):
    """The lock-in ablation is real and masking removes it."""
    unmasked = evaluate_faithful_planning(
        _planner(model, vocab, candidate_interface="full_catalogue",
                 mask_attempted=False),
        dataset, 4, slack=4, seed=0,
    )
    masked = evaluate_faithful_planning(
        _planner(model, vocab, candidate_interface="full_catalogue",
                 mask_attempted=True),
        dataset, 4, slack=4, seed=0,
    )
    assert masked != unmasked
    # the deterministic first-candidate control is the cleanest signal:
    # unmasked it re-picks catalogue[0] forever, masked it walks the list.
    assert (
        masked["first_feasible_policy"]["invalid_action_rate"]
        < unmasked["first_feasible_policy"]["invalid_action_rate"]
    )


def test_masked_catalogue_episodes_stay_solvable(model, vocab, dataset):
    """An action invalid when tried must be re-proposable after progress.

    The first masking implementation removed attempted actions permanently,
    so any necessary action tried before its dependencies resolved became
    unrecoverable and full_catalogue success was 0 for EVERY policy.  With
    the mask scoped to the current state, a random walk with a generous
    budget must solve some episodes again."""
    res = evaluate_faithful_planning(
        _planner(model, vocab, candidate_interface="full_catalogue",
                 mask_attempted=True),
        dataset, 4, slack=60, seed=0,
    )
    assert res["random_policy"]["success"] > 0


def test_masking_is_inert_under_feasible_menu(model, vocab, dataset):
    """The environment already filters resolved actions from the menu."""
    on = evaluate_faithful_planning(
        _planner(model, vocab, mask_attempted=True), dataset, 4,
        slack=2, seed=1,
    )
    off = evaluate_faithful_planning(
        _planner(model, vocab, mask_attempted=False), dataset, 4,
        slack=2, seed=1,
    )
    assert on == off


def test_catalogue_sequences_are_oracle_free(model, vocab, dataset):
    planner = _planner(
        model, vocab, candidate_interface="full_catalogue", lookahead=2,
        max_expand=8,
    )
    fp, _ = dataset.problem(1)
    env = FaithfulEnv(fp)

    def boom():
        raise AssertionError("feasible_actions() consulted by full_catalogue")

    env.feasible_actions = boom
    seqs = planner._sequences(env, random.Random(0))
    assert all(len(s) == 2 for s in seqs)
    assert {s[0] for s in seqs} == set(fp.action_order)


# --------------------------------------------------------------------------
# Cycle interfaces (ldad_cycle / codebook_ground), critical path #28.


@pytest.fixture(scope="module")
def ldad_model(vocab):
    torch.manual_seed(0)
    m = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        d_action=8, d_macro=4, predictor_kind="concat", macro_k=0,
        observed_action_ldad=True, max_chunk_len=64,
    )
    return m.eval()


def _fitted(ldad_model, vocab, dataset, **kw):
    planner = _planner(
        ldad_model, vocab, candidate_interface="codebook_ground",
        codebook_k=8, **kw,
    )
    planner.fit_action_prior([dataset.problem(3)[0]])
    return planner


def test_cycle_interfaces_require_ldad_decoder(model, vocab):
    for interface in ("ldad_cycle", "codebook_ground"):
        with pytest.raises(RuntimeError, match="observed_action_ldad"):
            _planner(model, vocab, candidate_interface=interface)


def test_ldad_cycle_never_consults_the_feasible_menu(
    ldad_model, vocab, dataset, monkeypatch
):
    """env.feasible_actions() may be called only for invalid classification
    at execution time, never during candidate generation."""
    from textjepa.data import faithful as faithful_mod
    from textjepa.planning import faithful_search as fs

    calls = {"n": 0}
    original = faithful_mod.FaithfulEnv.feasible_actions

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(faithful_mod.FaithfulEnv, "feasible_actions", counting)

    planner = _planner(
        ldad_model, vocab, candidate_interface="ldad_cycle", lookahead=2,
    )
    fp, _ = dataset.problem(0)

    generation_calls = []
    orig_seq = fs.FaithfulPlanner._cycle_sequences

    def spying(self, *args, **kwargs):
        before = calls["n"]
        out = orig_seq(self, *args, **kwargs)
        generation_calls.append(calls["n"] - before)
        return out

    monkeypatch.setattr(fs.FaithfulPlanner, "_cycle_sequences", spying)
    result = planner.plan_episode(fp, slack=3, seed=0)
    assert generation_calls, "cycle candidate generation never ran"
    # zero menu consultations inside candidate generation, even at depth 2
    assert all(n == 0 for n in generation_calls)
    assert result.steps > 0


def test_ldad_cycle_attempted_mask_matches_full_catalogue(
    ldad_model, vocab, dataset
):
    """The scoped mask never re-proposes an action dead in the CURRENT state,
    and (as for full_catalogue) shrinks back after progress."""
    fp, _ = dataset.problem(0)
    planner = _planner(
        ldad_model, vocab, candidate_interface="ldad_cycle",
        mask_attempted=True,
    )
    seen = []
    original = planner._cycle_sequences

    def spy(state, attempted, actions, codes, token_ids, _o=original):
        seqs = _o(state, attempted, actions, codes, token_ids)
        roots = {s[0] for s in seqs if s[0] is not None}
        seen.append((set(attempted), roots))
        return seqs

    planner._cycle_sequences = spy
    try:
        result = planner.plan_episode(fp, slack=6, seed=2)
    finally:
        planner._cycle_sequences = original
    assert len(seen) >= 2
    for step, (attempted, roots) in enumerate(seen):
        assert len(attempted) <= step
        assert not (attempted & roots)
    assert result.steps in (len(seen), len(seen) - 1)


def test_ldad_cycle_prior_top_k_truncates(ldad_model, vocab, dataset):
    fp, _ = dataset.problem(0)
    env = FaithfulEnv(fp)
    full = _planner(ldad_model, vocab, candidate_interface="ldad_cycle")
    actions, codes, tokens = full._episode_catalogue(env)
    assert actions == list(fp.action_order)
    s = torch.zeros(1, 32)
    all_seqs = full._cycle_sequences(s, frozenset(), actions, codes, tokens)
    assert len(all_seqs) == len(fp.action_order)
    top = _planner(
        ldad_model, vocab, candidate_interface="ldad_cycle", prior_top_k=3,
    )
    top_seqs = top._cycle_sequences(s, frozenset(), actions, codes, tokens)
    assert len(top_seqs) == 3
    # top-k roots are a prefix of the full ranking
    assert [q[0] for q in top_seqs] == [q[0] for q in all_seqs][:3]


def test_ldad_cycle_counts_invalid_and_stays_menu_free_at_execution(
    ldad_model, vocab, dataset
):
    planner = _planner(ldad_model, vocab, candidate_interface="ldad_cycle")
    fp, _ = dataset.problem(0)
    result = planner.plan_episode(fp, slack=4, seed=0)
    assert result.n_invalid > 0
    assert result.n_invalid <= result.steps


def test_cycle_reference_policies_run_on_full_catalogue(
    ldad_model, vocab, dataset
):
    """Random/first references for the cycle interfaces see the whole
    catalogue with the same mask -- identical to the full_catalogue refs."""
    cyc = evaluate_faithful_planning(
        _planner(ldad_model, vocab, candidate_interface="ldad_cycle"),
        dataset, 4, slack=3, seed=0,
    )
    full = evaluate_faithful_planning(
        _planner(ldad_model, vocab, candidate_interface="full_catalogue"),
        dataset, 4, slack=3, seed=0,
    )
    assert cyc["random_policy"] == full["random_policy"]
    assert cyc["first_feasible_policy"] == full["first_feasible_policy"]
    assert cyc["random_policy"]["invalid_action_rate"] > 0.0


def test_codebook_ground_requires_fitting_first(ldad_model, vocab, dataset):
    planner = _planner(
        ldad_model, vocab, candidate_interface="codebook_ground",
    )
    fp, _ = dataset.problem(0)
    with pytest.raises(RuntimeError, match="fit_action_prior"):
        planner.plan_episode(fp, slack=2, seed=0)


def test_codebook_ground_grounds_to_current_catalogue_only(
    ldad_model, vocab, dataset
):
    planner = _fitted(ldad_model, vocab, dataset)
    fp, _ = dataset.problem(0)
    env = FaithfulEnv(fp)
    actions, codes, tokens = planner._episode_catalogue(env)
    assert actions
    assert set(actions) <= set(fp.action_order)
    assert len(set(actions)) == len(actions)  # deduplicated
    assert len(actions) <= planner.codebook.shape[0]
    result = planner.plan_episode(fp, slack=4, seed=0)
    assert result.steps > 0


def test_codebook_fit_uses_training_problems_not_eval(
    ldad_model, vocab, dataset
):
    planner = _planner(
        ldad_model, vocab, candidate_interface="codebook_ground",
        codebook_k=4,
    )
    with pytest.raises(ValueError, match="no training action phrases"):
        planner.fit_action_prior([])
    book = planner.fit_action_prior([dataset.problem(3)[0]])
    assert book.shape[1] == 8  # d_action
    assert book.shape[0] <= 4
