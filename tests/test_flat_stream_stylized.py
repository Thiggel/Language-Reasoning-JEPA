"""Stylized-iGSM support for the flat-backbone intent JEPA stream."""

import pytest

from textjepa.data.flat_stream import FlatIntentStreamDataset, collate_flat
from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.igsm.env import INVALID_ACTION_OUTCOME
from textjepa.data.stylized_flat import (
    StylizedFlatDataset, build_flat_stylized_vocab,
)


def build(size=4, **kw):
    vocab = build_flat_stylized_vocab(23)
    base = IGSMDataset(
        vocab, size=size, seed=1, steps_range=(3, 9), leaf_prob=0.35,
        distractor_prob=0.0, geo_rank_k=2, geo_rank_horizon=8,
        geo_rank_horizons=[1, 2, 4, 8], geo_rank_rollout_for_h1=True,
        geo_rank_rollouts=2, all_action_supervision=True, **kw,
    )
    return vocab, StylizedFlatDataset(base)


def test_invalid_outcome_in_vocab():
    vocab = build_flat_stylized_vocab(23)
    assert vocab.decode(vocab.encode(INVALID_ACTION_OUTCOME)) == INVALID_ACTION_OUTCOME
    assert "<unk>" not in vocab.decode(vocab.encode(INVALID_ACTION_OUTCOME))


def test_stream_item_has_catalogue_and_counterfactuals():
    vocab, ds = build(
        invalid_counterfactual_k=8, invalid_counterfactual_unresolved_only=True,
        invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
    )
    stream = FlatIntentStreamDataset(ds)
    item = stream[0]
    assert item["s_pos"][0] < item["a_pos"][0] < item["s_pos"][1]
    assert len(item["catalogue"]) >= len(item["action_tokens"])
    # every candidate index resolves inside the catalogue
    assert max(item["ga_cand_cat"]) < len(item["catalogue"])
    # infeasible negatives are present and their executor outcome is the
    # literal invalid sentence (observable from text alone)
    assert len(item["ga_cand_cat"]) > 3
    assert any(item["ga_alt_invalid"])
    # legality negatives along imagined rollouts
    assert any(depth for cand in item["ga_roll_cf"] for roll in cand for depth in roll)


def test_no_counterfactuals_keeps_legacy_behaviour():
    _, ds = build()
    item = FlatIntentStreamDataset(ds)[0]
    assert not any(item["ga_alt_invalid"])
    assert not any(depth for c in item["ga_roll_cf"] for r in c for depth in r)


def test_collate_shapes():
    vocab, ds = build(
        size=4, invalid_counterfactual_k=8,
        invalid_counterfactual_unresolved_only=True,
        invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
    )
    stream = FlatIntentStreamDataset(ds)
    batch = collate_flat([stream[i] for i in range(4)], vocab.pad_id)
    B, L = batch["tokens"].shape
    assert B == 4
    assert batch["ga_roll_cf"].shape[:3] == batch["ga_roll_tokens"].shape[:3]
    assert batch["ga_roll_cf_mask"].any()
    assert (batch["cat_last"] < L).all()


def test_problem_view_matches_faithful_surface():
    _, ds = build()
    fp, _ = ds.problem(0)
    env = fp.make_env()
    assert fp.prompt_sentences and fp.prompt_sentences[-1].endswith("?")
    assert set(env.feasible_actions()) <= set(fp.action_order)
    assert fp.necessary
    q = env.feasible_actions()[0]
    assert env.action_text(q)
    # an infeasible action renders the invalid sentence and does not step
    infeasible = [a for a in fp.action_order if a not in env.feasible_actions()]
    if infeasible:
        clone = env.clone()
        assert clone.step_or_invalid(infeasible[0]) == INVALID_ACTION_OUTCOME
        assert clone.resolved == []
    env.step(q)
    assert q in env.resolved_set
    assert env.remaining_necessary() <= len(fp.necessary)
