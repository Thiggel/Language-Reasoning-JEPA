"""Path-level counterfactual ranking for the flat intent JEPA.

``energy_prefix_rank`` scores every PREFIX of an imagined trajectory instead
of only its endpoint, so a path that wastes a step on an infeasible intent is
penalised even though it reaches the same endpoint.  These tests pin the
wiring: the term is off by default and bit-identical to the legacy path, the
depth-1 prefix energy coincides with the existing one-step anchor energy,
each counterfactual path spends the same number of imagined steps as the
observed one but wastes one of them on an infeasible no-op, and gradients reach the encoder, predictor and Energy head.
"""

import pytest
import torch

from textjepa.data.flat_stream import FlatIntentStreamDataset, collate_flat
from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.stylized_flat import (
    StylizedFlatDataset, build_flat_stylized_vocab,
)
from textjepa.models.flat_intent_jepa import FlatIntentJEPA
from textjepa.objectives import EnergyCFFeasibilityRank, EnergyPrefixRank


def _batch(size=4):
    vocab = build_flat_stylized_vocab(23)
    base = IGSMDataset(
        vocab, size=size, seed=1, steps_range=(3, 9), leaf_prob=0.35,
        distractor_prob=0.0, geo_rank_k=2, geo_rank_horizon=8,
        geo_rank_horizons=[1, 2, 4, 8], geo_rank_rollout_for_h1=True,
        geo_rank_rollouts=2, all_action_supervision=True,
        invalid_counterfactual_k=8, invalid_counterfactual_unresolved_only=True,
        invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
    )
    stream = FlatIntentStreamDataset(StylizedFlatDataset(base))
    return vocab, collate_flat([stream[i] for i in range(size)], vocab.pad_id)


def _model(vocab, **kw):
    torch.manual_seed(0)
    return FlatIntentJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64, n_layers=2,
        n_heads=4, max_len=4096, **kw,
    )


def test_off_by_default_and_legacy_path_unchanged():
    vocab, batch = _batch()
    off, on = _model(vocab), _model(vocab, energy_prefix_rank=True)
    on.load_state_dict(off.state_dict())
    out_off, out_on = off(batch), on(batch)
    assert "energy_prefix_obs" not in out_off.extras
    assert EnergyPrefixRank()(out_off, batch).item() == 0.0
    legacy_off = EnergyCFFeasibilityRank()(out_off, batch)
    legacy_on = EnergyCFFeasibilityRank()(out_on, batch)
    torch.testing.assert_close(legacy_off, legacy_on)


def test_depth1_prefix_energy_equals_the_one_step_anchor_energy():
    vocab, batch = _batch()
    out = _model(vocab, energy_prefix_rank=True)(batch)
    e = out.extras
    B, C = e["ga_energy"].shape
    R = e["energy_prefix_obs"].shape[0] // (B * C)
    obs1 = e["energy_prefix_obs"][:, 0].reshape(B, C, R)
    valid = e["energy_prefix_obs_valid"][:, 0].reshape(B, C, R)
    ref = e["ga_energy"].unsqueeze(-1).expand(B, C, R)
    assert valid.any()
    torch.testing.assert_close(obs1[valid], ref[valid])


def test_counterfactual_path_is_length_matched():
    vocab, batch = _batch()
    e = _model(vocab, energy_prefix_rank=True)(batch).extras
    n_obs = e["energy_prefix_obs_valid"].sum(-1).unsqueeze(-1)
    n_cf = e["energy_prefix_cf_valid"].sum(-1)
    pair = e["energy_prefix_pair_valid"]
    assert pair.any()
    # same imagined budget, one step of it wasted on an infeasible intent
    assert torch.unique((n_cf - n_obs)[pair]).tolist() == [0]


@pytest.mark.parametrize("aggregate", ["mean_prefix", "endpoint"])
def test_gradients_reach_encoder_predictor_and_energy_head(aggregate):
    vocab, batch = _batch()
    model = _model(vocab, energy_prefix_rank=True)
    out = model(batch)
    EnergyPrefixRank(aggregate)(out, batch).backward()

    def norm(prefix):
        return sum(
            float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )

    assert norm("encoder.") > 0
    assert norm("predictor.") > 0
    assert norm("horizon_energy_head.") > 0


def test_endpoint_aggregate_selects_the_last_valid_prefix():
    vocab, batch = _batch()
    e = _model(vocab, energy_prefix_rank=True)(batch).extras
    energies, valid = e["energy_prefix_obs"], e["energy_prefix_obs_valid"]
    last = valid.sum(-1).clamp(min=1) - 1
    expected = energies.gather(-1, last.unsqueeze(-1)).squeeze(-1)
    got = EnergyPrefixRank("endpoint")._score(energies, valid)
    torch.testing.assert_close(got, expected)


def test_unknown_aggregate_is_rejected():
    with pytest.raises(ValueError, match="aggregate"):
        EnergyPrefixRank("last_two")
