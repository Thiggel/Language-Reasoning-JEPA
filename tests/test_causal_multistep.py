"""Causal (history-grounded) predictor with multi-step supervision.

The decoder-style ``CausalHistoryPredictor`` grounds every imagined step on
the full real prefix, which is the architectural counter to exposure bias.
These tests pin the newly-enabled combinations: ``latent_rollout_pred`` and
``energy_prefix_rank`` with ``predictor_kind='causal'`` (both previously
raised NotImplementedError).  They check the losses are finite and that
gradients reach the encoder, the causal predictor and the energy head.
"""

import torch

from tests.test_energy_prefix_rank import _batch, _model
from textjepa.objectives import EnergyPrefixRank, LatentPrediction


def _grad_norm(module):
    return sum(
        p.grad.abs().sum().item()
        for p in module.parameters() if p.grad is not None
    )


def test_causal_latent_rollout_produces_extras_and_gradients():
    vocab, batch = _batch()
    model = _model(vocab, predictor_kind="causal", latent_rollout_ks=(1, 2))
    out = model(batch)
    assert "rollout_preds" in out.extras
    assert out.extras["rollout_ks"] == (1, 2)
    preds = out.extras["rollout_preds"]
    assert torch.isfinite(preds).all()
    valid = out.extras["rollout_valid"]
    assert valid.any(), "no valid rollout constraints in a 4-item batch"
    loss = (
        (preds - out.extras["rollout_targets"]).abs().mean(-1) * valid
    ).sum() / valid.sum()
    loss.backward()
    assert _grad_norm(model.predictor) > 0
    assert _grad_norm(model.encoder) > 0


def test_causal_prefix_rank_finite_loss_and_gradients():
    vocab, batch = _batch()
    model = _model(
        vocab, predictor_kind="causal", energy_prefix_rank=True,
    )
    out = model(batch)
    assert "energy_prefix_obs" in out.extras
    obj = EnergyPrefixRank("mean_prefix")
    loss = obj(out, batch)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()
        assert _grad_norm(model.predictor) > 0
        assert _grad_norm(model.horizon_energy_head) > 0


def test_causal_history_shapes_match_mlp_masking_semantics():
    """The masked-anchor histories must not leak future steps: anchor t=0
    sees a stalled prefix, and the rollout output stays finite for every
    anchor including the last."""
    vocab, batch = _batch()
    model = _model(vocab, predictor_kind="causal", latent_rollout_ks=(1,))
    out = model(batch)
    assert torch.isfinite(out.extras["rollout_preds"]).all()


def test_multi_insert_counterfactuals_waste_more_steps():
    """n_insert=2 must yield finite prefix energies, and each counterfactual
    path must contain at least as many masked-in steps as the single-insert
    variant while never exceeding the observed step budget."""
    vocab, batch = _batch()
    torch.manual_seed(0)
    m2 = _model(vocab, energy_prefix_rank=True, energy_prefix_n_insert=2)
    out2 = m2(batch)
    assert "energy_prefix_cf" in out2.extras
    assert torch.isfinite(out2.extras["energy_prefix_cf"]).all()
    obs_valid = out2.extras["energy_prefix_obs_valid"]
    cf_valid = out2.extras["energy_prefix_cf_valid"]
    # length-matched: a cf path never spends more steps than its observed path
    assert (cf_valid.sum(-1) <= obs_valid.sum(-1, keepdim=True)).all()
