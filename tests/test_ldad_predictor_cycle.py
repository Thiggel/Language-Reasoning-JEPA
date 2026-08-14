"""Predictor-cycle LDAD and counterfactual cycle-score contrast terms.

Background (2026-08-14 faithful cycle diagnosis): the ldad_cycle planner
decodes phrases from predictor(s_t, u_t) - s_t, but the standard LDAD loss
only ever trains the decoder on encoder displacements s_{t+1} - s_t.  These
tests cover the two opt-in training terms that close that gap.
"""

import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.models import DiscourseJEPA
from textjepa.objectives import (
    ObservedActionLDADCFContrast,
    ObservedActionLDADPredictorCycle,
)


def _build_batch(geo_rank_k: int):
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=6, seed=0, geo_rank_k=geo_rank_k)
    return vocab, collate([ds[i] for i in range(6)], vocab.pad_id)


def _build_model(vocab, **flags):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        observed_action_ldad=True, **flags,
    )


def test_defaults_off_forward_and_losses_unchanged():
    """With both flags off nothing new is emitted and both losses are 0."""
    vocab, batch = _build_batch(geo_rank_k=2)
    model = _build_model(vocab)
    out = model(batch)
    assert "observed_action_pred_cycle_logits" not in out.extras
    assert "ldad_cf_exec_logits" not in out.extras
    assert ObservedActionLDADPredictorCycle()(out, batch).item() == 0.0
    assert ObservedActionLDADCFContrast()(out, batch).item() == 0.0
    # And the standard LDAD path still emits its usual logits.
    assert "observed_action_logits" in out.extras


def test_flags_require_ldad_and_one_step_horizon():
    vocab, _ = _build_batch(geo_rank_k=0)
    with pytest.raises(ValueError, match="observed_action_ldad"):
        DiscourseJEPA(
            vocab_size=len(vocab), pad_id=vocab.pad_id,
            d_model=64, chunk_layers=1, chunk_heads=2,
            state_layers=2, state_heads=2, d_action=8, d_macro=4,
            observed_action_ldad_predictor_cycle=True,
        )
    with pytest.raises(ValueError, match="one-step"):
        _build_model(
            vocab,
            observed_action_ldad_horizon=2,
            observed_action_ldad_cf_contrast=True,
        )


def test_predictor_cycle_overfit_reduces_decode_error():
    """The term-1 CE on the imagined displacement decreases under training."""
    torch.manual_seed(0)
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=3, seed=0)
    batch = collate([ds[i] for i in range(3)], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=32, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        predictor_heads=2, high_predictor_heads=2,
        observed_action_ldad=True,
        observed_action_ldad_predictor_cycle=True,
    )
    objective = ObservedActionLDADPredictorCycle()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = None
    for _ in range(60):
        opt.zero_grad()
        loss = objective(model(batch), batch)
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    final = objective(model(batch), batch).item()
    assert initial > 0
    assert final < 0.6 * initial


def test_predictor_cycle_gradient_reaches_predictor():
    vocab, batch = _build_batch(geo_rank_k=0)
    model = _build_model(vocab, observed_action_ldad_predictor_cycle=True)
    loss = ObservedActionLDADPredictorCycle()(model(batch), batch)
    assert loss > 0
    loss.backward()
    grads = [
        p.grad for p in model.core.predictor.parameters() if p.grad is not None
    ]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_cf_contrast_gradient_reaches_predictor_and_decoder():
    vocab, batch = _build_batch(geo_rank_k=2)
    model = _build_model(vocab, observed_action_ldad_cf_contrast=True)
    out = model(batch)
    assert "ldad_cf_exec_logits" in out.extras
    loss = ObservedActionLDADCFContrast()(out, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    for module in (model.core.predictor, model.observed_action_decoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


def test_cf_contrast_skips_batches_without_counterfactuals():
    """geo_rank_k=0 batches carry no ranking candidates: loss is 0, no crash."""
    vocab, batch = _build_batch(geo_rank_k=0)
    model = _build_model(vocab, observed_action_ldad_cf_contrast=True)
    out = model(batch)
    assert "ldad_cf_exec_logits" not in out.extras
    loss = ObservedActionLDADCFContrast()(out, batch)
    assert loss.item() == 0.0
    loss.backward()  # still a graph-connected zero
