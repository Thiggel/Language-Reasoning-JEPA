"""Depth-cliff fix variants for the flat intent JEPA (2026-08-26).

Three independently switchable, default-OFF additions:

* ``depth_uniform_anchors`` (data): the ranking anchor's steps-to-go is
  sampled uniformly up to the generator cap, so far-from-goal energy
  contrasts are as frequent as near-goal ones;
* ``hindsight_long_horizon_rank`` (objective + model + data): from the
  anchor, the OBSERVED continuation's imagined endpoint at a horizon drawn
  up to the FULL remaining trajectory length must get lower energy than
  endpoints of random feasible same-length rollouts, with the trajectory's
  own achieved terminal (hindsight) in the Energy's goal slot;
* ``mismatched_goal_rank`` (objective + model): E(s_t, s_{t+1}, own goal)
  must be lower than with another problem's goal swapped in — forces the
  head to READ the goal.

These tests pin: OFF by default with the legacy path unchanged, finite and
nonzero loss values with each flag on, gradient flow, and a few optimizer
steps per variant staying finite.
"""

import torch

from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
from textjepa.data.flat_stream import FlatIntentStreamDataset, collate_flat
from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.stylized_flat import (
    StylizedFlatDataset, build_flat_stylized_vocab,
)
from textjepa.models.flat_intent_jepa import FlatIntentJEPA
from textjepa.objectives import (
    EnergyCFFeasibilityRank,
    HindsightLongHorizonRank,
    MismatchedGoalRank,
)


def _base(size=4, **kw):
    vocab = build_flat_stylized_vocab(23)
    base = IGSMDataset(
        vocab, size=size, seed=1, steps_range=(3, 9), leaf_prob=0.35,
        distractor_prob=0.0, geo_rank_k=2, geo_rank_horizon=8,
        geo_rank_horizons=[1, 2, 4, 8], geo_rank_rollout_for_h1=True,
        geo_rank_rollouts=2, all_action_supervision=True,
        invalid_counterfactual_k=8, invalid_counterfactual_unresolved_only=True,
        invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
        **kw,
    )
    return vocab, base


def _batch(size=4, **kw):
    vocab, base = _base(size, **kw)
    stream = FlatIntentStreamDataset(StylizedFlatDataset(base))
    return vocab, collate_flat([stream[i] for i in range(size)], vocab.pad_id)


def _model(vocab, **kw):
    torch.manual_seed(0)
    return FlatIntentJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64, n_layers=2,
        n_heads=4, max_len=4096, **kw,
    )


# ------------------------------------------------------------ defaults off
def test_all_fixes_off_by_default_and_legacy_path_unchanged():
    vocab, batch = _batch()
    off = _model(vocab)
    on = _model(
        vocab, hindsight_long_horizon_rank=True, mismatched_goal_rank=True
    )
    on.load_state_dict(off.state_dict())
    out_off = off(batch)
    assert "energy_hl_obs" not in out_off.extras
    assert "energy_goal_own" not in out_off.extras
    assert HindsightLongHorizonRank()(out_off, batch).item() == 0.0
    assert MismatchedGoalRank()(out_off, batch).item() == 0.0
    # flags on, same weights: the legacy energy contrast is bit-identical
    # (hindsight extras additionally need ga_hl_* data, absent here)
    out_on = on(batch)
    assert "energy_hl_obs" not in out_on.extras  # no ga_hl_act in the batch
    legacy_off = EnergyCFFeasibilityRank()(out_off, batch)
    legacy_on = EnergyCFFeasibilityRank()(out_on, batch)
    torch.testing.assert_close(legacy_off, legacy_on)


# --------------------------------------------------- fix A: anchor depths
def test_depth_uniform_anchors_shifts_anchors_away_from_the_goal():
    n = 80
    _, base_off = _base(n)
    _, base_on = _base(n, depth_uniform_anchors=True)

    def mean_steps_to_go(ds):
        d = []
        for i in range(n):
            item = ds[i]
            if "ga_t" in item:
                t, total = int(item["ga_t"]), len(item["steps"])
                assert 0 <= t < total
                d.append(total - t)
        return sum(d) / len(d)

    assert mean_steps_to_go(base_on) > mean_steps_to_go(base_off) + 0.5


def test_depth_uniform_anchors_leaves_the_trajectory_itself_unchanged():
    _, base_off = _base(4)
    _, base_on = _base(4, depth_uniform_anchors=True)
    for i in range(4):
        a, b = base_off[i], base_on[i]
        assert a["steps"] == b["steps"]
        assert a["actions"] == b["actions"]
        assert a["prompt"] == b["prompt"]


# ------------------------------------------- fix B: hindsight long horizon
def test_hindsight_rollout_horizon_covers_the_full_remaining_length():
    n = 60
    _, base = _base(n, hindsight_long_rollout=True)
    seen_h = []
    for i in range(n):
        item = base[i]
        if "ga_hl_h" not in item:
            continue
        h, t = int(item["ga_hl_h"]), int(item["ga_t"])
        remaining = len(item["steps"]) - t
        assert 1 <= h <= remaining
        assert len(item["ga_hl_actions"]) == 2  # geo_rank_rollouts
        for seq in item["ga_hl_actions"]:
            assert len(seq) <= h
        seen_h.append(h)
    assert seen_h and max(seen_h) > 1  # horizons are not stuck at 1


def test_hindsight_long_horizon_rank_loss_is_finite_nonzero_with_grads():
    vocab, batch = _batch(hindsight_long_rollout=True)
    assert "ga_hl_act" in batch
    model = _model(vocab, hindsight_long_horizon_rank=True)
    out = model(batch)
    assert out.extras["energy_hl_valid"].any()
    loss = HindsightLongHorizonRank()(out, batch)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()

    def norm(prefix):
        return sum(
            float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )

    assert norm("encoder.") > 0
    assert norm("predictor.") > 0
    assert norm("horizon_energy_head.") > 0


def test_hindsight_long_max_caps_the_sampled_horizon():
    n = 40
    _, base = _base(n, hindsight_long_rollout=True, hindsight_long_max=2)
    hs = [int(base[i]["ga_hl_h"]) for i in range(n) if "ga_hl_h" in base[i]]
    assert hs and max(hs) <= 2


# ------------------------------------------------ fix C: mismatched goals
def test_mismatched_goal_rank_loss_is_finite_nonzero_with_grads():
    vocab, batch = _batch()
    model = _model(vocab, mismatched_goal_rank=True)
    out = model(batch)
    e_own, e_mis = out.extras["energy_goal_own"], out.extras["energy_goal_mis"]
    assert e_own.shape == e_mis.shape == out.step_mask.shape
    loss = MismatchedGoalRank()(out, batch)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    grads = sum(
        float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
        if n.startswith("horizon_energy_head.") and p.grad is not None
    )
    assert grads > 0


def test_mismatched_goal_rank_skips_single_problem_microbatches():
    vocab, batch = _batch(size=1)
    out = _model(vocab, mismatched_goal_rank=True)(batch)
    assert "energy_goal_own" not in out.extras
    assert MismatchedGoalRank()(out, batch).item() == 0.0


# --------------------------------------------------- short training smoke
def _train_steps(vocab, batch, objective, model, steps=2):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    values = []
    for _ in range(steps):
        out = model(batch)
        loss = objective(out, batch)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        values.append(float(loss))
    return values


def test_each_fix_survives_a_few_training_steps():
    vocab, batch = _batch(hindsight_long_rollout=True)
    model = _model(
        vocab, hindsight_long_horizon_rank=True, mismatched_goal_rank=True
    )
    hl, mg = HindsightLongHorizonRank(), MismatchedGoalRank()
    cf = EnergyCFFeasibilityRank()

    def objective(out, b):
        return hl(out, b) + mg(out, b) + cf(out, b)

    values = _train_steps(vocab, batch, objective, model, steps=2)
    assert all(v > 0.0 for v in values)


def test_faithful_dataset_supports_all_three_flags():
    vocab = cached_faithful_vocab(15, 20)
    base = FaithfulDataset(
        vocab, size=4, seed=1, max_op=15, max_edge=20, op_range=(3, 15),
        distractor_prob=0.0, geo_rank_k=2, geo_rank_horizon=8,
        geo_rank_horizons=[1, 2, 4, 8], geo_rank_rollout_for_h1=True,
        geo_rank_rollouts=2, all_action_supervision=True,
        invalid_counterfactual_k=8, invalid_counterfactual_unresolved_only=True,
        invalid_counterfactual_resolved_k=4, rollout_counterfactual_k=4,
        depth_uniform_anchors=True, hindsight_long_rollout=True,
    )
    stream = FlatIntentStreamDataset(base)
    batch = collate_flat([stream[i] for i in range(4)], vocab.pad_id)
    assert "ga_hl_act" in batch
    model = _model(
        vocab, hindsight_long_horizon_rank=True, mismatched_goal_rank=True
    )
    out = model(batch)
    for term in (HindsightLongHorizonRank(), MismatchedGoalRank()):
        loss = term(out, batch)
        assert torch.isfinite(loss) and loss.item() > 0.0
