import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.models import DiscourseJEPA
from textjepa.objectives import (
    CompositeObjective,
    DeltaAction,
    HierarchyPrediction,
    LatentPrediction,
    RolloutPrediction,
    ValueRegression,
    VICReg,
)


@pytest.fixture(scope="module")
def setup():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=8, seed=0)
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    )
    return vocab, batch, model


def test_forward_shapes(setup):
    _, batch, model = setup
    out = model(batch)
    B, T = batch["step_mask"].shape
    assert out.step_states.shape == (B, T, 64)
    assert out.preds.shape == (B, T, 64)
    assert out.rollout.shape == (B, T, 64)
    assert out.actions.shape == (B, T, 8)
    assert out.value_pred.shape == (B, T + 1)
    assert out.hi_preds is not None
    assert torch.isfinite(out.step_states).all()
    assert torch.isfinite(out.preds).all()


def test_losses_backward(setup):
    _, batch, model = setup
    objective = CompositeObjective(
        {
            "pred": LatentPrediction(),
            "roll": RolloutPrediction(),
            "hier": HierarchyPrediction(),
            "vic": VICReg(),
            "delta": DeltaAction(),
            "value": ValueRegression(),
        },
        {"pred": 1, "roll": 1, "hier": 0.5, "vic": 1, "delta": 2, "value": 1},
    )
    out = model(batch)
    loss, items = objective(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.chunk_encoder.parameters() if p.grad is not None]
    assert grads, "encoder received no gradient"


def test_teacher_gets_no_grad(setup):
    _, batch, model = setup
    for p in model.chunk_teacher.parameters():
        assert not p.requires_grad
    model.update_teachers(0.9)


def test_zero_step_encoding(setup):
    vocab, batch, model = setup
    empty = torch.full((2, 1, 1), vocab.pad_id, dtype=torch.long)
    no_steps = torch.zeros(2, 1, dtype=torch.bool)
    s0, states = model.encode_states(
        batch["prompt_tokens"][:2], batch["prompt_mask"][:2], empty, no_steps
    )
    assert torch.isfinite(s0).all()


def test_geometry_objectives(setup):
    from textjepa.objectives import GoalMonotonicity, TemporalStraightening

    _, batch, model = setup
    out = model(batch)
    for obj in (TemporalStraightening(), GoalMonotonicity()):
        loss = obj(out, batch)
        assert torch.isfinite(loss) and loss >= 0
