import random

import torch

from textjepa.data.edits.dataset import EditDataset, collate_edits
from textjepa.data.edits.trajectory import EditEnv, topo_necessary
from textjepa.data.igsm.dataset import DEFAULT_ADJECTIVES, DEFAULT_NOUNS, build_vocab
from textjepa.data.igsm.graph import sample_problem
from textjepa.models import EditJEPA
from textjepa.objectives import DeltaAction, LatentPrediction, ValueRegression


def _env(seed=0):
    rng = random.Random(seed)
    p = sample_problem(rng, DEFAULT_ADJECTIVES, DEFAULT_NOUNS)
    return EditEnv(p, rng), rng


def test_corruption_creates_defects_and_repair_fixes_them():
    for seed in range(5):
        env, rng = _env(seed)
        assert env.n_defects() >= 1
        guard = 0
        while not env.solved:
            env.apply(rng.choice(env.fixing_edits()))
            guard += 1
            assert guard < 50
        assert env.solved
        present = [b.var for b in env.buffer]
        assert sorted(present) == sorted(topo_necessary(env.p))


def test_fix_count_decrements_by_one():
    env, rng = _env(3)
    before = env.n_defects()
    env.apply(rng.choice(env.fixing_edits()))
    assert env.n_defects() == before - 1


def test_edit_dataset_and_model_forward():
    vocab = build_vocab(23)
    ds = EditDataset(vocab, size=6, seed=0)
    batch = collate_edits([ds[i] for i in range(6)], vocab.pad_id)
    # trajectories end solved
    for b in range(6):
        last = int(batch["step_mask"][b].sum()) - 1
        assert batch["remaining"][b, last] == 0
    # no UNK anywhere
    unk = vocab.token_to_id[vocab.UNK]
    assert (batch["buffer_tokens"] != unk).all()
    assert (batch["action_tokens"] != unk).all()

    model = EditJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, slot_layers=1, slot_heads=2,
        n_slots=2, d_action=8, d_macro=4,
    )
    out = model(batch)
    B, T = batch["step_mask"].shape
    assert out.step_states.shape == (B, T, 64)
    assert torch.isfinite(out.preds).all()
    loss = (
        LatentPrediction()(out, batch)
        + DeltaAction()(out, batch)
        + ValueRegression()(out, batch)
    )
    loss.backward()
    assert torch.isfinite(loss)
