import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.models import DiscourseJEPA
from textjepa.planning import LatentPlanner, evaluate_planning


def test_planner_runs_end_to_end():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=3, seed=0)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4,
    ).eval()
    planner = LatentPlanner(model, vocab, torch.device("cpu"), lookahead=1)
    results = evaluate_planning(planner, ds, n_episodes=3, slack=2)
    assert set(results) == {
        "latent_planner", "random_policy", "first_feasible_policy", "oracle"
    }
    assert results["oracle"]["success"] == 1.0
    for m in results.values():
        assert 0.0 <= m["success"] <= 1.0
