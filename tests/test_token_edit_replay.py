from functools import partial
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
from torch.utils.data import DataLoader

from scripts import collect_faithful_token_edit_replay as collector
from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    _apply,
    faithful_token_edit_vocab,
)
from textjepa.data.token_edit_replay import (
    REPLAY_FORMAT,
    FrozenPolicyReplayDataset,
    MixedReplayTokenEditDataset,
)
from textjepa.models.edit_jepa import EditJEPA


def replay_file(tmp_path, *, exact=False):
    vocab = faithful_token_edit_vocab()
    observed = [vocab.token_to_id[word] for word in ["token", "position", "with"]]
    prompt_token = vocab.token_to_id["delete"]
    goal_only = vocab.token_to_id["insert"]
    initial = [observed[:2], observed[2:]]
    action = ("replace", 0, prompt_token)
    outcome = [list(sentence) for sentence in initial]
    _apply(outcome, action)
    goal = [[goal_only, observed[1]], observed[2:]]
    record = {
        "problem_id": "igsm_real_token_edit:train:1:17",
        "source_seed": 1,
        "source_index": 17,
        "snapshot_id": "snapshot-17",
        "state_snapshot_ids": ["snapshot-17", "snapshot-18"],
        "prompt": [[prompt_token]],
        "buffer_snapshots": [initial, outcome],
        "behavior_actions": [list(action)],
        "terminal_privileged_goal_buffer": goal,
        "answer": 0,
        "information_regime": {
            "proposal_generation": "deployment_feasible_prompt_plus_current",
            "goal_buffer": "terminal_privileged_training_label_only",
            "canonical_action_access": False,
        },
    }
    path = tmp_path / ("exact.pt" if exact else "latent.pt")
    torch.save({
        "format": REPLAY_FORMAT,
        "manifest": {"checkpoint_sha256": "abc"},
        "records": [record],
    }, path)
    return vocab, path, record, goal_only


def test_replay_materializes_exact_outcomes_and_target_free_prompt_pool(tmp_path):
    vocab, path, record, goal_only = replay_file(tmp_path)
    dataset = FrozenPolicyReplayDataset(
        path, vocab, proposal_pool_k=12,
        proposal_token_pool="prompt_plus_current", seed=3,
    )
    item = dataset[0]
    assert item["goal_buffer"] == record["terminal_privileged_goal_buffer"]
    assert item["buffers"][-1] != item["goal_buffer"]
    assert item["information_regime"]["canonical_action_access"] is False
    allowed = {
        token for sentence in record["prompt"] + record["buffer_snapshots"][0]
        for token in sentence
    }
    for candidate, outcome, content in zip(
        item["proposal_op"][0], item["proposal_buffers"][0],
        item["proposal_edit_content_token"][0],
    ):
        del candidate
        assert content == vocab.pad_id or content in allowed
        assert goal_only not in allowed
        assert outcome != record["buffer_snapshots"][0]


def test_mixed_replay_replaces_half_without_changing_dataset_length(tmp_path):
    vocab, path, _, _ = replay_file(tmp_path)
    expert = FaithfulTokenEditDataset(
        vocab, size=4, seed=5, max_op=6, max_edge=12, op_range=(3, 6),
        min_edits=2, max_edits=2, proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current", fresh_per_epoch=False,
    )
    replay = FrozenPolicyReplayDataset(
        path, vocab, proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current",
    )
    mixed = MixedReplayTokenEditDataset(expert, replay, fraction=0.5)
    assert len(mixed) == len(expert) == 4
    assert ["replay_snapshot_id" in mixed[index] for index in range(4)] == [
        True, True, False, False
    ]
    assert all("goal_buffer" in mixed[index] for index in range(4))


def test_replay_max_depth_truncates_actions_and_states(tmp_path):
    vocab, path, record, _ = replay_file(tmp_path)
    second_action = ("delete", 1, None)
    second_outcome = [list(sentence) for sentence in record["buffer_snapshots"][-1]]
    _apply(second_outcome, second_action)
    payload = torch.load(path, weights_only=False)
    payload["records"][0]["behavior_actions"].append(list(second_action))
    payload["records"][0]["buffer_snapshots"].append(second_outcome)
    torch.save(payload, path)
    dataset = FrozenPolicyReplayDataset(
        path, vocab, proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current", max_depth=1,
    )
    item = dataset[0]
    assert len(item["actions"]) == 1
    assert len(item["buffers"]) == 2


def test_exact_replay_teacher_accepts_unbounded_distance(tmp_path):
    vocab, path, _, _ = replay_file(tmp_path, exact=True)
    dataset = FrozenPolicyReplayDataset(
        path, vocab, proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current",
        gar_teacher="token_edit_distance",
    )
    item = dataset[0]
    assert len(item["gar_proposal_token_edit_target"][0]) == 4


def test_separate_privileged_goal_collates_and_drives_finite_gar_targets(tmp_path):
    vocab, path, _, _ = replay_file(tmp_path)
    dataset = FrozenPolicyReplayDataset(
        path, vocab, proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current",
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=1,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    assert "goal_buffer_tokens" in batch
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=64, d_action=8, predictor_layers=1,
        predictor_heads=4, macro_k=0, token_aligned=True,
        token_predictor_layers=1, selected_k=2,
    )
    out = model(batch)
    assert out.extras["gar_uses_separate_terminal_privileged_goal"] is True
    assert torch.isfinite(out.extras["gar_action_target"]).all()
    assert torch.isfinite(out.extras["gar_alt_action_target"]).all()


def test_collector_behavior_is_independent_of_terminal_goal(monkeypatch):
    vocab = faithful_token_edit_vocab()
    initial = [[vocab.token_to_id["token"], vocab.token_to_id["position"]]]
    prompt = [[vocab.token_to_id["with"]]]
    first = {"prompt": prompt, "buffers": [initial, [[1, 2]]], "answer": 0}
    second = {"prompt": prompt, "buffers": [initial, [[3, 4]]], "answer": 0}
    monkeypatch.setattr(
        collector, "gar_scores",
        lambda model, buffer, candidates, pad_id, device, batch_size: list(
            range(len(candidates))
        ),
    )
    kwargs = dict(
        model=None, vocab=vocab, source_index=0, source_seed=1,
        device="cpu", candidate_budget=12, rollout_depth=2,
        score_batch_size=8, proposal_seed=9,
    )
    left = collector.collect_record(item=first, **kwargs)
    right = collector.collect_record(item=second, **kwargs)
    assert left["behavior_actions"] == right["behavior_actions"]
    assert left["buffer_snapshots"] == right["buffer_snapshots"]
    assert left["terminal_privileged_goal_buffer"] != right[
        "terminal_privileged_goal_buffer"
    ]


def test_replay_pilot_config_requires_path_and_holds_presentation_count():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=edit_token_structured_gar_replay_pilot",
                "data.replay_path=/tmp/replay.pt",
            ],
        )
    assert cfg.data.train_size == 6000
    assert cfg.data.replay_fraction == 0.5
    assert cfg.data.proposal_pool_k == 64
    assert cfg.data.proposal_token_pool == "prompt_plus_current"
    assert cfg.model.selected_k == 4
    assert cfg.objective.gar_action_value.weight == 0.3
    assert cfg.objective.gar_action_value.pairwise_weight == 1.0
    assert cfg.train.batch_size == 8
    assert cfg.train.epochs == 3
