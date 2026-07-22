import random

import torch

from textjepa.data.igsm.dataset import (
    DEFAULT_ADJECTIVES,
    DEFAULT_NOUNS,
    IGSMDataset,
    build_vocab,
    collate,
    rollout_trace,
)
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import sample_problem
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence
from textjepa.data.lm import (
    IntentPolicyLMDataset,
    IntentSentencePolicyDataset,
    collate_intent_sentence_policy,
    collate_lm,
)
from textjepa.data.sampling import FreshEpochSampler
from textjepa.utils.checkpoint import build_dataset


def _problem(seed=0):
    rng = random.Random(seed)
    return sample_problem(rng, DEFAULT_ADJECTIVES, DEFAULT_NOUNS), rng


def test_values_and_ancestors():
    p, _ = _problem()
    assert all(0 <= v < p.modulus for v in p.values)
    assert p.query in p.query_ancestors
    for i in p.query_ancestors:
        assert all(pa in p.query_ancestors for pa in p.vars[i].parents)


def test_env_trace_solves():
    p, rng = _problem(1)
    trace = rollout_trace(p, rng, distractor_prob=0.3, max_distractors=2)
    env = SymbolicEnv(p)
    for a in trace:
        env.step(a)
    assert env.solved
    assert env.remaining_necessary() == 0


def test_rendering_is_tokenizable():
    p, rng = _problem(2)
    vocab = build_vocab(p.modulus)
    unk = vocab.token_to_id[vocab.UNK]
    texts = prompt_sentences(p, rng)
    texts += [step_sentence(p, i) for i in p.query_ancestors]
    texts += [action_phrase(p, i) for i in p.query_ancestors]
    for t in texts:
        assert unk not in vocab.encode(t), f"UNK in: {t}"


def test_dataset_collate_shapes():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=8, seed=0)
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    B, T = batch["step_mask"].shape
    assert B == 8
    assert batch["step_tokens"].shape[:2] == (B, T)
    assert batch["op"].shape == (B, T)
    # remaining hits zero exactly at the last valid step
    for b in range(B):
        last = int(batch["step_mask"][b].sum()) - 1
        assert batch["remaining"][b, last] == 0


def test_macro_counterfactuals_have_exact_outcomes_and_collate():
    vocab = build_vocab(23)
    ds = IGSMDataset(
        vocab,
        size=16,
        seed=23,
        distractor_prob=0.2,
        max_distractors=2,
        macro_alt_k=4,
        macro_alt_horizon=3,
    )
    items = [ds[i] for i in range(16)]
    batch = collate(items, vocab.pad_id)
    B, A, K = batch["macro_alt_action_tokens"].shape[:3]
    assert B == 16 and 1 <= A <= 5 and K == 3
    assert batch["macro_alt_valid"].shape == (16, A)
    assert batch["macro_alt_prefix_remaining"].shape == (16, A, K)
    assert batch["macro_alt_step_mask"].shape[:2] == (16, A)
    valid = batch["macro_alt_valid"]
    assert torch.all(batch["macro_alt_remaining"][valid] >= 0)
    assert torch.equal(
        batch["macro_alt_prefix_remaining"][:, :, -1][valid],
        batch["macro_alt_remaining"][valid],
    )
    assert torch.all(
        batch["macro_alt_advantage"][valid]
        <= torch.tensor(3.0)
    )
    # Enabling the supervision view must not alter the factual trajectory.
    paired = IGSMDataset(
        vocab, size=16, seed=23, distractor_prob=0.2, max_distractors=2
    )
    for i, item in enumerate(items):
        assert item["steps"] == paired[i]["steps"]
        assert item["actions"] == paired[i]["actions"]


def test_all_action_feasibility_supervision_collates():
    vocab = build_vocab(23)
    ds = IGSMDataset(
        vocab, size=8, seed=41, all_action_supervision=True
    )
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    B, V = batch["action_candidate_mask"].shape
    assert B == 8
    assert batch["action_candidate_tokens"].shape[:2] == (B, V)
    assert batch["action_feasible"].shape == (
        B, batch["step_mask"].shape[1], V
    )
    valid = batch["step_mask"].unsqueeze(-1) & batch[
        "action_candidate_mask"
    ].unsqueeze(1)
    assert batch["action_feasible"][valid].any()
    assert (~batch["action_feasible"][valid]).any()


def test_intent_policy_lm_masks_only_actions():
    vocab = build_vocab(23)
    discourse = IGSMDataset(
        vocab, size=2, seed=7, n_vars_range=(6, 8), steps_range=(3, 5),
        distractor_prob=0.0, max_distractors=0,
    )
    source = discourse[0]
    policy = IntentPolicyLMDataset(discourse)
    item = policy[0]
    prompt_len = sum(map(len, source["prompt"]))
    assert len(item["tokens"]) == len(item["loss_mask"])
    assert sum(item["loss_mask"]) == sum(len(a) for a in source["actions"])
    assert not any(item["loss_mask"][:prompt_len])
    batch = collate_lm([item, policy[1]], pad_id=vocab.pad_id)
    assert batch["tokens"].shape == batch["loss_mask"].shape
    assert batch["loss_mask"].dtype == torch.bool

    sentence_item = IntentSentencePolicyDataset(discourse)[0]
    assert len(sentence_item["steps"]) == 2 * len(source["steps"])
    assert sentence_item["target_mask"] == [True, False] * len(source["steps"])
    sentence_batch = collate_intent_sentence_policy(
        [sentence_item, IntentSentencePolicyDataset(discourse)[1]], vocab.pad_id
    )
    assert sentence_batch["target_mask"].shape == sentence_batch["step_mask"].shape
    assert torch.all(sentence_batch["target_mask"] <= sentence_batch["step_mask"])


def test_determinism():
    vocab = build_vocab(23)
    a = IGSMDataset(vocab, size=4, seed=7)[2]
    b = IGSMDataset(vocab, size=4, seed=7)[2]
    assert a["steps"] == b["steps"] and a["answer"] == b["answer"]


def test_action_shuffle_preserves_geometric_teacher_samples():
    vocab = build_vocab(23)
    common = dict(
        size=20, seed=7, geo_rank_k=2, geo_rank_horizon=2,
        geo_rank_policy="greedy",
    )
    aligned = IGSMDataset(vocab, shuffle_actions=False, **common)
    shuffled = IGSMDataset(vocab, shuffle_actions=True, **common)
    changed = False
    for index in range(20):
        a, b = aligned[index], shuffled[index]
        for key in (
            "prompt", "steps", "ga_t", "ga_candidate_ids",
            "ga_alt_actions", "ga_alt_steps",
        ):
            assert a.get(key) == b.get(key)
        assert sorted(a["actions"]) == sorted(b["actions"])
        changed |= a["actions"] != b["actions"]
    assert changed


def test_counterfactual_set_preserves_stylized_trajectory_and_teacher():
    vocab = build_vocab(23)
    common = dict(
        size=20, seed=19, geo_rank_k=2, geo_rank_horizon=4,
        geo_rank_rollouts=2,
    )
    no_set = IGSMDataset(vocab, n_alt=0, **common)
    with_set = IGSMDataset(vocab, n_alt=64, **common)
    paired_keys = (
        "prompt", "steps", "actions", "op", "value", "remaining",
        "resolved_n", "necessary", "answer", "var_idx", "ga_t",
        "ga_candidate_ids", "ga_alt_actions", "ga_alt_steps",
        "ga_rollout_steps",
    )
    for index in range(20):
        a, b = no_set[index], with_set[index]
        for key in paired_keys:
            assert a.get(key) == b.get(key), (index, key)


def test_counterfactual_set_preserves_faithful_trajectory_and_teacher():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    vocab = cached_faithful_vocab()
    common = dict(
        size=8, seed=37, max_op=8, max_edge=12, op_range=(3, 8),
        geo_rank_k=2, geo_rank_horizon=2, geo_rank_rollouts=2,
    )
    no_set = FaithfulDataset(vocab, n_alt=0, **common)
    with_set = FaithfulDataset(vocab, n_alt=64, **common)
    paired_keys = (
        "prompt", "steps", "actions", "op", "value", "remaining",
        "resolved_n", "necessary", "answer", "var_idx", "ga_t",
        "ga_alt_actions", "ga_alt_steps", "ga_rollout_steps",
    )
    for index in range(8):
        a, b = no_set[index], with_set[index]
        for key in paired_keys:
            assert a.get(key) == b.get(key), (index, key)


def test_fresh_epoch_sampler_is_disjoint_and_reproducible():
    vocab = build_vocab(23)
    ds = IGSMDataset(vocab, size=8, seed=7)
    sampler = FreshEpochSampler(ds, seed=3, shuffle=True)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert set(epoch0).isdisjoint(epoch1)
    assert set(epoch0) == set(range(8))
    assert set(epoch1) == set(range(8, 16))
    sampler2 = FreshEpochSampler(ds, seed=3, shuffle=True)
    sampler2.set_epoch(1)
    assert epoch1 == list(sampler2)


def test_test_split_uses_independent_generator_seed():
    from omegaconf import OmegaConf

    vocab = build_vocab(23)
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.data = OmegaConf.load("configs/data/igsm.yaml")
    val = build_dataset(cfg, vocab, split="val", size=2)
    test = build_dataset(cfg, vocab, split="test", size=2)
    assert val.seed == 2 and test.seed == 3
    assert val[0]["prompt"] != test[0]["prompt"]


def test_faithful_test_split_uses_independent_generator_seed():
    from omegaconf import OmegaConf

    from textjepa.data.faithful import cached_faithful_vocab

    cfg = OmegaConf.load("configs/config.yaml")
    cfg.data = OmegaConf.load("configs/data/igsm_real.yaml")
    vocab = cached_faithful_vocab()
    val = build_dataset(cfg, vocab, split="val", size=2)
    test = build_dataset(cfg, vocab, split="test", size=2)
    assert val.seed == 2 and test.seed == 3
    val_problem, _ = val.problem(0)
    test_problem, _ = test.problem(0)
    assert val_problem.prompt_sentences != test_problem.prompt_sentences


def test_faithful_action_menu_does_not_expose_graph_iteration_order():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    vocab = cached_faithful_vocab()
    dataset = FaithfulDataset(
        vocab, size=8, seed=47, max_op=12, max_edge=16,
        op_range=(6, 12),
    )
    problems = [dataset.problem(index)[0] for index in range(8)]
    assert all(set(problem.action_order) == set(problem.params)
               for problem in problems)
    assert any(problem.action_order != problem.params for problem in problems)


def test_geometric_rollout_candidates_are_deterministic_and_collate():
    vocab = build_vocab(23)
    ds = IGSMDataset(
        vocab, size=6, seed=11, geo_rank_k=2,
        geo_rank_horizon=4, geo_rank_rollouts=2,
    )
    items = [ds[i] for i in range(6)]
    assert items[2].get("ga_rollout_steps") == ds[2].get("ga_rollout_steps")
    batch = collate(items, vocab.pad_id)
    assert batch["ga_rollout_step_tokens"].shape[:3] == (6, 3, 2)
    assert batch["ga_rollout_step_mask"].shape[:3] == (6, 3, 2)
    assert torch.all(
        batch["ga_rollout_step_mask"].sum(-1)
        <= batch["ga_t"].view(-1, 1, 1) + 4
    )


def test_geometry_greedy_metadata_collates_without_symbolic_quality_labels():
    vocab = build_vocab(23)
    ds = IGSMDataset(
        vocab, size=6, seed=17, geo_rank_k=2,
        geo_rank_horizon=4, geo_rank_policy="greedy",
        geo_rank_beam_width=3,
    )
    items = [ds[i] for i in range(6)]
    batch = collate(items, vocab.pad_id)
    assert batch["ga_greedy"] is True
    assert batch["ga_horizon"] == 4
    assert batch["ga_beam_width"] == 3
    assert batch["ga_candidate_ids"].shape == (6, 3)
    assert "ga_rollout_step_tokens" not in batch
    assert len(batch["ga_problems"]) == 6
    assert len(batch["ga_traces"]) == 6


def test_faithful_geometry_greedy_uses_reference_environment():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    vocab = cached_faithful_vocab()
    ds = FaithfulDataset(
        vocab, size=12, seed=31, max_op=8, max_edge=12,
        op_range=(3, 8), geo_rank_k=2,
        geo_rank_horizon=2, geo_rank_policy="greedy",
        geo_rank_beam_width=2,
    )
    items = [ds[i] for i in range(12)]
    items = [item for item in items if "ga_t" in item][:2]
    assert len(items) == 2
    batch = collate(items, vocab.pad_id)
    assert batch["ga_greedy"] is True
    assert batch["ga_beam_width"] == 2
    assert batch["ga_env_kinds"] == ["faithful", "faithful"]
    assert all(
        isinstance(action, tuple)
        for action in batch["ga_candidate_objects"][0]
        if action is not None
    )


def test_faithful_macro_and_action_support_supervision_collates():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    vocab = cached_faithful_vocab()
    ds = FaithfulDataset(
        vocab, size=4, seed=43, max_op=10, max_edge=14,
        op_range=(6, 10), macro_alt_k=3, macro_alt_horizon=3,
        all_action_supervision=True,
    )
    items = [ds[i] for i in range(4)]
    assert all(item["macro_alt_actions"][0] for item in items)
    assert all(item["action_feasible"] for item in items)
    batch = collate(items, vocab.pad_id)
    assert batch["macro_alt_action_tokens"].shape[:2] == (4, 4)
    assert batch["macro_alt_valid"][:, 0].all()
    assert batch["action_feasible"].shape[0] == 4
