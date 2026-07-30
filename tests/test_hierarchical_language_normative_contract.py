import pytest
import torch
from types import SimpleNamespace

from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    PRIMARY_EOS_TOKEN_ID,
    TRANSFORMERS_VERSION,
    LanguagePlanningExample,
    assign_igsm_split,
    collate_counterfactual_records,
    collate_language_planning_examples,
    prompt_token_ids,
    normalize_natural_reasoning_steps,
    render_igsm_solution,
    tokenize_steps,
)
from scripts.collect_hierarchical_language_features import (
    _backend_metadata,
    _trim_generated,
    collect_counterfactuals,
)
from textjepa.analysis.compute import ComputeLedger
from textjepa.models.hierarchical_language_jepa import (
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.objectives.hierarchical_language import EMAShrunkMahalanobis
from textjepa.planning.hierarchical_language import (
    build_terminal_state_set,
    construct_value_teacher,
    oracle_high_level_prefix_cost,
    score_token_space_candidates,
    value_guided_high_level_prefix_cost,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


class FakeTokenizer:
    add_bos_token = False

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, **kwargs
    ):
        assert tokenize and add_generation_prompt
        assert kwargs["enable_thinking"] is False
        return [10, 11, 12]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


class NewTransformersFakeTokenizer(FakeTokenizer):
    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, **kwargs
    ):
        return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}


class UnstableBoundaryTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if "second" in text:
            ids[0] += 1
        return ids


class FakeFrozenLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def generate(self, input_ids, **kwargs):
        suffix = torch.tensor(
            [65, 10], device=input_ids.device
        )[None].expand(len(input_ids), -1)
        return torch.cat([input_ids, suffix], -1)

    def forward(
        self, input_ids, output_hidden_states, return_dict,
        logits_to_keep, **kwargs
    ):
        hidden = input_ids.float()[..., None].expand(
            *input_ids.shape, 8
        )
        keep = min(int(logits_to_keep), input_ids.shape[1])
        logits = torch.zeros(
            len(input_ids), keep, 256, device=input_ids.device
        )
        return SimpleNamespace(hidden_states=(hidden,), logits=logits)


def tiny(*, macro=False, value=False):
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=200, pad_id=0,
        d_token=6, d_sentence=4, d_action=3, d_task=5,
        predictor_width=8, token_layers=1, sentence_layers=1,
        n_heads=2, token_context=4, sentence_context=3, max_span=16,
        enable_macro_actions=macro, enable_value=value,
    ))


def test_pinned_reference_constants_and_prompt_contract():
    assert MODEL_ID == "Qwen/Qwen3.5-0.8B"
    assert len(MODEL_REVISION) == 40
    assert TRANSFORMERS_VERSION == "5.13.0"
    assert PAD_TOKEN_ID == 248044
    assert PRIMARY_EOS_TOKEN_ID == 248046
    assert prompt_token_ids(FakeTokenizer(), "problem") == [10, 11, 12]
    assert prompt_token_ids(
        NewTransformersFakeTokenizer(), "problem"
    ) == [10, 11, 12]


def test_backend_metadata_is_weights_only_serializable(tmp_path):
    path = tmp_path / "backend.pt"
    torch.save(_backend_metadata(), path)
    assert isinstance(
        torch.load(path, weights_only=True)["torch_version"], str
    )


def test_global_prefix_boundaries_reconstruct_solution_exactly():
    tokenizer = FakeTokenizer()
    prompt = prompt_token_ids(tokenizer, "q")
    lines = render_igsm_solution(["derive x", "solve x"], "5")
    ids, boundaries, end = tokenize_steps(tokenizer, prompt, lines)
    example = LanguagePlanningExample(
        input_ids=torch.tensor(ids),
        attention_mask=torch.ones(len(ids), dtype=torch.bool),
        prompt_len=len(prompt), solution_end=end,
        step_boundaries=torch.tensor(boundaries),
        reasoning_depth=2,
        canonical_state_ids=torch.arange(4),
        problem_id="q", template_family="t", graph_family="g",
    )
    example.validate()
    batch = collate_language_planning_examples([example])
    assert batch["boundaries"][0, 0] == len(prompt)
    assert batch["boundaries"][0, -1] == end


def test_unstable_tokenizer_step_boundary_is_rejected():
    with pytest.raises(ValueError, match="stable tokenizer"):
        tokenize_steps(
            UnstableBoundaryTokenizer(), [1],
            ["first\n", "second\n"],
        )


def test_short_steps_merge_and_long_punctuation_free_steps_split():
    tokenizer = FakeTokenizer()
    pieces = normalize_natural_reasoning_steps(
        tokenizer, "a\nlong enough line\n", min_tokens=4, max_tokens=64
    )
    assert pieces == ["a\nlong enough line\n"]
    long = "x" * 130 + "\n"
    pieces = normalize_natural_reasoning_steps(
        tokenizer, long, min_tokens=4, max_tokens=64
    )
    assert "".join(pieces) == long
    assert all(len(piece) <= 64 for piece in pieces)


def test_short_newline_candidate_is_still_a_completed_boundary():
    assert _trim_generated(FakeTokenizer(), [65, 10]) == (
        [65, 10], True, False
    )


def test_counterfactual_collector_batches_all_eight_slots():
    tokenizer = FakeTokenizer()
    prompt = prompt_token_ids(tokenizer, "q")
    ids, boundaries, end = tokenize_steps(
        tokenizer, prompt, ["reason\n", "Final answer\n"]
    )
    example = LanguagePlanningExample(
        input_ids=torch.tensor(ids), attention_mask=torch.ones(
            len(ids), dtype=torch.bool
        ),
        prompt_len=len(prompt), solution_end=end,
        step_boundaries=torch.tensor(boundaries), reasoning_depth=1,
        canonical_state_ids=torch.arange(3),
        problem_id="p", template_family="t", graph_family="g",
    )
    ledger = ComputeLedger()
    records = collect_counterfactuals(
        [example], tokenizer, FakeFrozenLM(), "cpu", 0, ledger,
        generation_batch_size=32, reencode_batch_size=32,
    )
    assert len(records) == 16
    assert all(record["completed_boundary"] for record in records)
    assert all(
        len(record["root_token_history_hidden"]) >= 1
        for record in records
    )
    summary = ledger.summary()
    assert summary["components"]["counterfactual_generation"]["calls"] > 0
    assert summary["components"][
        "counterfactual_exact_reencoding"
    ]["estimated_flops"] > 0


def test_pinned_qwen_accepts_every_counterfactual_generate_kwarg():
    from transformers import AutoConfig, AutoModelForMultimodalLM

    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    with torch.device("meta"):
        model = AutoModelForMultimodalLM.from_config(config)
    kwargs = {
        "max_new_tokens": 64,
        "min_new_tokens": 4,
        "eos_token_id": [PRIMARY_EOS_TOKEN_ID, PAD_TOKEN_ID],
        "pad_token_id": PAD_TOKEN_ID,
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 0,
    }
    _, model_kwargs = model._prepare_generation_config(None, **kwargs)
    model._validate_model_kwargs(model_kwargs.copy())
    assert "generator" not in kwargs


def test_length_ood_precedes_held_out_family_assignment():
    split = assign_igsm_split(
        {
            "reasoning_depth": 7,
            "symbolically_verified": True,
            "graph_family": "held",
            "template_family": "known",
            "problem_id": "p",
        },
        held_out_graph_families={"held"},
        held_out_template_families=set(),
    )
    assert split.value == "near_length_ood"


def test_canonical_token_transition_uses_action_at_prefix_n():
    model = tiny()
    hidden = torch.randn(1, 9, 8)
    ids = torch.arange(1, 10)[None]
    output = model.token_forward(
        hidden, ids,
        prompt_len=torch.tensor([3]), solution_end=torch.tensor([7]),
    )
    assert output["token_action_ids"].tolist() == [[4, 5, 6, 7]]
    assert output["token_states"].shape[1] == 4
    assert output["token_targets"].shape[1] == 4


def test_task_embedding_uses_prompt_end_and_ignores_solution_suffix():
    torch.manual_seed(0)
    model = tiny()
    first = torch.randn(2, 8, 8)
    first[1, :3] = first[0, :3]
    first[1, 3:] = torch.randn_like(first[1, 3:]) * 20
    task = model.task_embedding(first, torch.tensor([3, 3]))
    assert torch.allclose(task[0], task[1])


def test_pre_action_context_is_invariant_to_current_and_future_actions():
    torch.manual_seed(2)
    model = tiny()
    model.eval()
    states = torch.randn(1, 4, 4)
    action_a = torch.randn(1, 4, 3)
    action_b = action_a.clone()
    action_b[:, 1:] += 100
    valid = torch.ones(1, 4, dtype=torch.bool)
    _, context_a = model.sentence_predictor(
        states, action_a, valid, return_context=True
    )
    _, context_b = model.sentence_predictor(
        states, action_b, valid, return_context=True
    )
    assert torch.allclose(context_a[:, 1], context_b[:, 1])
    assert not torch.allclose(context_a[:, 2], context_b[:, 2])


def test_nested_sentence_state_is_definitional_before_worker():
    model = tiny()
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.NESTED_SENTENCE
    )
    hidden = torch.randn(1, 9, 8)
    ids = torch.randint(1, 100, (1, 9))
    total, losses = learner(
        hidden, ids, torch.tensor([[3, 6, 8]]),
        attention_mask=torch.ones(1, 9, dtype=torch.bool),
        prompt_len=torch.tensor([3]), solution_end=torch.tensor([8]),
        random_context_truncation=False,
    )
    assert total.isfinite()
    assert "alignment" not in losses
    with torch.no_grad():
        token = model.e0(hidden)
        expected = model.e0_to_1(token)
        actual = model.encode_sentence(hidden)
    assert torch.allclose(actual, expected)


def test_mid_sentence_counterfactual_is_token_only():
    records = [{
        "source": "sample_t1.0",
        "token_ids": torch.tensor([4, 5, 6]),
        "root_hidden": torch.randn(8),
        "suffix_hidden": torch.randn(3, 8),
        "completed_boundary": False,
        "sentence_eligible": False,
        "terminal_eos": False,
    }]
    batch = collate_counterfactual_records(records)
    assert not batch["sentence_eligible"].any()
    learner = HierarchicalLanguageLearner(
        tiny(), ResearchStage.COUNTERFACTUAL_TOKEN
    )
    _, losses = learner.counterfactual_loss(
        batch["root_hidden"], batch["suffix_hidden"],
        batch["token_ids"], batch["lengths"],
        sentence_eligible=batch["sentence_eligible"],
    )
    assert "counterfactual_token" in losses
    assert "counterfactual_sentence" not in losses


def test_counterfactual_replay_preserves_root_predictor_context():
    torch.manual_seed(10)
    model = tiny()
    model.eval()
    root = torch.randn(8)
    suffix = torch.randn(1, 8)
    history = torch.stack([
        torch.stack([torch.zeros(8), root]),
        torch.stack([torch.full((8,), 5.0), root]),
    ])
    captured = []
    handle = model.p0.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach())
    )
    learner = HierarchicalLanguageLearner(
        model, ResearchStage.TOKEN_JEPA
    )
    learner.counterfactual_loss(
        root.repeat(2, 1), suffix.repeat(2, 1, 1),
        torch.tensor([[4], [4]]), torch.tensor([1, 1]),
        root_token_history_hidden=history,
        root_token_history_action_ids=torch.tensor([[3], [3]]),
        root_token_history_lengths=torch.tensor([2, 2]),
    )
    handle.remove()
    assert len(captured) == 2
    assert not torch.allclose(captured[0][:, -1], captured[1][:, -1])


def test_flat_oracle_scorer_never_uses_sentence_encoder():
    metric = EMAShrunkMahalanobis(2, shrinkage=1)
    result = score_token_space_candidates(
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[1, 2], [3, 4]]),
        torch.zeros(2), torch.zeros(2), metric,
        prior_weight=0, temperature=1, vocab_size=10,
    )
    assert result.best_index == 0


def test_terminal_set_gathers_state_before_chat_eos():
    model = tiny()
    hidden = torch.randn(1, 2, 7, 8)
    goals, mask = build_terminal_state_set(
        model, hidden, torch.tensor([[5, 6]]),
        torch.tensor([[True, True]]),
    )
    expected = model.encode_sentence(torch.stack([
        hidden[0, 0, 4], hidden[0, 1, 5]
    ]), target=True)
    assert torch.allclose(goals[0], expected)
    assert mask.all()


def test_high_level_oracle_and_value_choose_best_prefix_without_budget():
    metric = EMAShrunkMahalanobis(1, shrinkage=1)
    states = torch.tensor([[[1.0], [0.0], [4.0]]])
    oracle, horizon = oracle_high_level_prefix_cost(
        states, torch.zeros(1, 3), torch.zeros(1, 1, 1), metric,
        terminal_temperature=1, step_cost=0, prior_weight=0,
    )
    assert horizon.item() == 1
    contexts = torch.zeros(1, 3, 2)
    value_cost, value_horizon = value_guided_high_level_prefix_cost(
        states, contexts, torch.zeros(1, 2), torch.zeros(1, 3),
        lambda state, context, task: state.square().squeeze(-1),
        step_cost=0, prior_weight=0,
    )
    assert value_horizon.item() == 1
    assert oracle.item() >= 0 and value_cost.item() == 0


def test_value_teacher_includes_one_action_terminal_prefix():
    metric = EMAShrunkMahalanobis(1, shrinkage=1)
    metric.covariance.fill_(1)

    def advance(state, context, action):
        return state + action, context + action, torch.zeros(len(state))

    target, successor, _ = construct_value_teacher(
        torch.zeros(1, 1), torch.zeros(1, 1),
        torch.ones(1, 1, 1),
        torch.full((1, 1, 1, 1, 1), 100.0),
        torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool),
        advance, metric, terminal_temperature=0.1,
        teacher_temperature=0.1, step_cost=0, prior_weight=0,
    )
    assert target.item() == pytest.approx(0, abs=1e-6)
    assert successor.item() == 1
