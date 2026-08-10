from __future__ import annotations

import torch
from torch import nn
import pytest

from textjepa.analysis.predictive_state import (
    effective_rank,
    progress_metrics,
    spearman,
)
from textjepa.data.predictive_state import (
    REASONING_STATE_SCHEMA,
    pack_documents,
    split_wikitext_articles,
    split_whitespace_documents,
    token_tensor_fingerprint,
    validate_reasoning_bundle,
)
from textjepa.models.action_transition import (
    ActionConditionedTransition,
    LoRALinear,
    ResidualCapture,
    TransitionConfig,
    install_upper_lora,
)
from textjepa.objectives.predictive_state import stage1_loss, transition_loss
from textjepa.planning.predictive_state import (
    GoalDistanceModel,
    block_beam_search,
    goal_distance_loss,
    potential_shaped_reward,
)
from textjepa.training.predictive_state import (
    UpperStackRunner,
    active_segment_key_mask,
    dense_stage1_prediction,
    generate_jump_trajectory,
    recurrent_rollout_loss,
    teacher_forward,
)
from scripts.evaluate_action_transition_rollout import (
    full_greedy_decode,
    speculative_greedy_decode,
)
from scripts.collect_predictive_state_reasoning import prompt_state_index
from scripts.probe_predictive_state_goal_geometry import goal_specificity_controls
from scripts.plan_predictive_state_blocks import answers_equal, generate_jump_candidates
from scripts.build_predictive_state_shaped_rewards import calibrate_potential_weight


def test_document_packing_masks_only_cross_document_transition():
    packed = pack_documents(
        [[1, 2], [3, 4]], context_length=6, eos_token_id=9
    )
    assert packed["input_ids"].tolist() == [[1, 2, 9, 3, 4, 9]]
    assert packed["target_mask"].tolist() == [
        [True, True, False, True, True]
    ]


def test_document_packing_resets_positions_to_block_cross_document_attention():
    packed = pack_documents(
        [[1, 2], [3, 4]], context_length=6, eos_token_id=9
    )
    assert packed["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]


def test_token_tensor_fingerprint_is_path_and_serialization_independent():
    left = pack_documents([[1, 2], [3, 4]], context_length=6, eos_token_id=9)
    right = {key: value.clone() for key, value in left.items()}
    assert token_tensor_fingerprint(left) == token_tensor_fingerprint(right)
    right["input_ids"][0, 0] += 1
    assert token_tensor_fingerprint(left) != token_tensor_fingerprint(right)


def test_whitespace_only_lines_split_wikitext_documents():
    text = " \n = Heading = \n \n First paragraph.\n \t\nSecond paragraph.\n"
    assert split_whitespace_documents(text) == [
        "= Heading =", "First paragraph.", "Second paragraph."
    ]


def test_wikitext_article_split_keeps_paragraph_context():
    text = " = A = \n \n first\n \nsecond\n \n = B = \n \nthird\n"
    assert split_wikitext_articles(text) == [
        "= A =\n\nfirst\n\nsecond", "= B =\n\nthird"
    ]


def _reasoning_payload(record_overrides=None):
    record = {
        "problem_id": "p1", "trajectory_id": "t1",
        "prompt_state": torch.zeros(4),
        "states": torch.zeros(3, 4),
        "valid": torch.ones(3, dtype=torch.bool),
        "remaining_chunks": torch.tensor([2.0, 1.0, 0.0]),
        "correct": True, "terminal_index": 2,
    }
    record.update(record_overrides or {})
    return {
        "schema_version": REASONING_STATE_SCHEMA,
        "metadata": {
            "model_id": "m", "model_revision": "r",
            "checkpoint_fingerprint": "f",
            "oracle_terminal_states": True,
            "candidate_privileged_outcomes": True,
            "cross_project_information": False,
        },
        "records": [record],
    }


@pytest.mark.parametrize("payload", [
    {
        "schema_version": REASONING_STATE_SCHEMA,
        "metadata": {
            "model_id": "m", "model_revision": "r",
            "checkpoint_fingerprint": "f",
            "oracle_terminal_states": True,
            "candidate_privileged_outcomes": True,
            "cross_project_information": False,
        },
        "records": [],
    },
    _reasoning_payload({"terminal_index": 3}),
    _reasoning_payload({"terminal_index": 1, "valid": torch.tensor([True, False, True])}),
    _reasoning_payload({"prompt_state": torch.zeros(5)}),
    _reasoning_payload({"states": torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, float("nan"), 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])}),
])
def test_reasoning_bundle_rejects_empty_or_malformed_records(payload):
    with pytest.raises(ValueError):
        validate_reasoning_bundle(payload)


def test_prompt_state_index_uses_joint_token_offsets():
    # The final token straddles the prompt/trajectory boundary and must not be
    # treated as a pure prompt state.
    assert prompt_state_index([(0, 2), (2, 5), (5, 8)], 6) == 1


def test_oracle_probe_reports_all_goal_specificity_controls():
    def record(problem, trajectory, answer, offset):
        states = torch.tensor([
            [3.0 + offset, 0.0], [1.0 + offset, 0.0], [offset, 0.0]
        ])
        return {
            "problem_id": problem, "trajectory_id": trajectory,
            "states": states, "valid": torch.ones(3, dtype=torch.bool),
            "terminal_index": 2, "correct": True, "answer": answer,
        }
    records = [
        record("p1", "a", "7", 0.0),
        record("p1", "b", "7", 0.1),
        record("p2", "c", "7", 1.0),
        record("p3", "d", "9", 2.0),
    ]
    controls = goal_specificity_controls(
        records, test_ids={"p1", "p2", "p3"}, transform=None,
        kind="euclidean", mean=None,
    )
    assert set(controls) == {
        "another_correct_same_problem", "different_problem_same_answer",
        "random_terminal", "same_relative_position_other_trajectory",
    }
    assert controls["another_correct_same_problem"]["pairs"] == 2
    assert controls["different_problem_same_answer"]["pairs"] == 3
    assert controls["random_terminal"]["pairs"] == 4


def test_transition_detaches_action_but_not_state_and_records_capacity():
    config = TransitionConfig(
        hidden_size=8, source_layers=(3, 4), target_layer=2,
        action_dim=8, variant="full",
    )
    predictor = ActionConditionedTransition(config)
    states = {
        3: torch.randn(2, 8, requires_grad=True),
        4: torch.randn(2, 8, requires_grad=True),
    }
    action = torch.randn(2, 8, requires_grad=True)
    predictor(states, action).sum().backward()
    assert states[3].grad is not None
    assert states[4].grad is not None
    assert action.grad is None
    assert predictor.metadata()["parameters"] == sum(
        parameter.numel() for parameter in predictor.parameters()
    )
    assert predictor.output.weight.std().item() < 0.002


def test_fp16_residuals_keep_trainable_predictor_gradients_fp32():
    predictor = ActionConditionedTransition(TransitionConfig(
        hidden_size=8, source_layers=(3, 4), target_layer=2,
        action_dim=8, variant="full",
    ))
    states = {3: torch.randn(2, 8).half(), 4: torch.randn(2, 8).half()}
    action = torch.randn(2, 8).half()
    prediction = predictor(states, action)
    assert prediction.dtype == torch.float16
    prediction.float().square().mean().backward()
    assert all(parameter.dtype == torch.float32 for parameter in predictor.parameters())
    assert all(parameter.grad is not None for parameter in predictor.parameters())


def test_action_only_and_no_action_interfaces_are_distinct():
    action_only = ActionConditionedTransition(TransitionConfig(
        hidden_size=8, source_layers=(3, 4), target_layer=2,
        action_dim=8, variant="action_only",
    ))
    no_action = ActionConditionedTransition(TransitionConfig(
        hidden_size=8, source_layers=(3, 4), target_layer=2,
        action_dim=8, variant="no_action",
    ))
    action = torch.randn(2, 8)
    assert action_only({}, action).shape == (2, 8)
    assert no_action({3: action, 4: action}, None).shape == (2, 8)
    full = ActionConditionedTransition(TransitionConfig(
        hidden_size=8, source_layers=(3, 4), target_layer=2,
        action_dim=8, variant="full",
    ))
    assert sum(p.numel() for p in no_action.parameters()) == sum(
        p.numel() for p in full.parameters()
    )
    assert sum(p.numel() for p in action_only.parameters()) == sum(
        p.numel() for p in full.parameters()
    )


def test_transition_loss_separates_direction_and_scale():
    target = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    equal_direction = transition_loss(2 * target, target, mask)
    assert equal_direction.cosine.item() < 1e-6
    assert equal_direction.scale.item() > 0
    exact = transition_loss(target, target, mask)
    assert exact.total.item() < 1e-6


def test_stage1_scale_coefficient_is_not_multiplied_by_prediction_weight():
    logits = torch.randn(1, 3, 7)
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.randn(1, 2, 8)
    prediction = 3.0 * target.roll(1, dims=-1)
    result = stage1_loss(
        logits=logits,
        input_ids=tokens,
        target_mask=mask,
        prediction=prediction,
        target_state=target,
        prediction_weight=0.1,
        scale_weight=0.01,
    )
    expected = (
        result.ntp
        + 0.1 * result.transition.cosine
        + 0.01 * result.transition.scale
    )
    torch.testing.assert_close(result.total, expected)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, hidden):
        return hidden + 1


class _ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block() for _ in range(4)])


def test_upper_lora_freezes_base_and_starts_with_zero_update():
    model = _ToyLM()
    original = model.model.layers[2].self_attn.q_proj
    hidden = torch.randn(3, 4)
    expected = original(hidden)
    accounting = install_upper_lora(
        model, first_trainable_layer=3, rank=2, alpha=4
    )
    wrapped = model.model.layers[2].self_attn.q_proj
    assert isinstance(wrapped, LoRALinear)
    torch.testing.assert_close(wrapped(hidden), expected)
    assert accounting["trainable_parameters"] > 0
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_a.dtype == torch.float32
    assert wrapped.lora_b.dtype == torch.float32


def test_raw_residual_capture_uses_one_indexed_blocks():
    model = _ToyLM()
    with ResidualCapture(model, (2, 4)) as capture:
        hidden = torch.zeros(1, 4)
        for block in model.model.layers:
            hidden = block(hidden)
    torch.testing.assert_close(capture.values[2], torch.full((1, 4), 2.0))
    torch.testing.assert_close(capture.values[4], torch.full((1, 4), 4.0))


def test_goal_distance_loss_rewards_decreasing_correct_trajectory():
    distance = torch.tensor([[2.0, 1.0, 0.0], [6.0, 5.0, 3.0]])
    valid = torch.ones_like(distance, dtype=torch.bool)
    remaining = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    loss = goal_distance_loss(
        distance, valid=valid, remaining_chunks=remaining,
        correct=torch.tensor([True, False]),
        terminal_indices=torch.tensor([2, 2]),
    )
    assert loss.terminal.item() == 0
    assert loss.time.item() == 0
    assert loss.monotonicity.item() == 0
    assert loss.negative.item() == 1.0


def test_potential_shaping_uses_negative_distance_potential_and_clips():
    reward = torch.tensor([0.0, 1.0])
    shaped, increment = potential_shaped_reward(
        reward, torch.tensor([3.0, 2.0]), torch.tensor([2.0, 0.0]),
        gamma=1.0, weight=1.0, clip=0.1,
    )
    torch.testing.assert_close(increment, torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(shaped, torch.tensor([0.1, 1.1]))


def test_shaping_budget_uses_one_global_fixed_weight():
    weight = calibrate_potential_weight(
        [torch.tensor([4.0, -6.0]), torch.tensor([1.0, 1.0])],
        requested_weight=0.2, maximum_absolute_return=0.5,
    )
    assert weight == 0.05


def test_block_beam_keeps_highest_scored_leaf():
    def generate(latent, tokens, count):
        chunks = torch.arange(count).view(1, count, 1).repeat(len(tokens), 1, 1)
        return chunks, chunks[..., 0].float()

    def advance(latent, chunks):
        return latent + chunks.float()

    def score(latent, _tokens, cumulative):
        return latent[:, 0] + cumulative

    result = block_beam_search(
        initial_latent=torch.zeros(1, 1),
        initial_tokens=torch.zeros(1, 0, dtype=torch.long),
        generate_candidates=generate, advance=advance, score_leaves=score,
        beam_width=2, candidates_per_beam=2, chunk_length=1, depth=2,
    )
    assert result.tokens.shape == (2, 2)
    assert result.score[0] >= result.score[1]


def test_block_beam_never_mixes_independent_batch_roots():
    def generate(latent, tokens, count):
        chunks = torch.arange(count).view(1, count, 1).repeat(len(tokens), 1, 1)
        return chunks, torch.zeros(len(tokens), count)

    def advance(latent, chunks):
        return latent + chunks.float()

    def score(latent, _tokens, _cumulative):
        return latent[:, 0]

    result = block_beam_search(
        initial_latent=torch.tensor([[0.0], [100.0]]),
        initial_tokens=torch.zeros(2, 0, dtype=torch.long),
        generate_candidates=generate, advance=advance, score_leaves=score,
        beam_width=2, candidates_per_beam=3, chunk_length=1, depth=1,
    )
    assert result.root_ids.tolist() == [0, 0, 1, 1]
    assert result.tokens[:, 0].tolist() == [2, 1, 2, 1]


def test_progress_and_rank_metrics_have_expected_direction():
    result = progress_metrics([torch.tensor([3.0, 2.0, 1.0, 0.0])])
    assert result["progress_spearman_mean"] == 1.0
    assert spearman(torch.arange(4.0), torch.arange(4.0)) == 1.0
    assert effective_rank(torch.tensor([1.0, 1.0])) == 2.0


def test_goal_model_shapes_prompt_conditioned_distances():
    model = GoalDistanceModel(6, 4)
    states = torch.randn(2, 3, 6)
    prompt = torch.randn(2, 6)
    assert model(states, prompt).shape == (2, 3)


def test_tiny_qwen_teacher_and_recurrent_upper_stack():
    transformers = __import__("transformers")
    Qwen2Config = transformers.Qwen2Config
    Qwen2ForCausalLM = transformers.Qwen2ForCausalLM
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )
    model = Qwen2ForCausalLM(config)
    predictor = ActionConditionedTransition(TransitionConfig(
        hidden_size=16, source_layers=(3, 4), target_layer=2,
        action_dim=16, variant="full",
    ))
    tokens = torch.randint(0, 32, (2, 10))
    with ResidualCapture(model, (2, 3, 4)) as capture:
        teacher = teacher_forward(
            model, tokens, capture=capture, use_cache=False
        )
        prediction, target = dense_stage1_prediction(
            predictor, model, teacher, tokens
        )
        assert prediction.shape == target.shape == (2, 9, 16)
        rollout = recurrent_rollout_loss(
            model=model, predictor=predictor, input_ids=tokens,
            start_index=2, horizon=3, capture=capture,
        )
    assert torch.isfinite(rollout.total)
    assert len(rollout.per_step) == 3
    assert 0 <= rollout.top20_agreement <= 1


def test_upper_stack_accepts_per_example_positions_and_key_mask():
    transformers = __import__("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )).eval()
    tokens = torch.randint(0, 32, (2, 3))
    cache = model(tokens, use_cache=True, return_dict=True).past_key_values
    runner = UpperStackRunner(model, target_layer=2, source_layers=(3, 4))
    states, logits = runner.step(
        torch.randn(2, 1, 16), past_key_values=cache,
        position_index=torch.tensor([[3], [7]]),
        key_valid=torch.tensor([
            [True, True, True, True],
            [False, True, True, True],
        ]),
    )
    assert states[4].shape == (2, 1, 16)
    assert logits.shape == (2, 1, 32)


def test_jump_trajectory_retains_each_visited_source_state():
    transformers = __import__("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )).eval()
    predictor = ActionConditionedTransition(TransitionConfig(
        hidden_size=16, source_layers=(3, 4), target_layer=2,
        action_dim=16, variant="full",
    )).eval()
    capture = ResidualCapture(model, (2, 3, 4))
    actions, states = generate_jump_trajectory(
        model=model, predictor=predictor,
        prompt_ids=torch.randint(0, 32, (2, 3)), action_count=5,
        capture=capture, temperature=0.0,
    )
    assert actions.shape == (2, 5)
    assert states[3].shape == states[4].shape == (2, 5, 16)
    assert not states[3].requires_grad
    capture.__exit__(None, None, None)


def test_active_segment_mask_excludes_prior_packed_documents():
    positions = torch.tensor([
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5],
    ])
    assert active_segment_key_mask(positions, 4).tolist() == [
        [False, False, False, True, True],
        [True, True, True, True, True],
    ]


def test_speculative_jump_verification_preserves_exact_greedy_tokens():
    transformers = __import__("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )).eval()
    predictor = ActionConditionedTransition(TransitionConfig(
        hidden_size=16, source_layers=(3, 4), target_layer=2,
        action_dim=16, variant="full",
    )).eval()
    capture = ResidualCapture(model, (2, 3, 4))
    prompt = torch.randint(0, 32, (1, 4))
    expected = full_greedy_decode(model, prompt, 5)
    actual, statistics = speculative_greedy_decode(
        model, predictor, capture, prompt, 5, draft_length=2
    )
    torch.testing.assert_close(actual, expected)
    assert statistics["verifier_passes"] >= 1
    capture.__exit__(None, None, None)


def test_jump_planning_generates_actions_and_leaves_on_recurrent_path():
    transformers = __import__("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
    )).eval()
    predictor = ActionConditionedTransition(TransitionConfig(
        hidden_size=16, source_layers=(3, 4), target_layer=2,
        action_dim=16, variant="full",
    )).eval()
    capture = ResidualCapture(model, (2, 3, 4))
    chunks, log_probability, leaves = generate_jump_candidates(
        model, predictor, capture, torch.randint(0, 32, (1, 3)),
        torch.randint(0, 32, (1, 2)), count=3, length=4,
        temperature=0.8, top_p=0.95,
    )
    assert chunks.shape == (3, 4)
    assert log_probability.shape == (3,)
    assert leaves.shape == (3, 32)
    capture.__exit__(None, None, None)


def test_numeric_verifier_accepts_equivalent_answer_formats():
    assert answers_equal("1,024.0", "1024")
    assert not answers_equal("1024", "1025")
