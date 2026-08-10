from __future__ import annotations

import torch
from torch import nn

from textjepa.analysis.predictive_state import (
    effective_rank,
    progress_metrics,
    spearman,
)
from textjepa.data.predictive_state import (
    pack_documents,
    split_whitespace_documents,
)
from textjepa.models.action_transition import (
    ActionConditionedTransition,
    LoRALinear,
    ResidualCapture,
    TransitionConfig,
    install_upper_lora,
)
from textjepa.objectives.predictive_state import transition_loss
from textjepa.planning.predictive_state import (
    GoalDistanceModel,
    block_beam_search,
    goal_distance_loss,
    potential_shaped_reward,
)
from textjepa.training.predictive_state import (
    dense_stage1_prediction,
    recurrent_rollout_loss,
    teacher_forward,
)


def test_document_packing_masks_only_cross_document_transition():
    packed = pack_documents(
        [[1, 2], [3, 4]], context_length=6, eos_token_id=9
    )
    assert packed["input_ids"].tolist() == [[1, 2, 9, 3, 4, 9]]
    assert packed["target_mask"].tolist() == [
        [True, True, False, True, True]
    ]


def test_whitespace_only_lines_split_wikitext_documents():
    text = " \n = Heading = \n \n First paragraph.\n \t\nSecond paragraph.\n"
    assert split_whitespace_documents(text) == [
        "= Heading =", "First paragraph.", "Second paragraph."
    ]


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


def test_transition_loss_separates_direction_and_scale():
    target = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    equal_direction = transition_loss(2 * target, target, mask)
    assert equal_direction.cosine.item() < 1e-6
    assert equal_direction.scale.item() > 0
    exact = transition_loss(target, target, mask)
    assert exact.total.item() < 1e-6


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
