from functools import partial
import math

import torch
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.pooled_sentence_jepa import (
    CausalAttentionPooler, PooledSentenceJEPA,
)
from scripts.eval_pooled_sentence_planning import (
    beam_plan, summarize_drift, summarize_examples, validate_generated_trace,
)
from scripts.train_pooled_sentence_jepa import (
    accumulation_group_size, compute_losses, optimizer_step_count,
)
from omegaconf import OmegaConf


def _batch(size=2):
    vocab = build_vocab(23)
    ds = SemanticBoundaryLMDataset(
        vocab, size=size, seed=71, boundary_mode="semantic", modulus=23,
        n_vars_range=(8, 10), leaf_prob=0.35, steps_range=(4, 6),
        distractor_prob=0.0, max_distractors=0,
    )
    return next(iter(DataLoader(
        ds, batch_size=size,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    ))), vocab


def _model(vocab, scope="sentence", decoder=True):
    return PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], d_state=32, encoder_layers=1,
        pool_heads=4, predictor_layers=1, n_heads=4, ff_mult=2,
        max_len=768, d_action=8, dense_depth=2, pooling_scope=scope,
        use_token_prior=True, use_prefix_decoder=decoder,
        decoder_dim=24, decoder_layers=1, decoder_heads=4,
        decoder_max_len=64, decoder_prefixes_per_sequence=3,
    )


def _visreg_model(vocab):
    return PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], d_state=32, encoder_layers=1,
        pool_heads=4, predictor_layers=1, n_heads=4, ff_mult=2,
        max_len=768, d_action=8, dense_depth=2, pooling_scope="sentence",
        use_token_prior=True, use_prefix_decoder=False,
        target_mode="visreg", visreg_projections=16,
    )


def test_pooler_sentence_mask_excludes_previous_segment_but_global_does_not():
    torch.manual_seed(1)
    hidden = torch.randn(1, 6, 16)
    tokens = torch.tensor([[2, 3, 10, 4, 5, 6]])
    changed = hidden.clone()
    changed[:, :3] += 100
    sentence = CausalAttentionPooler(16, 4, "sentence", (10, 15)).eval()
    global_pool = CausalAttentionPooler(16, 4, "global", (10, 15)).eval()
    with torch.no_grad():
        local_a = sentence(hidden, tokens, pad_id=0)
        local_b = sentence(changed, tokens, pad_id=0)
        global_a = global_pool(hidden, tokens, pad_id=0)
        global_b = global_pool(changed, tokens, pad_id=0)
    assert torch.allclose(local_a[:, 3:], local_b[:, 3:], atol=1e-5)
    assert not torch.allclose(global_a[:, 3:], global_b[:, 3:])


def test_fused_pooler_matches_explicit_attention_reference():
    torch.manual_seed(4)
    hidden = torch.randn(2, 7, 16)
    tokens = torch.tensor([
        [2, 3, 10, 4, 5, 0, 0],
        [7, 8, 9, 10, 4, 5, 6],
    ])
    pooler = CausalAttentionPooler(16, 4, "sentence", (10, 15)).eval()
    with torch.no_grad():
        actual = pooler(hidden, tokens, pad_id=0)
        batch, length, dim = hidden.shape
        reshape = lambda value: value.reshape(batch, length, 4, 4).transpose(1, 2)
        query = reshape(pooler.query(hidden + pooler.query_bias))
        key = reshape(pooler.key(hidden))
        value = reshape(pooler.value(hidden))
        score = query @ key.transpose(-1, -2) / math.sqrt(4)
        allowed = pooler._allowed(tokens, 0)[:, None]
        pooled = score.masked_fill(~allowed, -torch.inf).softmax(-1) @ value
        pooled = pooled.transpose(1, 2).reshape(batch, length, dim)
        expected = pooler.norm(pooler.output(pooled))
        expected = expected.masked_fill(tokens.eq(0).unsqueeze(-1), 0.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_pooled_states_and_targets_are_strictly_causal():
    batch, vocab = _batch()
    model = _model(vocab).eval()
    with torch.no_grad():
        original = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    changed = batch["tokens"].clone()
    cut = int(batch["prompt_len"][0]) + 3
    changed[0, cut:] = torch.randint(1, len(vocab), changed[0, cut:].shape)
    with torch.no_grad():
        other = model(changed, batch["prompt_len"], batch["sentence_ends"])
    assert torch.allclose(original["states"][0, :cut], other["states"][0, :cut], atol=1e-5)
    assert torch.allclose(original["target"][0, :3], other["target"][0, :3], atol=1e-5)


def test_next_token_action_predicts_next_pooled_prefix_state():
    batch, vocab = _batch()
    model = _model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    for row in range(len(batch["tokens"])):
        prompt = int(batch["prompt_len"][row])
        count = int(out["lengths"][row])
        assert torch.allclose(out["prev"][row, 0], out["states"][row, prompt - 1])
        assert torch.allclose(
            out["target"][row, :count], out["targets"][row, prompt:prompt + count]
        )
    assert model.predictor.causal_sequence
    assert model.predictor.residual


def test_visreg_mode_reuses_online_states_as_gradient_targets_without_teacher():
    batch, vocab = _batch(1)
    model = _visreg_model(vocab).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    assert model.teacher is None
    assert out["targets"] is out["states"]
    assert out["target"].requires_grad
    out["target"].retain_grad()
    loss = (out["pred"][out["valid"]] - out["target"][out["valid"]]).square().mean()
    loss.backward()
    assert out["target"].grad is not None
    assert out["target"].grad.abs().sum() > 0


def test_visreg_mode_counterfactual_targets_use_online_encoder():
    batch, vocab = _batch(1)
    model = _visreg_model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
        counterfactual = model.token_counterfactuals(
            out, batch["tokens"], batch["prompt_len"], k=3, max_anchors=2,
        )
    assert counterfactual["exact_outcome"].shape[:2] == (2, 3)
    assert torch.isfinite(counterfactual["advantage_target"]).all()


def test_visreg_full_objective_is_finite_and_reaches_online_encoder():
    batch, vocab = _batch(1)
    model = _visreg_model(vocab).train()
    cfg = OmegaConf.load("configs/pooled_sentence_jepa.yaml")
    cfg.objective.vicreg = 0.0
    cfg.objective.visreg = 1.0
    cfg.objective.gar_k = 3
    cfg.objective.gar_max_anchors = 2
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    loss, metrics = compute_losses(out, cfg, model, batch)
    assert torch.isfinite(loss)
    assert "visreg" in metrics and "vicreg" not in metrics
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.state_encoder.parameters()
    )


def test_zero_gar_weight_skips_counterfactual_construction():
    batch, vocab = _batch(1)
    model = _visreg_model(vocab).train()
    cfg = OmegaConf.load("configs/pooled_sentence_jepa.yaml")
    cfg.objective.vicreg = 0.0
    cfg.objective.visreg = 1.0
    cfg.objective.gar = 0.0
    model.token_counterfactuals = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("disabled GAR must not construct counterfactuals")
    )
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    loss, metrics = compute_losses(out, cfg, model, batch)
    assert torch.isfinite(loss)
    assert metrics["gar_regression"] == 0
    loss.backward()


def test_dense_rollout_supervises_every_valid_anchor_with_observed_history():
    batch, vocab = _batch()
    model = _model(vocab, decoder=False).eval()
    model.dense_depth = 4
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    assert len(out["dense_predictions"]) == 4
    for horizon, (predictions, targets, mask) in enumerate(zip(
        out["dense_predictions"], out["dense_targets"],
        out["dense_masks"],
    ), start=1):
        expected_width = out["prev"].shape[1] - horizon + 1
        assert predictions.shape[:2] == (len(batch["tokens"]), expected_width)
        assert torch.equal(targets, out["target"][:, horizon - 1:])
        expected_mask = torch.stack([
            out["valid"][:, start:start + horizon].all(1)
            for start in range(expected_width)
        ], 1)
        assert torch.equal(mask, expected_mask)
        # Each cell must equal a literal open-loop rollout from that anchor,
        # retaining the complete observed causal history before the anchor.
        for start in range(expected_width):
            explicit = model.predictor.rollout(
                out["prev"][:, start],
                out["actions"][:, start:start + horizon],
                state_history=out["prev"][:, :start + 1],
                action_history=out["actions"][:, :start],
            )[:, -1]
            rows = mask[:, start]
            assert torch.allclose(
                predictions[rows, start], explicit[rows], atol=1e-5, rtol=1e-4
            )


def test_dense_rollout_depth_one_is_exactly_the_teacher_forced_prediction():
    batch, vocab = _batch()
    model = _model(vocab, decoder=False).eval()
    model.dense_depth = 1
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    assert len(out["dense_predictions"]) == 1
    assert torch.equal(out["dense_predictions"][0], out["pred"])
    assert torch.equal(out["dense_targets"][0], out["target"])
    assert torch.equal(out["dense_masks"][0], out["valid"])


def test_checkpointed_dense_rollout_backpropagates_through_predictor():
    batch, vocab = _batch(1)
    model = _model(vocab, decoder=False).train()
    model.dense_depth = 4
    model.dense_checkpoint = True
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    loss = sum(
        prediction[mask].square().mean()
        for prediction, mask in zip(
            out["dense_predictions"], out["dense_masks"]
        ) if mask.any()
    )
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.predictor.parameters()
    )


def test_prefix_decoder_is_causal_conditioned_and_reaches_pooler():
    batch, vocab = _batch()
    model = _model(vocab, decoder=True).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    decoded = model.prefix_decoder_batch(
        out, batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
    )
    assert decoded["logits"].shape[:2] == decoded["targets"].shape
    assert decoded["valid"].any()
    assert not torch.allclose(decoded["logits"], decoded["shuffled_logits"])
    loss = torch.nn.functional.cross_entropy(
        decoded["logits"][decoded["valid"]], decoded["targets"][decoded["valid"]]
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.state_encoder.pooler.parameters()
    )


def test_prefix_decoder_cannot_see_future_teacher_forced_tokens():
    batch, vocab = _batch()
    model = _model(vocab, decoder=True).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
        decoded = model.prefix_decoder_batch(
            out, batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
        )
        changed = decoded["targets"].clone()
        changed[:, -1] = (changed[:, -1] + 1) % len(vocab)
        other = model.prefix_decoder(
            decoded["states"], changed, decoded["valid"]
        )
    # The last target is shifted into no earlier decoder input.
    assert torch.allclose(decoded["logits"][:, :-1], other[:, :-1], atol=1e-6)


def test_all_trainable_transformers_are_dropout_free():
    _, vocab = _batch(1)
    model = _model(vocab, decoder=True)
    dropouts = [module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropouts and max(dropouts) == 0.0


def test_default_model_is_approximately_fifty_million_parameters():
    _, vocab = _batch(1)
    for decoder in (False, True):
        model = PooledSentenceJEPA(
            len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
            question_id=vocab.token_to_id["?"], use_prefix_decoder=decoder,
        )
        # The frozen exponential-moving-average target encoder is a training
        # target, not extra trainable/inference capacity.
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 40_000_000 <= parameters <= 60_000_000


def test_planning_summary_reports_uncertainty_and_error_position():
    result = summarize_examples(
        generated=[[1, 2, 3, 4], [1, 9, 3, 4]],
        references=[[1, 2, 3, 4], [1, 2, 3, 4]],
        boundary_ids={4}, bootstrap_seed=7,
    )
    assert result["token_accuracy"] == 0.875
    assert result["exact_trace_success"] == 0.5
    assert result["boundary_token_accuracy"] == 1.0
    assert result["mean_first_error_fraction"] == 0.625
    assert result["token_accuracy_ci95"][0] <= 0.875 <= result["token_accuracy_ci95"][1]
    assert len(result["position_quartile_accuracy"]) == 4


def test_beam_plan_returns_each_selected_predicted_state_for_drift_audit():
    batch, vocab = _batch(1)
    model = _model(vocab, decoder=False).eval()
    prompt_len = int(batch["prompt_len"][0])
    prefix = batch["tokens"][0, :prompt_len].tolist()
    full_len = int(batch["tokens"][0].ne(vocab.pad_id).sum())
    with torch.no_grad():
        goal = model.teacher(batch["tokens"][:, :full_len])[:, -1]
        plan = beam_plan(
            model, prefix, goal, depth=2, width=2, score_mode="value",
            proposal_mode="prior", proposal_topk=3, prompt_length=prompt_len,
        )
    assert len(plan["tokens"]) == 2
    assert plan["predicted_states"].shape == (2, model.d_state)
    summary = summarize_drift([
        {"offset": 1, "normalized_mse": 1.0, "raw_mse": 2.0,
         "cosine_distance": 0.5},
        {"offset": 1, "normalized_mse": 3.0, "raw_mse": 4.0,
         "cosine_distance": 1.5},
    ])
    assert summary["1"] == {
        "normalized_mse": 2.0, "raw_mse": 3.0,
        "cosine_distance": 1.0,
    }


def test_posthoc_generation_validator_accepts_valid_alternative_trace_only():
    batch, vocab = _batch(1)
    # Reconstruct the deterministic raw item used by _batch.
    ds = SemanticBoundaryLMDataset(
        vocab, size=1, seed=71, boundary_mode="semantic", modulus=23,
        n_vars_range=(8, 10), leaf_prob=0.35, steps_range=(4, 6),
        distractor_prob=0.0, max_distractors=0,
    )
    item = ds[0]
    problem, _ = ds.igsm.problem(0)
    reasoning = item["tokens"][item["prompt_len"]:]
    valid = validate_generated_trace(reasoning, problem, vocab)
    assert valid["solved"]
    assert valid["valid_prefix_sentences"] > 0
    corrupted = list(reasoning)
    corrupted[0] = vocab.token_to_id["answer"]
    invalid = validate_generated_trace(corrupted, problem, vocab)
    assert not invalid["solved"]
    assert invalid["first_invalid_sentence"] == 0


def test_gradient_accumulation_preserves_optimizer_step_count_and_tail_scale():
    assert optimizer_step_count(2500, 2) == 1250
    assert optimizer_step_count(5000, 4) == 1250
    assert optimizer_step_count(10, 4) == 3
    assert [accumulation_group_size(i, 10, 4) for i in range(10)] == [
        4, 4, 4, 4, 4, 4, 4, 4, 2, 2,
    ]
