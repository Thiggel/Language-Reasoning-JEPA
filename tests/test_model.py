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


def test_default_predictor_is_causal_and_uses_history(setup):
    """Past states affect later predictions; future states cannot leak back."""
    _, _, model = setup
    predictor = model.core.predictor.eval()
    assert getattr(predictor, "causal_sequence", False)
    states = torch.randn(2, 5, 64)
    actions = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    reference = predictor(states, actions, valid)

    changed_past = states.clone()
    changed_past[:, 0] += 3.0
    past_output = predictor(changed_past, actions, valid)
    assert not torch.allclose(reference[:, 4], past_output[:, 4])

    changed_future = states.clone()
    changed_future[:, 4] += 3.0
    future_output = predictor(changed_future, actions, valid)
    torch.testing.assert_close(reference[:, :4], future_output[:, :4])


def test_causal_rollout_retains_observed_prefix(setup):
    _, _, model = setup
    predictor = model.core.predictor.eval()
    states = torch.randn(2, 3, 64)
    past_actions = torch.randn(2, 2, 8)
    future_actions = torch.randn(2, 2, 8)
    rollout = predictor.rollout(
        states[:, -1],
        future_actions,
        state_history=states,
        action_history=past_actions,
    )
    manual_first = predictor(
        states,
        torch.cat([past_actions, future_actions[:, :1]], dim=1),
    )[:, -1]
    torch.testing.assert_close(rollout[:, 0], manual_first)

    changed = states.clone()
    changed[:, 0] += 2.0
    changed_rollout = predictor.rollout(
        changed[:, -1],
        future_actions,
        state_history=changed,
        action_history=past_actions,
    )
    assert not torch.allclose(rollout[:, 0], changed_rollout[:, 0])


def test_causal_counterfactuals_use_independent_true_prefixes(setup):
    _, _, model = setup
    core = model.core.eval()
    batch, steps, alternatives = 2, 4, 3
    states = torch.randn(batch, steps, 64)
    actions = torch.randn(batch, steps, 8)
    alt_actions = torch.randn(batch, steps, alternatives, 8)
    valid = torch.ones(batch, steps, dtype=torch.bool)

    actual = core._predict_counterfactuals(
        states, actions, alt_actions, valid
    )
    assert actual.shape == (batch, steps, alternatives, 64)

    expected = torch.empty_like(actual)
    for step in range(steps):
        for alternative in range(alternatives):
            prefix_actions = actions[:, :step + 1].clone()
            prefix_actions[:, step] = alt_actions[:, step, alternative]
            expected[:, step, alternative] = core.predictor(
                states[:, :step + 1],
                prefix_actions,
                valid[:, :step + 1],
            )[:, -1]
    torch.testing.assert_close(actual, expected)


def test_flat_model_freezes_all_hierarchy_parameters(setup):
    vocab, _, _ = setup
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        predictor_heads=2, high_predictor_heads=2, macro_k=0,
    )
    hierarchy_modules = (
        model.core.macro_encoder,
        model.core.hi_predictor,
        model.core.hi_value_head,
        model.core.macro_value_head,
        model.core.macro_support_head,
        model.core.action_support_head,
        model.core.subgoal_action_head,
        model.core.controller_remaining_head,
        model.core.controller_residual_head,
    )
    assert all(
        not parameter.requires_grad
        for module in hierarchy_modules
        for parameter in module.parameters()
    )


def test_controller_outcome_heads(setup):
    _, _, model = setup
    state = torch.randn(7, 64)
    initial = torch.randn(7, 64)
    subgoal = torch.randn(7, 64)
    remaining = model.core.controller_remaining_head(
        state, initial, subgoal
    )
    residual = model.core.controller_residual_head(
        state, initial, subgoal
    )
    assert remaining.shape == residual.shape == (7,)
    loss = remaining.mean() + residual.mean()
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.core.controller_remaining_head.parameters()
    )


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


def test_geo_projection(setup):
    from textjepa.objectives import GoalMonotonicity, TemporalStraightening

    vocab, batch, _ = setup
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4, geo_proj=True, value_detach=False,
    )
    out = model(batch)
    assert "geo_states" in out.extras and "geo_states_tgt" in out.extras
    loss = TemporalStraightening()(out, batch) + GoalMonotonicity()(out, batch)
    loss.backward()
    assert any(p.grad is not None for p in model.core.geo_head.parameters())


def test_ranking_and_distill(setup):
    from textjepa.objectives import (
        ActionRanking, CounterfactualOutcomePrediction, ValueDistill,
    )

    vocab, _, _ = setup
    ds = IGSMDataset(vocab, size=6, seed=0, n_alt=3)
    batch = collate([ds[i] for i in range(6)], vocab.pad_id)
    assert batch["alt_tokens"].shape[:2] == batch["step_mask"].shape
    assert batch["alt_tokens"].shape[2] == 3
    assert batch["alt_step_tokens"].shape[:3] == batch["alt_tokens"].shape[:3]
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2, state_layers=2, state_heads=2,
        d_action=8, d_macro=4, value_detach=False,
    )
    out = model(batch)
    assert out.extras["alt_value"].shape == batch["alt_remaining"].shape
    assert out.extras["cf_chunk_pred"].shape == out.extras["cf_chunk_tgt"].shape
    for obj in (ActionRanking(), ValueDistill()):
        loss = obj(out, batch)
        assert torch.isfinite(loss) and loss >= 0
    ActionRanking()(out, batch).backward()
    grads = [p.grad for p in model.core.predictor.parameters() if p.grad is not None]
    assert grads, "ranking gave the predictor no gradient"
    model.zero_grad(set_to_none=True)
    CounterfactualOutcomePrediction()(model(batch), batch).backward()
    assert any(p.grad is not None for p in model.core.predictor.parameters())


def test_macro_counterfactual_dynamics_value_ranking_and_support(setup):
    from textjepa.objectives import (
        MacroActionValue,
        MacroAdvantageRanking,
        MacroCounterfactualDynamics,
        HierarchyReachability,
        LowerHierarchyRollout,
        DenseHierarchyValueRegression,
        HierarchyValueRegression,
        MacroStateAdvantageRanking,
        MacroStateValue,
        MacroSupport,
        MacroTop1Ranking,
        DenseRolloutPrediction,
        DenseHierarchyRolloutPrediction,
        MacroRecedingRanking,
        MacroRecedingValue,
        SubgoalActionRanking,
    )

    vocab, _, _ = setup
    ds = IGSMDataset(
        vocab, size=8, seed=29, macro_alt_k=3, macro_alt_horizon=3
    )
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        high_predictor_kind="causal", high_predictor_layers=2,
        high_predictor_heads=2,
        dense_rollout_depth=3,
        high_dense_rollout_depth=3,
    )
    out = model(batch)
    A = batch["macro_alt_action_tokens"].shape[1]
    assert out.extras["macro_cf_pred"].shape == (8, A, 64)
    assert out.extras["macro_cf_action_value"].shape == (8, A)
    assert out.extras["macro_cf_support_pos"].shape == (8, A)
    assert out.extras["subgoal_action_cost"].shape == (8, A, A)
    assert out.hi_preds.shape[1] == (
        batch["step_mask"].shape[1] // 3
    )
    assert [prediction.shape[1] for prediction in out.extras[
        "dense_rollout_predictions"
    ]] == [out.preds.shape[1] - offset for offset in range(3)]
    assert [prediction.shape[1] for prediction in out.extras[
        "high_dense_rollout_predictions"
    ]] == [
        out.hi_preds.shape[1] - offset
        for offset in range(min(3, out.hi_preds.shape[1]))
    ]
    assert out.extras["hi_remaining_target"].shape == out.hi_mask.shape
    losses = [
        MacroCounterfactualDynamics()(out, batch),
        HierarchyReachability()(out, batch),
        LowerHierarchyRollout()(out, batch),
        MacroStateValue()(out, batch),
        MacroStateAdvantageRanking()(out, batch),
        MacroActionValue()(out, batch),
        MacroAdvantageRanking()(out, batch),
        MacroSupport()(out, batch),
        MacroTop1Ranking()(out, batch),
        DenseRolloutPrediction(horizon_discount=0.7)(out, batch),
        DenseHierarchyRolloutPrediction(horizon_discount=0.7)(out, batch),
        HierarchyValueRegression()(out, batch),
        DenseHierarchyValueRegression(horizon_discount=0.7)(out, batch),
        MacroRecedingValue()(out, batch),
        MacroRecedingRanking()(out, batch),
        SubgoalActionRanking()(out, batch),
    ]
    assert all(torch.isfinite(loss) and loss >= 0 for loss in losses)
    sum(losses).backward()
    assert any(p.grad is not None for p in model.core.hi_predictor.parameters())
    assert any(p.grad is not None for p in model.core.macro_value_head.parameters())
    assert any(p.grad is not None for p in model.core.macro_support_head.parameters())
    assert any(
        p.grad is not None
        for p in model.core.subgoal_action_head.parameters()
    )
    assert any(p.grad is not None for p in model.core.predictor.parameters())


def test_action_feasibility_head_scores_all_problem_actions(setup):
    from textjepa.objectives import ActionFeasibility

    vocab, _, _ = setup
    ds = IGSMDataset(
        vocab, size=8, seed=43, all_action_supervision=True
    )
    batch = collate([ds[i] for i in range(8)], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
    )
    out = model(batch)
    assert out.extras["action_support_logits"].shape == batch[
        "action_feasible"
    ].shape
    loss = ActionFeasibility()(out, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.core.action_support_head.parameters()
    )


def test_multistep_geometric_rollout_ranking(setup):
    from textjepa.objectives import GeoAdvantageRank

    vocab, _, _ = setup
    ds = IGSMDataset(
        vocab, size=4, seed=13, geo_rank_k=2,
        geo_rank_horizon=4, geo_rank_rollouts=2,
    )
    batch = collate([ds[i] for i in range(4)], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        value_detach=False,
    )
    out = model(batch)
    assert out.extras["ga_label"].shape == (4, 3)
    assert out.extras["ga_rollout_distance"].shape == (4, 3, 2)
    loss = GeoAdvantageRank()(out, batch)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert any(p.grad is not None for p in model.core.value_head.parameters())


def test_geometry_greedy_rollout_ranking(setup):
    from textjepa.objectives import GeoAdvantageRank

    vocab, _, _ = setup
    ds = IGSMDataset(
        vocab, size=4, seed=19, geo_rank_k=2,
        geo_rank_horizon=3, geo_rank_policy="greedy",
        geo_rank_beam_width=2,
    )
    batch = collate([ds[i] for i in range(4)], vocab.pad_id)
    assert batch["ga_greedy"] is True
    assert batch["ga_beam_width"] == 2
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        value_detach=False,
    )
    out = model(batch)
    assert out.extras["ga_label"].shape == (4, 3)
    assert out.extras["ga_greedy_distance"].shape == (4, 3)
    assert torch.isfinite(out.extras["ga_label"][out.extras["ga_valid"]]).all()
    loss = GeoAdvantageRank()(out, batch)
    assert torch.isfinite(loss) and loss >= 0


def test_observed_action_ldad_reconstructs_raw_tokens(setup):
    from textjepa.objectives import ObservedActionLDAD

    vocab, batch, _ = setup
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        observed_action_ldad=True,
    )
    out = model(batch)
    logits = out.extras["observed_action_logits"]
    assert logits.shape[:2] == batch["action_tokens"].shape[:2]
    assert logits.shape[-1] == len(vocab)
    loss = ObservedActionLDAD()(out, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        p.grad is not None for p in model.observed_action_decoder.parameters()
    )
    assert any(p.grad is not None for p in model.state_model.parameters())


def test_multistep_observed_action_ldad_reconstructs_ordered_phrases(setup):
    from textjepa.objectives import ObservedActionLDAD

    vocab, batch, _ = setup
    horizon = 3
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        observed_action_ldad=True, observed_action_ldad_horizon=horizon,
    )
    out = model(batch)
    logits = out.extras["observed_action_multistep_logits"]
    assert logits.shape[0] == batch["action_tokens"].shape[0]
    assert logits.shape[1] == batch["action_tokens"].shape[1] - horizon + 1
    assert logits.shape[-3] == horizon
    assert logits.shape[-1] == len(vocab)
    loss = ObservedActionLDAD()(out, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        p.grad is not None for p in model.observed_action_decoder.parameters()
    )
    assert any(p.grad is not None for p in model.state_model.parameters())


def test_token_bottleneck_action_encoder(setup):
    vocab, batch, _ = setup
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        max_chunk_len=48, action_encoder_kind="token_bottleneck",
        action_token_dim=4,
    )
    out = model(batch)
    assert out.actions.shape == (*batch["action_tokens"].shape[:2], 8)
    assert torch.isfinite(out.actions).all()
    out.preds.square().mean().backward()
    assert any(
        p.grad is not None for p in model.action_encoder.parameters()
    )


def test_shuffled_actions_deterministic(setup):
    vocab, _, _ = setup
    ds = IGSMDataset(vocab, size=4, seed=0, shuffle_actions=True)
    a, b = ds[1], ds[1]
    assert a["actions"] == b["actions"]  # same index -> same shuffle


def test_sentence_stream_variational_and_ldad(setup):
    from textjepa.models import SentenceStreamVJEPA
    from textjepa.objectives import (
        ActionKL, LatentDifferenceActionReconstruction, SIGReg,
        TargetDistributionKL, VariationalLatentPrediction,
    )

    vocab, batch, _ = setup
    model = SentenceStreamVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, latent_ldad=True, use_ema_target=False,
    )
    out = model(batch)
    expected = batch["prompt_mask"].sum(1) + batch["step_mask"].sum(1) - 1
    assert torch.equal(out.step_mask.sum(1), expected)
    assert out.extras["pred_logvar"].shape == out.preds.shape
    assert out.extras["target_logvar"].shape == out.preds.shape
    assert "latent_ldad_pred" in out.extras
    losses = [
        VariationalLatentPrediction()(out, batch),
        TargetDistributionKL()(out, batch),
        ActionKL()(out, batch),
        LatentDifferenceActionReconstruction()(out, batch),
        SIGReg(num_slices=16, n_points=9)(out, batch),
    ]
    assert all(torch.isfinite(x) and x >= 0 for x in losses)
    sum(losses).backward()
    assert any(p.grad is not None for p in model.chunk_encoder.parameters())


def test_sentence_stream_online_stopgrad_target(setup):
    from textjepa.models import SentenceStreamVJEPA

    vocab, batch, _ = setup
    model = SentenceStreamVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode="online_sg",
    )
    out = model(batch)
    assert out.step_states.requires_grad
    assert not out.step_states_tgt.requires_grad
    assert out.extras["target_mode"] == "online_sg"


def test_sentence_stream_mixture_action_prior(setup):
    from textjepa.models import SentenceStreamVJEPA
    from textjepa.objectives import ActionKL

    vocab, batch, _ = setup
    model = SentenceStreamVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode="online_sg",
        action_prior_components=4,
    )
    out = model(batch)
    assert out.extras["action_p_mu"].shape == out.actions.shape
    samples = model.var_action.sample_prior(out.prev_states.detach(), k=7)
    assert samples.shape == (*out.prev_states.shape[:-1], 7, 8)
    assert torch.isfinite(samples).all()
    loss = ActionKL()(out, batch)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert any(p.grad is not None for p in model.var_action.prior.parameters())


@pytest.mark.parametrize("prior_components", [1, 4])
def test_sentence_stream_counterfactual_outcome_set(setup, prior_components):
    from textjepa.models import SentenceStreamVJEPA
    from textjepa.objectives import (
        CounterfactualActionPrior,
        CounterfactualVariationalPrediction,
    )

    vocab, _, _ = setup
    dataset = IGSMDataset(vocab, size=4, seed=23, n_alt=3)
    batch = collate([dataset[i] for i in range(4)], vocab.pad_id)
    model = SentenceStreamVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode="online_sg",
        action_prior_components=prior_components,
        counterfactual_set=True,
    )
    out = model(batch)
    n_valid = int((batch["alt_remaining"] >= 0).sum())
    assert out.extras["counterfactual_variational_nll"].shape == (n_valid,)
    assert out.extras["counterfactual_action_kl"].shape == (n_valid,)
    prediction = CounterfactualVariationalPrediction()(out, batch)
    prior = CounterfactualActionPrior()(out, batch)
    assert torch.isfinite(prediction) and prediction >= 0
    assert torch.isfinite(prior) and prior >= 0
    (prediction + 0.1 * prior).backward()
    assert any(p.grad is not None for p in model.transition.parameters())
    assert any(p.grad is not None for p in model.var_action.prior.parameters())


def test_counterfactual_histories_use_bounded_encoder_batches(setup, monkeypatch):
    from textjepa.models import SentenceStreamVJEPA

    vocab, _, _ = setup
    model = SentenceStreamVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode="online_sg",
    )
    batch_sizes = []

    def fake_encode(tokens, mask, teacher):
        batch_sizes.append(tokens.shape[0])
        return torch.zeros(tokens.shape[0], tokens.shape[1], 64)

    monkeypatch.setattr(model, "_encode", fake_encode)
    tokens = torch.zeros(600, 17, 24, dtype=torch.long)
    mask = torch.ones(600, 17, dtype=torch.bool)
    encoded = model._encode_stream_batches(tokens, mask, teacher=False)
    assert batch_sizes == [256, 256, 88]
    assert encoded.shape == (600, 17, 64)


@pytest.mark.parametrize("action_mode", ["latent", "pooled", "token_bottleneck"])
def test_variational_discourse_action_modes(setup, action_mode):
    from textjepa.models import DiscourseVJEPA
    from textjepa.objectives import (
        ActionKL, ObservedActionLDAD, TargetDistributionKL,
        VariationalLatentPrediction,
    )

    vocab, batch, _ = setup
    observed = action_mode != "latent"
    model = DiscourseVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode="online_sg",
        action_mode=action_mode, action_token_dim=4,
        observed_action_ldad=observed,
    )
    out = model(batch)
    assert out.preds.shape == out.step_states.shape
    assert out.extras["pred_logvar"].shape == out.preds.shape
    assert out.extras["action_mode"] == action_mode
    loss = (
        VariationalLatentPrediction()(out, batch)
        + TargetDistributionKL()(out, batch)
        + ActionKL()(out, batch)
    )
    if observed:
        assert "observed_action_logits" in out.extras
        loss = loss + ObservedActionLDAD()(out, batch)
    loss.backward()
    assert any(p.grad is not None for p in model.transition.parameters())
    if observed:
        assert any(
            p.grad is not None for p in model.observed_action_decoder.parameters()
        )


@pytest.mark.parametrize(
    ("target_mode", "target_requires_grad"),
    [("ema", False), ("online_sg", False), ("online_grad", True)],
)
def test_observed_action_variational_target_modes(
    setup, target_mode, target_requires_grad
):
    """The faithful variational LDAD factorial must change only the declared
    target gradient path while retaining the observed raw-action decoder."""
    from textjepa.models import DiscourseVJEPA
    from textjepa.objectives import ObservedActionLDAD, VariationalLatentPrediction

    vocab, batch, _ = setup
    model = DiscourseVJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8,
        max_chunk_len=48, target_mode=target_mode,
        action_mode="pooled", observed_action_ldad=True,
    )
    out = model(batch)
    assert out.step_states_tgt.requires_grad is target_requires_grad
    assert out.extras["target_mode"] == target_mode
    loss = VariationalLatentPrediction()(out, batch)
    loss = loss + ObservedActionLDAD()(out, batch)
    loss.backward()
    assert any(p.grad is not None for p in model.transition.parameters())
    assert any(
        p.grad is not None for p in model.observed_action_decoder.parameters()
    )
