import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _summarize(tmp_path, backbone_mode):
    tmp_path.mkdir(parents=True, exist_ok=True)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "variant": "full",
        "backbone_mode": backbone_mode,
        "model_id": "Qwen/Qwen2.5-0.5B",
        "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "evaluations": {"100": {
            "predictor_removed_nll": 2.9,
            "transition_cosine_loss": 0.2,
            "permuted_action_cosine_loss": 0.4,
            "target_geometry": {"effective_rank": 2.1},
        }},
    }))
    output = tmp_path / "run_summary.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_predictive_state_stage1.py"),
         "--metrics", str(metrics), "--output", str(output)],
        check=True, capture_output=True,
    )
    return json.loads(output.read_text())


def test_adapted_cell_is_not_summarized_as_a_frozen_diagnostic(tmp_path):
    # A LoRA cell must not inherit the frozen diagnostic's claim string, and a
    # single arm never carries a representation claim by itself.
    frozen = _summarize(tmp_path / "frozen", "frozen")
    assert frozen["scientific_validity"] == "diagnostic_only"
    assert "frozen-backbone diagnostic" in frozen["claim"]

    adapted = _summarize(tmp_path / "lora", "lora")
    assert adapted["scientific_validity"] == "not_admitted_pending_controls"
    assert "frozen" not in adapted["claim"]
    assert "NTP-only" in adapted["claim"]


def test_stage1_screen_cell_shares_one_token_block_file_and_freezes_the_target():
    text = (ROOT / "scripts/run_predictive_state_stage1_screen.sh").read_text()
    # Every arm must read the same pre-built tensors rather than rebuild them.
    assert "PREDICTIVE_STATE_TOKEN_BLOCKS:?" in text
    assert "prepare_predictive_state_corpus.py" not in text
    # The mode is parameterized now, but lora stays the default.
    assert '--backbone-mode "${PREDICTIVE_STATE_BACKBONE_MODE:-lora}"' in text


def test_stage1_screen_launcher_matches_tokens_and_isolates_the_objective():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_stage1_screen.sh").read_text()
    assert "gruenau-gpus" in text
    assert "memory.used" in text and "utilization.gpu" in text
    assert "refusing busy GPU" in text
    assert "run_summary.json" in text
    # The NTP-only arm is what makes any NLL movement attributable.
    assert "ntp_only" in text
    for variant in ("no_action", "action_only"):
        assert variant in text


def test_gruenau_launcher_enforces_dashboard_and_direct_gpu_admission():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_diagnostics_v1.sh").read_text()
    assert "gruenau-gpus" in text
    assert "memory.used" in text and "utilization.gpu" in text
    assert "refusing busy GPU" in text
    assert 'print (($2 + 0 < 1024 && $3 + 0 < 10) ? "FREE" : "BUSY")' in text
    assert "run_summary.json" in text


def test_slurm_wrapper_validates_command_and_records_failure_provenance():
    text = (ROOT / "scripts/slurm_predictive_state.sbatch").read_text()
    assert "[[ $# -gt 0 ]]" in text
    assert "trap" in text
    assert "resolved_config.json" in text
    assert "environment.json" in text
    assert "run_summary.json" in text


def test_object_labels_track_first_and_repeat_mentions(tmp_path):
    # The discourse labels drive every probe, so a silent error here would look
    # like a representation finding. First mention is new, later ones given,
    # and recency counts tokens back to the previous mention of that object.
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location(
        "probe_entity_state",
        ROOT / "scripts/probe_predictive_state_entity_state.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def decode(self, ids):
            return {5: "planet", 6: "the", 7: "orbit"}.get(ids[0], "xx")

    class Args:
        minimum_mentions = 3
        entities_per_block = 4

    #                 0  1  2  3  4  5  6  7
    blocks = torch.tensor([[5, 6, 7, 5, 6, 7, 5, 7]])
    labels = module.object_labels(Tokenizer(), blocks, Args())

    # "the" is a stopword; "planet" and "orbit" each occur three times, and the
    # count tie is broken by token id, so "orbit" (7) takes rank 0.
    assert labels["entity"][0].tolist() == [1, -1, 0, 1, -1, 0, 1, 0]
    assert labels["given_new"][0].tolist() == [0, -1, 0, 1, -1, 1, 1, 1]
    assert labels["recency"][0].tolist() == [-1, -1, -1, 3, -1, 3, 3, 2]
    # The prefix set is what the state could know before the current token.
    assert labels["prefix_set"][0, 0].tolist() == [0, 0, 0, 0]
    assert labels["prefix_set"][0, 3].tolist() == [1, 1, 0, 0]


def test_projection_bottleneck_narrows_every_predictor_input_path():
    # The skip path must be routed through the projection too, otherwise a
    # "bottlenecked" predictor still receives the full state and the arm tests
    # nothing. Parameter count alone would not catch that.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from textjepa.models.action_transition import (
        ActionConditionedTransition, TransitionConfig,
    )

    def build(projection):
        return ActionConditionedTransition(TransitionConfig(
            hidden_size=896, source_layers=(18, 24), target_layer=12,
            action_dim=896, projection_size=projection, variant="full",
        ))

    wide, narrow = build(None), build(32)
    assert narrow.skip.in_features == 3 * 32
    assert wide.skip.in_features == 3 * (896 // 2)
    for module in (narrow.gate, narrow.value):
        assert module.in_features == 3 * 32
    wide_count = sum(p.numel() for p in wide.parameters())
    narrow_count = sum(p.numel() for p in narrow.parameters())
    assert narrow_count < wide_count / 4


def test_stage1_screen_cell_forwards_an_optional_predictor_bottleneck():
    text = (ROOT / "scripts/run_predictive_state_stage1_screen.sh").read_text()
    assert "PREDICTIVE_STATE_PROJECTION_SIZE" in text
    assert "--projection-size" in text
    # Unset must stay unset rather than passing an empty flag.
    assert 'if [[ -n "${PREDICTIVE_STATE_PROJECTION_SIZE:-}" ]]' in text


def test_pressure_launcher_chains_cells_and_keeps_per_cell_provenance():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_stage1_pressure.sh").read_text()
    assert "gruenau-gpus" in text
    assert "refusing busy GPU" in text
    # Cells sharing a device run sequentially, but each keeps its own markers.
    assert "_chain" in text
    assert "run_summary.json" in text
    assert 'printf %q "$run_dir"' in text
    # Both pressure ladders must be present, anchored at the previous corner.
    for weight in ("0.3", "1.0", "3.0"):
        assert f"lpred{weight}" in text
    for projection in ("32", "8"):
        assert f"proj{projection}" in text


def test_state_bottleneck_leaves_the_action_channel_at_full_width():
    # A single projection width would starve token identity along with the
    # state: eight dimensions cannot separate a 151k-token vocabulary, so the
    # arm would degrade for the wrong reason. The state bottleneck must be
    # independent of the action channel.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from textjepa.models.action_transition import (
        ActionConditionedTransition, TransitionConfig,
    )

    config = TransitionConfig(
        hidden_size=896, source_layers=(18, 24), target_layer=12,
        action_dim=896, projection_size=8, action_projection_size=448,
        variant="full",
    )
    predictor = ActionConditionedTransition(config)
    assert predictor.state_projections["18"].out_features == 8
    assert predictor.state_projections["24"].out_features == 8
    assert predictor.action_projection.out_features == 448
    # Concatenated input is two narrow state channels plus a full action one.
    assert predictor.skip.in_features == 8 + 8 + 448
    # The emitted state is always the full residual width, whatever the input.
    assert predictor.skip.out_features == 896
    assert predictor.output.out_features == 896

    # Omitting the override keeps the old single-knob behaviour.
    shared = ActionConditionedTransition(TransitionConfig(
        hidden_size=896, source_layers=(18, 24), target_layer=12,
        action_dim=896, projection_size=8, variant="full",
    ))
    assert shared.action_projection.out_features == 8
    assert shared.skip.in_features == 24


def test_linear_predictor_drops_the_mlp_and_still_emits_a_full_state():
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from textjepa.models.action_transition import (
        ActionConditionedTransition, TransitionConfig,
    )

    linear = ActionConditionedTransition(TransitionConfig(
        hidden_size=896, source_layers=(18, 24), target_layer=12,
        action_dim=896, projection_size=896, linear_only=True, variant="full",
    ))
    assert linear.output is None and linear.gate is None and linear.value is None
    # Projection at full width keeps the composed map full rank, so a failure
    # cannot be blamed on the projection instead of on linearity.
    assert linear.skip.in_features == 3 * 896
    assert linear.skip.out_features == 896
    # The terminal module still starts near zero.
    assert float(linear.skip.weight.std()) < 1e-2
    prediction = linear(
        {18: torch.randn(2, 5, 896), 24: torch.randn(2, 5, 896)},
        torch.randn(2, 5, 896),
    )
    assert prediction.shape == (2, 5, 896)


def test_single_source_control_is_parameter_matched_to_the_two_source_predictor():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from textjepa.models.action_transition import (
        ActionConditionedTransition, TransitionConfig,
    )

    def count(sources, projection):
        predictor = ActionConditionedTransition(TransitionConfig(
            hidden_size=896, source_layers=sources, target_layer=12,
            action_dim=896, projection_size=projection, variant="full",
        ))
        return sum(p.numel() for p in predictor.parameters()), predictor

    both, wide = count((18, 24), None)
    single, narrow = count((24,), 672)
    # Widening the surviving channel to 672 matches both the parameter count
    # and the concatenated input width, so dropping a state is the only change.
    assert both == single
    assert wide.skip.in_features == narrow.skip.in_features


def test_stage1_screen_cell_forwards_linear_and_single_source_controls():
    text = (ROOT / "scripts/run_predictive_state_stage1_screen.sh").read_text()
    assert "--linear-predictor" in text
    assert "PREDICTIVE_STATE_SOURCE_LAYERS" in text
    assert "--source-layer" in text


def test_pressure_launcher_defers_a_busy_device_without_aborting_the_round():
    # A single occupied GPU must not prevent placement onto the others; the
    # round is otherwise silently truncated at the first busy device.
    text = (ROOT / "scripts/launch_gruenau_predictive_state_stage1_pressure.sh").read_text()
    assert 'if ! check_gpu "$host" "$gpu"; then' in text
    assert "deferred" in text


def test_deferred_runner_reapplies_admission_before_every_cell():
    # The waiter exists so a congested cluster does not truncate a round, but it
    # must not weaken the gate: each cell re-checks memory and utilization
    # immediately before it starts.
    text = (ROOT / "scripts/wait_and_run_predictive_state_cells.sh").read_text()
    assert "memory.used" in text and "utilization.gpu" in text
    assert '"$used" -lt 1024' in text and '"$util" -lt 10' in text
    # Bounded, so a permanently busy device fails visibly instead of hanging.
    assert "PREDICTIVE_STATE_WAIT_CHECKS" in text
    assert "never became free" in text


def test_ntp_weight_scales_only_the_optimized_total():
    # The reported ntp must stay unweighted so held-out language modeling is
    # comparable across cells that optimize it at different strengths.
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from textjepa.objectives.predictive_state import stage1_loss

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 32)
    input_ids = torch.randint(32, (2, 6))
    mask = torch.ones(2, 5, dtype=torch.bool)
    prediction, target = torch.randn(2, 5, 8), torch.randn(2, 5, 8)

    kwargs = dict(logits=logits, input_ids=input_ids, target_mask=mask,
                  prediction=prediction, target_state=target,
                  prediction_weight=1.0, scale_weight=0.0)
    full = stage1_loss(**kwargs, ntp_weight=1.0)
    small = stage1_loss(**kwargs, ntp_weight=0.01)
    assert torch.allclose(full.ntp, small.ntp)
    assert float(small.total) < float(full.total)
    assert torch.allclose(small.total - 0.01 * small.ntp,
                          full.total - full.ntp, atol=1e-5)

    # The predictor-free branch must honour the weight too.
    bare = stage1_loss(logits=logits, input_ids=input_ids, target_mask=mask,
                       prediction=None, target_state=None,
                       prediction_weight=1.0, scale_weight=0.0, ntp_weight=0.5)
    assert torch.allclose(bare.total, 0.5 * bare.ntp)


def test_full_upper_finetuning_keeps_the_target_stack_frozen():
    # Maximum adaptation freedom must still leave the prediction target fixed,
    # or the target can drift to meet the predictor.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from torch import nn
    from textjepa.models.action_transition import (
        backbone_parameters, unfreeze_upper_layers,
    )

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(6))
            self.embed = nn.Embedding(8, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    model = Model()
    accounting = unfreeze_upper_layers(model, first_trainable_layer=4)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("model.layers.3", "model.layers.4",
                               "model.layers.5")) for name in trainable)
    assert not any(name.startswith(("model.layers.0", "model.layers.1",
                                    "model.layers.2", "model.embed"))
                   for name in trainable)
    assert accounting["trainable_parameters"] == sum(
        p.numel() for p in backbone_parameters(model)
    )
    # Unlike LoRA it adapts the blocks themselves, so no adapter is recorded.
    assert accounting["rank"] is None and accounting["replaced_modules"] == []


def test_objective_balance_round_actually_reaches_auxiliary_dominance():
    # The pressure round never left next-token dominance because that term has
    # a fixed weight of one. This round must downweight it directly.
    text = (ROOT / "scripts/launch_gruenau_predictive_state_objective_balance.sh").read_text()
    assert "PREDICTIVE_STATE_NTP_WEIGHT" in text
    assert "ntp_weights=(0.01 0.0 1.0 0.01 0.01 0.01)" in text
    # Full finetuning of the upper stack, and a longer cell.
    assert "full_upper" in text
    assert "4880" in text
    # The gate and the deferral path must both survive.
    assert "refusing busy GPU" in text
    assert "wait_and_run_predictive_state_cells.sh" in text


def test_checkpoint_loader_rebuilds_full_upper_topology(tmp_path):
    # A full_upper cell records a null rank; the loader must not try to install
    # LoRA for it, or every downstream probe fails on those checkpoints.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from torch import nn
    from textjepa.training.predictive_state import _restore_adaptation
    from textjepa.models.action_transition import LoRALinear

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(4))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    full = Model()
    _restore_adaptation(full, {"first_trainable_layer": 3, "rank": None,
                               "alpha": None})
    assert not any(isinstance(m, LoRALinear) for m in full.modules())
    assert all(name.startswith(("model.layers.2", "model.layers.3"))
               for name, p in full.named_parameters() if p.requires_grad)

    adapted = Model()
    _restore_adaptation(adapted, {"first_trainable_layer": 3, "rank": 2,
                                  "alpha": 4})
    assert any(isinstance(m, LoRALinear) for m in adapted.modules())

    frozen = Model()
    _restore_adaptation(frozen, None)
    assert not any(p.requires_grad for p in frozen.parameters())


def test_jsonl_corpus_reader_keeps_one_document_per_record(tmp_path):
    # Web corpora hold blank lines inside documents, so a separator-based
    # splitter would silently shred them. One record per line avoids that, and
    # the reader must stop at the limit rather than load the whole file.
    import importlib.util
    import json as json_module

    spec = importlib.util.spec_from_file_location(
        "prepare_corpus", ROOT / "scripts/prepare_predictive_state_corpus.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json_module.dumps(row) for row in [
        {"text": "first paragraph\n\nsecond paragraph of the same document"},
        {"text": "   "},
        {"text": "another document"},
        {"text": "a third one"},
    ]) + "\n", encoding="utf-8")

    documents = module.read_documents(
        path, input_format="jsonl", text_field="text", limit=None
    )
    # The blank record is dropped; internal blank lines are preserved.
    assert len(documents) == 3
    assert "\n\n" in documents[0]
    assert documents[1] == "another document"

    limited = module.read_documents(
        path, input_format="jsonl", text_field="text", limit=2
    )
    assert len(limited) == 2


def test_fineweb_scale_round_reruns_its_own_control_and_scales_the_batch():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_fineweb_scale.sh").read_text()
    # Changing corpus invalidates every earlier number, so the control must be
    # re-measured here rather than borrowed from the WikiText rounds.
    assert "qwen05-fwe-ntp-only-s0-v1" in text
    assert "variants=(ntp_only full full)" in text
    assert 'run_predictive_state_stage1_screen.sh $(printf %q "$variant")' in text
    # Global batch 65,536 tokens: microbatch 8 x accumulation 8 x 1024.
    assert "PREDICTIVE_STATE_ACCUMULATION=8" in text
    assert "fineweb_edu_qwen_ctx1024.pt" in text
    assert "refusing busy GPU" in text


def test_rollout_trainer_adapts_a_full_upper_backbone():
    # Stage 2 must not silently freeze the backbone when Stage 1 used direct
    # finetuning instead of an adapter.
    text = (ROOT / "scripts/train_action_transition_rollout.py").read_text()
    assert "backbone_parameters" in text
    assert "lora_parameters(model)) or list(backbone_parameters(model))" in text


def test_stage2_curriculum_chains_checkpoints_and_gates_replay():
    text = (ROOT / "scripts/run_predictive_state_stage2_curriculum.sh").read_text()
    # Each horizon must resume from the previous one, or it is five restarts.
    assert 'checkpoint="$stage_dir/last.pt"' in text
    assert "--on-policy-fraction" in text
    # Replay only from horizon 16, matching the trainer's own guard.
    assert 'PREDICTIVE_STATE_REPLAY_FRACTIONS:-0.0 0.0 0.0 0.25 0.25' in text
    # The gate is measured past the longest trained horizon.
    assert "--horizon 128 --horizon 256" in text


def test_stage2_launcher_carries_a_stage1_checkpoint_and_the_gpu_gate():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_stage2.sh").read_text()
    assert "PREDICTIVE_STATE_STAGE1_CHECKPOINT" in text
    assert "missing Stage 1 checkpoint" in text
    assert "refusing busy GPU" in text
    assert "wait_and_run_predictive_state_cells.sh" in text


def test_block_refresh_materializes_only_the_new_block():
    # Re-prefilling the whole prefix on every refresh is O(prefix) and would
    # erase the speedup the operating point exists to provide.
    text = (ROOT / "scripts/evaluate_action_transition_rollout.py").read_text()
    assert "crop_cache(cache, start)" in text
    assert "full_tokens[:, -block:]" in text
    assert "past_key_values=cache" in text
    # Positions must continue from the cache, not restart at zero.
    assert "torch.arange(\n                start, full_tokens.shape[1]" in text


def test_autonomous_sweep_is_not_capped_by_the_benchmark_length():
    # refresh_32 and refresh_64 never fired while the sweep was capped at 32
    # actions, so every interval reported the jump-only number.
    text = (ROOT / "scripts/evaluate_action_transition_rollout.py").read_text()
    assert "--autonomous-actions" in text and "--refresh-interval" in text
    assert "args.autonomous_actions" in text
    assert "modelled_speedup_vs_full" in text


def test_gold_steps_drop_calculator_annotations():
    # Gold GSM8K steps contain <<...>> annotations that model samples never
    # produce; leaving them in lets an energy head separate observed from
    # sampled continuations on formatting alone.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cf_gen", ROOT / "scripts/generate_counterfactual_continuations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps = module.gold_steps(
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "She sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    )
    assert all("<<" not in step and ">>" not in step for step in steps)
    assert steps[0] == "Natalia sold 48/2 = 24 clips in May."
    assert steps[-1] == "#### 72"


def test_speculative_losslessness_uses_a_matched_numerics_reference():
    # A cached and an uncached bf16 forward disagree on near-ties and the flip
    # cascades, so checking the speculative output against the cached reference
    # reports a false loss. Measured cached-vs-uncached agreement was 0.52.
    text = (ROOT / "scripts/evaluate_action_transition_rollout.py").read_text()
    assert "def uncached_greedy_decode(" in text
    assert "matches_verifier_path_greedy" in text
    assert "cached_vs_uncached_reference_agreement" in text


def test_harness_refresh_rematerializes_the_block_not_one_token():
    # Refreshing a single token leaves the jumped block's upper cache entries
    # in place, which is the thing a refresh exists to replace.
    text = (ROOT / "scripts/evaluate_lm_harness_jump.py").read_text()
    assert "start = max(0, index - self._refresh + 1)" in text
    assert "crop_cache(cache, start)" in text
    assert "tokens[:, start:index + 1]" in text
