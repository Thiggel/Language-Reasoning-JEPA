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
    assert "--backbone-mode lora" in text


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
