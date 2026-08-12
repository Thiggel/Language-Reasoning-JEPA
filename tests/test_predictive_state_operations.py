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
