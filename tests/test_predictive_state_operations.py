from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gruenau_launcher_enforces_dashboard_and_direct_gpu_admission():
    text = (ROOT / "scripts/launch_gruenau_predictive_state_diagnostics_v1.sh").read_text()
    assert "gruenau-gpus" in text
    assert "memory.used" in text and "utilization.gpu" in text
    assert "refusing busy GPU" in text
    assert "run_summary.json" in text


def test_slurm_wrapper_validates_command_and_records_failure_provenance():
    text = (ROOT / "scripts/slurm_predictive_state.sbatch").read_text()
    assert "[[ $# -gt 0 ]]" in text
    assert "trap" in text
    assert "resolved_config.json" in text
    assert "environment.json" in text
    assert "run_summary.json" in text
