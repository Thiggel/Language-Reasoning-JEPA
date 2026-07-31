from pathlib import Path


def test_fixed_budget_runner_sets_short_job_local_multiprocessing_tmpdir():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "SLURM_TMPDIR:-/tmp/tj-fixed-" in script
    assert 'export TMPDIR="$tmp_dir" TMP="$tmp_dir" TEMP="$tmp_dir"' in script
