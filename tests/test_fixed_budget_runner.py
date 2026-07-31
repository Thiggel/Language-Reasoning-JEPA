from pathlib import Path


def test_fixed_budget_runner_sets_short_job_local_multiprocessing_tmpdir():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "SLURM_TMPDIR:-/tmp/tj-fixed-" in script
    assert 'export TMPDIR="$tmp_dir" TMP="$tmp_dir" TEMP="$tmp_dir"' in script


def test_fixed_budget_runner_adds_optional_sentence_score_to_hydra_config():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "+score=decoder" in script
    assert " slack=$slack score=decoder " not in script


def test_fixed_budget_runner_forwards_optional_jepa_overrides():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert 'if [[ -n "${JEPA_OVERRIDES:-}" ]]' in script
    assert '"${jepa_overrides[@]}"' in script
    assert 'bash "$TEXTJEPA_ROOT/scripts/run_intent_recovery_audit_cell.sh"' in script


def test_recovery_runner_passes_extra_arguments_to_both_jepa_trainers():
    script = Path("scripts/run_intent_recovery_audit_cell.sh").read_text()
    assert 'experiment_overrides=("${@:5}")' in script
    assert script.count('"${experiment_overrides[@]}"') == 2
    assert 'max_chunks=96 \\\n      "${experiment_overrides[@]}"' in script
