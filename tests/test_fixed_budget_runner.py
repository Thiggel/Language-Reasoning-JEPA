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


def test_fixed_budget_runner_supports_looped_compute_curves():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "looped_token_lm)" in script
    assert "looped_sentence_lm|looped_sentence_latent_lm)" in script
    assert 'EVAL_LOOPS:-"1 2 4 8 16 32"' in script
    assert '+eval_loops="$loops"' in script
    assert "metrics_by_loop_and_slack" in script
    assert "train_loop_histogram" in script
    assert "n_parameters" in script
    assert '--eval-loops "$loops"' in script
    assert script.count("measure_flops=true") == 2


def test_planners_offer_opt_in_measured_compute_accounting():
    for path in ("scripts/plan_lm.py", "scripts/plan_sentlm.py"):
        script = Path(path).read_text()
        assert 'cfg.get("measure_flops", False)' in script
        assert "measured_flops_per_episode" in script
        assert "flop_measurement_supported" in script


def test_looped_runner_can_reuse_a_completed_training_checkpoint():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "LOOPED_CHECKPOINT" in script
    assert 'loop_checkpoint=${LOOPED_CHECKPOINT:-$model_dir/best.pt}' in script
    assert 'if [[ -z "${LOOPED_CHECKPOINT:-}" ]]' in script
    assert '--checkpoint "$loop_checkpoint"' in script


def test_fixed_budget_runner_defers_optional_rendering_from_cluster_jobs():
    script = Path("scripts/run_intent_fixed_budget_cell.sh").read_text()
    assert "--coordinates-out" in script
    assert "--figure-out" not in script


def test_length_ood_runner_uses_exact_controlled_cells():
    script = Path("scripts/run_intent_length_ood_cell.sh").read_text()
    assert 'EVAL_LENGTHS:-"3 5 7 9 10 11"' in script
    assert 'EVAL_SLACKS:-"0 1 2 4"' in script
    assert "eval_n_vars_range=[12,12]" in script
    assert "eval_strict_steps_range=true" in script
    assert "eval_sample_max_tries=100000" in script
    assert 'split=test' in script
    assert '"candidate_interface": "current symbolic feasible-action menu"' in script
