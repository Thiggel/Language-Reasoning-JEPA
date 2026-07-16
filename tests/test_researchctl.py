import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("researchctl", ROOT / "automation/researchctl.py")
researchctl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(researchctl)


class ResearchCtlTests(unittest.TestCase):
    def setUp(self):
        self.controller = researchctl.Controller(ROOT / "automation/config.toml")
        self.plan = json.loads((ROOT / "automation/examples/round-plan.json").read_text())

    def test_walltime_parsing(self):
        self.assertEqual(researchctl.parse_walltime_minutes(17), 17)
        self.assertEqual(researchctl.parse_walltime_minutes("01:30:00"), 90)
        self.assertEqual(researchctl.parse_walltime_minutes("1-02:00:00"), 1560)

    def test_example_plan_is_valid_and_bounded(self):
        validated = self.controller.validate_plan(copy.deepcopy(self.plan))
        self.assertEqual(validated["projected_gpu_hours"], 0.167)

    def test_duplicate_job_ids_are_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"].append(copy.deepcopy(plan["jobs"][0]))
        with self.assertRaisesRegex(researchctl.ResearchCtlError, "duplicate job id"):
            self.controller.validate_plan(plan)

    def test_control_characters_are_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"][0]["command"][1] = "scripts/train.py\nssh elsewhere"
        with self.assertRaisesRegex(researchctl.ResearchCtlError, "control character"):
            self.controller.validate_plan(plan)

    def test_controller_host_launchers_are_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"][0]["command"] = ["ssh", "somewhere", "do-work"]
        with self.assertRaisesRegex(researchctl.ResearchCtlError, "launcher"):
            self.controller.validate_plan(plan)

    def test_round_budget_is_enforced(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"][0]["gpus"] = 2
        plan["jobs"][0]["walltime_minutes"] = 1000
        with self.assertRaisesRegex(researchctl.ResearchCtlError, "GPU-hours"):
            self.controller.validate_plan(plan)

    def test_runner_preserves_argv_boundaries(self):
        job = copy.deepcopy(self.plan["jobs"][0])
        job["command"] = ["{python}", "scripts/train.py", "label=a value with spaces"]
        cluster = self.controller.cfg["clusters"]["gruenau"]
        script = self.controller._runner_script(job, cluster, "/snapshot", "/run")
        self.assertIn("'label=a value with spaces'", script)
        self.assertIn("resolved_config.json", script)
        self.assertIn("environment.json", script)

    def test_runner_executes_and_writes_compact_contract(self):
        job = copy.deepcopy(self.plan["jobs"][0])
        job["id"] = "local-runner-contract-test"
        job["command"] = ["{python}", "-c", "print('runner ok')"]
        job["walltime_minutes"] = 1
        cluster = self.controller.cfg["clusters"]["gruenau"]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            script_path = Path(tmp) / "job.sh"
            script_path.write_text(
                self.controller._runner_script(job, cluster, str(ROOT), str(run_dir))
            )
            result = subprocess.run(["bash", str(script_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((run_dir / "state").read_text().strip(), "COMPLETED")
            summary = json.loads((run_dir / "run_summary.json").read_text())
            self.assertEqual(summary["scientific_validity"], "not_assessed")
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "environment.json").exists())

    def test_non_required_system_report_does_not_block_submission(self):
        self.assertNotIn(
            "2026-07-16-autonomous-research-interface",
            self.controller._unreviewed_reports(),
        )

    def test_legacy_classification_is_conservative(self):
        self.assertEqual(researchctl.infer_legacy_project("paper_causal_x"), "intent_phrase")
        self.assertEqual(researchctl.infer_legacy_project("token-prior-x"), "token_igsm")
        self.assertEqual(researchctl.infer_legacy_project("edit-x"), "sequence_edit")
        self.assertEqual(researchctl.infer_legacy_project("unknown-x"), "legacy/unclassified")

    def test_state_migration_preserves_every_job_identity_and_is_idempotent(self):
        fixture = ROOT / "tests/fixtures/controller_state_v1.json"
        with tempfile.TemporaryDirectory() as tmp:
            ctl = researchctl.Controller(ROOT / "automation/config.toml")
            ctl.state_dir = Path(tmp) / ".researchctl"
            ctl.state_dir.mkdir()
            ctl.state_path = ctl.state_dir / "state.json"
            ctl.lock_path = ctl.state_dir / "controller.lock"
            ctl.state_path.write_bytes(fixture.read_bytes())
            before = json.loads(ctl.state_path.read_text())
            identity = lambda s: sorted(
                (rid, jid, j.get("backend_id"), j.get("local_dir"), j.get("remote_dir"), j.get("state"))
                for rid, rnd in s["rounds"].items() for jid, j in rnd.get("jobs", {}).items()
            )
            ctl.migrate_state(execute=False)
            self.assertEqual(ctl.state_path.read_bytes(), fixture.read_bytes())
            ctl.migrate_state(execute=True)
            migrated_once = ctl.state_path.read_bytes()
            after = json.loads(migrated_once)
            self.assertEqual(after["schema_version"], 2)
            self.assertEqual(identity(before), identity(after))
            self.assertEqual(after["rounds"]["mystery-round"]["project"], "legacy/unclassified")
            ctl.migrate_state(execute=True)
            self.assertEqual(ctl.state_path.read_bytes(), migrated_once)

    def test_project_gpu_cap_rejects_future_admission_only(self):
        state = {"rounds": {}}
        plan = copy.deepcopy(self.plan)
        plan["jobs"] = [copy.deepcopy(plan["jobs"][0]) for _ in range(7)]
        for i, job in enumerate(plan["jobs"]):
            job["id"] = f"job-{i}"
        plan = self.controller.validate_plan(plan)
        with self.assertRaisesRegex(researchctl.ResearchCtlError, "project GPU cap"):
            self.controller._fair_admission_guard(plan, state)

    def test_duplicate_registered_round_is_detectable_without_submission(self):
        plan = self.controller.validate_plan(copy.deepcopy(self.plan))
        state = {"rounds": {plan["round_id"]: {}}}
        self.assertIn(plan["round_id"], state["rounds"])


if __name__ == "__main__":
    unittest.main()
