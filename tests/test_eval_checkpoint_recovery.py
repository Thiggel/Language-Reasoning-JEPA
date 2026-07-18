import json
import os
import subprocess
from pathlib import Path


def test_checkpoint_recovery_packages_evaluation_metrics(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    evaluator = scripts / "eval_run.sh"
    evaluator.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "model_dir=$1\n"
        "printf '%s\\n' '{\"latent_planner\":{\"success\":0.7}}' "
        '> "$model_dir/plan_slack0_look1.json"\n'
        "printf '%s\\n' '{\"latent_planner\":{\"success\":0.9}}' "
        '> "$model_dir/plan_slack2_look1.json"\n'
    )
    evaluator.chmod(0o755)

    source_model = tmp_path / "source-model"
    source_model.mkdir()
    (source_model / "best.pt").write_bytes(b"checkpoint")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = Path(__file__).parents[1] / "scripts" / (
        "eval_intent_checkpoint_recovery.sh"
    )
    env = dict(os.environ)
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "TEXTJEPA_ROOT": str(root),
        }
    )

    subprocess.run(
        ["bash", str(script), "python", str(source_model), "lr1e3-full10"],
        check=True,
        env=env,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics == {
        "label": "lr1e3-full10",
        "source_model": str(source_model),
        "checkpoint_status": "best checkpoint from walltime-truncated training",
        "strict": {"latent_planner": {"success": 0.7}},
        "slack2": {"latent_planner": {"success": 0.9}},
    }
    assert (run_dir / "model" / "best.pt").read_bytes() == b"checkpoint"
