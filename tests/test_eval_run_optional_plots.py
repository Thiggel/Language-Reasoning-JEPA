import os
import subprocess
from pathlib import Path


def test_eval_run_skips_plots_when_matplotlib_is_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).parents[1] / "scripts" / "eval_run.sh"
    evaluator = scripts / "eval_run.sh"
    evaluator.write_text(source.read_text())

    calls = tmp_path / "calls"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == -c ]]; then exit 1; fi\n"
        f"printf '%s\\n' \"$1\" >> {calls}\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)

    env = dict(os.environ, PY=str(fake_python))
    result = subprocess.run(
        ["bash", str(evaluator), str(tmp_path / "model"), "cpu"],
        cwd=root,
        env=env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    invoked = calls.read_text().splitlines()
    assert "scripts/probe.py" in invoked
    assert invoked.count("scripts/plan.py") == 2
    assert "scripts/analyze.py" not in invoked
    assert "skipping optional geometry plots" in result.stderr
