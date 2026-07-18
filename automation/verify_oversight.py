#!/usr/bin/env python3
"""Verification gate run before the controller commits Codex oversight work."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str]) -> int:
    print("+", " ".join(argv), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(argv, cwd=ROOT, env=env).returncode


def main() -> int:
    python = sys.executable
    if run([python, str(ROOT / "automation/validate_reports.py"), str(ROOT / "research/reports")]):
        return 1
    return run([python, "-m", "pytest", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
