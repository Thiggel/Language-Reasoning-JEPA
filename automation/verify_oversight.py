#!/usr/bin/env python3
"""Verification gate run before the controller commits Codex oversight work."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str]) -> int:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, cwd=ROOT).returncode


def main() -> int:
    python = str(ROOT / ".venv2/bin/python")
    if run([python, str(ROOT / "automation/validate_reports.py"), str(ROOT / "research/reports")]):
        return 1
    return run([python, "-m", "pytest", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())

