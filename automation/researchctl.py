#!/usr/bin/env python3
"""Durable, bounded experiment controller for TextJEPA.

Codex writes a declarative plan.  This program owns inventory, validation,
immutable code snapshots, submission, polling, result retrieval, storage
guards, and fresh Codex wake-ups.  It intentionally uses only the Python
standard library so it can run from systemd/cron without activating a venv.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any, Iterable


TERMINAL_SLURM = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}
SUCCESS_SLURM = {"COMPLETED"}
ACTIVE_STATES = {"SUBMITTED", "PENDING", "RUNNING", "COMPLETING"}
SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_SLURM = re.compile(r"^[a-zA-Z0-9_.:+,@%/-]+$")
PROJECTS = ("intent_phrase", "token_igsm", "sequence_edit")


class ResearchCtlError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run(
    argv: list[str], *, cwd: Path | None = None, input_text: str | None = None,
    timeout: int = 60, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv, cwd=cwd, input=input_text, text=True, capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ResearchCtlError(f"command failed ({result.returncode}): {shlex.join(argv)}\n{detail}")
    return result


def parse_walltime_minutes(value: Any) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ResearchCtlError("walltime_minutes must be an integer or Slurm time string")
    days = 0
    clock = value
    if "-" in value:
        day, clock = value.split("-", 1)
        days = int(day)
    parts = [int(x) for x in clock.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes = parts
        seconds = 0
    else:
        raise ResearchCtlError(f"invalid walltime: {value}")
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def slurm_time(minutes: int) -> str:
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{mins:02d}:00"


def infer_legacy_project(round_id: str, jobs: Iterable[str] = ()) -> str:
    """Conservatively classify legacy identifiers without changing their jobs."""
    text = " ".join((round_id, *jobs)).lower()
    if "smoke" in text:
        return "shared"
    if "edit" in text:
        return "sequence_edit"
    if any(token in text for token in (
        "hard", "token-hierarchy", "semantic-boundary", "text-hier",
        "token-prior", "oracle-cem", "hierarchy-gradient", "fixed-hierarchy",
    )):
        return "token_igsm"
    if any(token in text for token in (
        "paper_causal", "paper-causal", "disc", "real", "intent-policy",
        "observed-action", "gar-geometry",
    )):
        return "intent_phrase"
    return "legacy/unclassified"


def protected_path_violations(paths: Iterable[str], protected: Iterable[str]) -> list[str]:
    roots = tuple(item.rstrip("/") for item in protected)
    return [path for path in paths if any(path == item or path.startswith(item + "/") for item in roots)]


def sibling_memory_violations(paths: Iterable[str], project: str) -> list[str]:
    cycle_names = {"intent_phrase": "intent_phrase", "token_igsm": "hard_text", "sequence_edit": "sequence_edit"}
    roots: list[str] = []
    for sibling in PROJECTS:
        if sibling == project:
            continue
        roots.extend((
            f"projects/{sibling}",
            f"research/reports/{sibling}",
            f"research/cycles/{cycle_names[sibling]}",
        ))
    if project != "intent_phrase":
        roots.append("research/intent_phrase")
    if project != "token_igsm":
        roots.append("research/hard_text")
    if project != "sequence_edit":
        roots.extend(("research/sequence_edit", "research/archive/edit_track"))
    return protected_path_violations(paths, roots)


class Controller:
    def __init__(self, config_path: Path | None = None) -> None:
        guessed_root = Path(__file__).resolve().parents[1]
        config_path = config_path or Path(
            os.environ.get("RESEARCH_CONFIG", guessed_root / "automation/config.toml")
        )
        if not config_path.exists():
            raise ResearchCtlError(
                f"configuration not found: {config_path}; copy automation/config.example.toml"
            )
        self.config_path = config_path.resolve()
        self.cfg = tomllib.loads(self.config_path.read_text())
        self.root = Path(self.cfg["project"]["root"]).expanduser().resolve()
        self.state_dir = self.root / self.cfg["controller"].get("state_dir", ".researchctl")
        self.run_root = self.root / self.cfg["controller"].get(
            "run_root", "runs/autonomy"
        )
        self.plan_path = self.root / self.cfg["controller"].get(
            "plan_path", "research/NEXT_PLAN.json"
        )
        self.state_path = self.state_dir / "state.json"
        self.lock_path = self.state_dir / "controller.lock"
        self.oversight_lock_path = self.state_dir / "oversight.lock"
        self.stop_path = self.root / self.cfg["controller"].get(
            "stop_file", "research/STOP"
        )
        self.python = sys.executable
        registry = self.root / self.cfg["controller"].get("project_registry", "projects")
        self.projects: dict[str, dict[str, Any]] = {}
        for slug in PROJECTS:
            manifest = registry / slug / "controller.toml"
            if manifest.exists():
                value = tomllib.loads(manifest.read_text())
                if value.get("project", {}).get("slug") != slug:
                    raise ResearchCtlError(f"project manifest slug mismatch: {manifest}")
                self.projects[slug] = value

    @contextlib.contextmanager
    def lock(self) -> Iterable[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResearchCtlError("another researchctl process holds the lock") from exc
            handle.write(f"pid={os.getpid()} time={now()}\n")
            handle.flush()
            yield

    @contextlib.contextmanager
    def oversight_lock(self) -> Iterable[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.oversight_lock_path.open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResearchCtlError("another project oversight process is already running") from exc
            handle.write(f"pid={os.getpid()} time={now()}\n")
            handle.flush()
            yield

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "created_at": now(), "rounds": {}, "paused": False}
        return json.loads(self.state_path.read_text())

    def project_manifest(self, project: str) -> dict[str, Any]:
        if project not in self.projects:
            raise ResearchCtlError(f"unknown project: {project}")
        return self.projects[project]

    def project_plan_path(self, project: str) -> Path:
        return self.root / self.project_manifest(project)["project"]["plan_path"]

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        json_dump(self.state_path, state)

    def init(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self.save_state(self.load_state())
        print(f"initialized {self.state_dir}")

    def _ssh(self, cluster: dict[str, Any], command: str, *, timeout: int = 30,
             input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        return run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", cluster["ssh"], command],
            input_text=input_text, timeout=timeout, check=check,
        )

    def doctor(self) -> None:
        checks: list[tuple[str, bool, str]] = []
        checks.append(("project root", self.root.is_dir(), str(self.root)))
        checks.append(("git repository", (self.root / ".git").exists(), str(self.root / ".git")))
        checks.append(("Python >=3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
        for binary in ("git", "ssh", "rsync", "timeout"):
            path = shutil.which(binary)
            checks.append((binary, bool(path), path or "missing"))
        codex = self.cfg["codex"].get("executable", "codex")
        checks.append(("Codex executable", bool(shutil.which(codex) or Path(codex).exists()), codex))
        ignored = all(
            run(["git", "check-ignore", "-q", str(path)], cwd=self.root, check=False).returncode == 0
            for path in (self.state_dir, self.run_root)
        )
        checks.append(("transient paths ignored", ignored, ".researchctl and runs/autonomy"))
        dirty = run(["git", "status", "--porcelain"], cwd=self.root).stdout.strip()
        checks.append(("integration worktree", True, "dirty user changes preserved" if dirty else "clean"))
        for slug, manifest in self.projects.items():
            worktree = Path(manifest["project"]["worktree"])
            project_dirty = run(["git", "status", "--porcelain"], cwd=worktree, check=False).stdout.strip() if worktree.exists() else "missing"
            checks.append((f"clean project worktree {slug}", worktree.is_dir() and not project_dirty, str(worktree)))
        for name, cluster in self.cfg.get("clusters", {}).items():
            if not cluster.get("enabled", False):
                continue
            probe = self._ssh(cluster, "true", check=False)
            checks.append((f"SSH {name}", probe.returncode == 0, cluster["ssh"]))
            if cluster["kind"] == "slurm" and probe.returncode == 0:
                slurm = self._ssh(cluster, "command -v sbatch && command -v squeue", check=False)
                checks.append((f"Slurm {name}", slurm.returncode == 0, slurm.stdout.strip() or slurm.stderr.strip()))
            if probe.returncode == 0:
                remote_path = cluster["project_root"]
                remote_python = cluster["python"]
                writable = self._ssh(
                    cluster,
                    f"p={shlex.quote(remote_path)}; while [ ! -e \"$p\" ] && [ \"$p\" != / ]; do p=$(dirname \"$p\"); done; "
                    f"test -w \"$p\" && test -x {shlex.quote(remote_python)}",
                    check=False,
                )
                checks.append((f"storage/env {name}", writable.returncode == 0, f"{remote_path}; {remote_python}"))
        width = max(len(x[0]) for x in checks)
        for label, ok, detail in checks:
            print(f"{'OK' if ok else 'FAIL':4}  {label:<{width}}  {detail}")
        if not all(ok for _label, ok, _detail in checks):
            raise ResearchCtlError("doctor found blocking failures")

    def _gruenau_inventory(self, cluster: dict[str, Any]) -> dict[str, Any]:
        gpus: list[dict[str, Any]] = []
        raw_lines = []
        def probe_node(node: int) -> tuple[int, subprocess.CompletedProcess[str]]:
            ssh_host = cluster["ssh_template"].format(node=node)
            command = (
                "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu "
                "--format=csv,noheader,nounits"
            )
            result = run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, command],
                timeout=12, check=False,
            )
            return node, result

        nodes = cluster.get("nodes", [])
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(nodes))) as pool:
            results = list(pool.map(probe_node, nodes))
        for node, result in sorted(results):
            raw_lines.append(f"[{node}] rc={result.returncode}\n{result.stdout}{result.stderr}")
            if result.returncode:
                continue
            for line in result.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) != 5:
                    continue
                index, name, total, used, util = fields
                free = int(used) < int(cluster.get("free_memory_used_mb", 100)) and int(util) < int(cluster.get("free_utilization_percent", 5))
                gpus.append({
                    "node": int(node), "index": int(index), "name": name,
                    "memory_total_mb": int(total), "memory_used_mb": int(used),
                    "utilization_percent": int(util), "free": free,
                })
        return {"gpus": gpus, "raw": "\n".join(raw_lines)}

    def _slurm_inventory(self, cluster: dict[str, Any]) -> dict[str, Any]:
        command = (
            "printf '%s\\n' '__SINFO__'; "
            "sinfo -h -o '%P|%a|%l|%D|%t|%G'; "
            "printf '%s\\n' '__MY_QUEUE__'; "
            "squeue -u \"$USER\" -h -o '%i|%T|%P|%b|%M|%l'; "
            "printf '%s\\n' '__GPU_QUEUE__'; "
            "squeue -h -t RUNNING,COMPLETING,PENDING -o '%i|%u|%T|%P|%b' | head -500"
        )
        result = self._ssh(cluster, command, timeout=45, check=False)
        return {"ok": result.returncode == 0, "raw": result.stdout, "error": result.stderr.strip()}

    def _storage_probe(self, cluster: dict[str, Any]) -> dict[str, Any]:
        path = cluster["project_root"]
        command = (
            f"p={shlex.quote(path)}; while [ ! -e \"$p\" ] && [ \"$p\" != / ]; do p=$(dirname \"$p\"); done; "
            "df -Pk \"$p\" | tail -1; "
            "timeout 5 quota -s 2>/dev/null || true"
        )
        result = self._ssh(cluster, command, timeout=20, check=False)
        return {"ok": result.returncode == 0, "raw": result.stdout, "error": result.stderr.strip()}

    def inventory(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"schema_version": 1, "created_at": now(), "clusters": {}}
        for name, cluster in self.cfg.get("clusters", {}).items():
            if not cluster.get("enabled", False):
                snapshot["clusters"][name] = {"enabled": False}
                continue
            print(f"probing {name}...", file=sys.stderr)
            if cluster["kind"] == "gruenau":
                data = self._gruenau_inventory(cluster)
                data["storage"] = self._storage_probe(cluster)
            else:
                data = self._slurm_inventory(cluster)
                data["storage"] = self._storage_probe(cluster)
            data["kind"] = cluster["kind"]
            snapshot["clusters"][name] = data
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.state_dir / "inventory" / f"{stamp}.json"
        json_dump(path, snapshot)
        latest = self.state_dir / "inventory" / "latest.json"
        json_dump(latest, snapshot)
        print(path)
        return snapshot

    def storage(self) -> None:
        paths = [self.root, self.run_root, self.state_dir, Path.home()]
        seen: set[tuple[int, int]] = set()
        print("LOCAL FILESYSTEMS")
        for path in paths:
            path = path if path.exists() else path.parent
            stat = os.stat(path)
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            usage = shutil.disk_usage(path)
            used_pct = 100 * usage.used / usage.total
            print(f"{path}: free={usage.free / 2**30:.1f} GiB used={used_pct:.1f}%")
        print("\nREMOTE FILESYSTEMS")
        for name, cluster in self.cfg.get("clusters", {}).items():
            if cluster.get("enabled", False):
                probe = self._storage_probe(cluster)
                print(f"[{name}]\n{probe['raw'].strip() or probe['error']}")

    def _local_storage_guard(self) -> None:
        storage = self.cfg["storage"]
        usage = shutil.disk_usage(self.run_root if self.run_root.exists() else self.root)
        free_gb = usage.free / 2**30
        used_pct = 100 * usage.used / usage.total
        if free_gb < float(storage["min_free_gb"]) or used_pct > float(storage["max_used_percent"]):
            raise ResearchCtlError(
                f"storage guard: {free_gb:.1f} GiB free, {used_pct:.1f}% used; refusing new work"
            )

    def _unreviewed_reports(self) -> list[str]:
        report_root = self.root / "research/reports"
        receipts = self.state_dir / "read_receipts"
        unreviewed: list[str] = []
        if not report_root.exists():
            return unreviewed
        for path in report_root.rglob("report.json"):
            with contextlib.suppress(OSError, json.JSONDecodeError):
                metadata = json.loads(path.read_text())
                if metadata.get("review_required", True):
                    report_id = str(metadata["id"])
                    receipt_path = receipts / f"{report_id}.json"
                    expected_hash = hashlib.sha256(
                        (path.parent / metadata.get("report", "REPORT.md")).read_bytes()
                    ).hexdigest()[:16]
                    receipt_hash = None
                    with contextlib.suppress(OSError, json.JSONDecodeError):
                        receipt_hash = json.loads(receipt_path.read_text()).get("hash")
                    if receipt_hash != expected_hash:
                        unreviewed.append(report_id)
        return sorted(unreviewed)

    def _human_review_guard(self) -> None:
        unreviewed = self._unreviewed_reports()
        limit = int(self.cfg["limits"].get("max_unreviewed_reports", 1))
        if len(unreviewed) > limit:
            raise ResearchCtlError(
                f"human review guard: {len(unreviewed)} reports await reading (limit {limit}): "
                + ", ".join(unreviewed)
            )

    def _remote_storage_guard(self, name: str, cluster: dict[str, Any]) -> None:
        probe = self._storage_probe(cluster)
        if not probe["ok"]:
            raise ResearchCtlError(f"cannot verify storage on {name}: {probe['error']}")
        first = probe["raw"].splitlines()[0].split()
        if len(first) < 5:
            raise ResearchCtlError(f"cannot parse storage probe on {name}: {probe['raw']}")
        available_gb = int(first[3]) / 1024 / 1024
        used_pct = float(first[4].rstrip("%"))
        storage = self.cfg["storage"]
        if available_gb < float(storage["remote_min_free_gb"]) or used_pct > float(storage["remote_max_used_percent"]):
            raise ResearchCtlError(
                f"remote storage guard {name}: {available_gb:.1f} GiB free, {used_pct:.1f}% used"
            )

    def load_plan(self, path: Path | None = None, project: str | None = None) -> dict[str, Any]:
        path = path or (self.project_plan_path(project) if project else self.plan_path)
        if not path.exists():
            raise ResearchCtlError(f"plan not found: {path}")
        return json.loads(path.read_text())

    def validate_plan(self, plan: dict[str, Any], *, resolve_auto: bool = False) -> dict[str, Any]:
        errors: list[str] = []
        required = {"schema_version", "project", "round_id", "decision", "git_commit", "jobs"}
        errors.extend(f"missing top-level key: {key}" for key in required - plan.keys())
        errors.extend(f"unknown top-level key: {key}" for key in plan.keys() - required - {"projected_gpu_hours"})
        if errors:
            raise ResearchCtlError("invalid plan:\n- " + "\n- ".join(errors))
        if plan["schema_version"] != 2:
            errors.append("schema_version must be 2 for new plans")
        project = str(plan.get("project", ""))
        if project not in self.projects:
            errors.append(f"project must name a registered project: {project}")
        if not SAFE_ID.fullmatch(str(plan["round_id"])):
            errors.append("round_id must be a safe, intuitive identifier")
        commit = str(plan["git_commit"])
        if commit == "AUTO" and resolve_auto:
            commit = run(["git", "rev-parse", "HEAD"], cwd=self.root).stdout.strip()
            plan = dict(plan)
            plan["git_commit"] = commit
        if commit != "AUTO" and not COMMIT_RE.fullmatch(commit):
            errors.append("git_commit must be AUTO or a full 40-character SHA")
        jobs = plan["jobs"]
        if not isinstance(jobs, list) or not jobs:
            errors.append("jobs must be a non-empty list")
            jobs = []
        limits = self.cfg["limits"]
        if len(jobs) > int(limits["max_jobs_per_round"]):
            errors.append("job count exceeds max_jobs_per_round")
        seen: set[str] = set()
        gpu_hours = 0.0
        allowed_job_keys = {
            "id", "cluster", "command", "gpus", "walltime_minutes", "purpose",
            "expected_artifacts", "env", "cpus", "partition", "qos", "account",
            "gpu_type", "min_gpu_memory_mb", "node",
        }
        for index, job in enumerate(jobs):
            prefix = f"jobs[{index}]"
            if not isinstance(job, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("id", "cluster", "command", "gpus", "walltime_minutes", "purpose", "expected_artifacts"):
                if key not in job:
                    errors.append(f"{prefix} missing {key}")
            errors.extend(f"{prefix} has unknown key: {key}" for key in job.keys() - allowed_job_keys)
            job_id = str(job.get("id", ""))
            if not SAFE_ID.fullmatch(job_id):
                errors.append(f"{prefix}.id is unsafe")
            if job_id in seen:
                errors.append(f"duplicate job id: {job_id}")
            seen.add(job_id)
            cluster_name = job.get("cluster")
            cluster = self.cfg.get("clusters", {}).get(cluster_name)
            if not cluster or not cluster.get("enabled", False):
                errors.append(f"{prefix}.cluster is unknown or disabled: {cluster_name}")
                continue
            command = job.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                errors.append(f"{prefix}.command must be a non-empty argv string list")
            elif any("\x00" in x or "\n" in x for x in command):
                errors.append(f"{prefix}.command contains a forbidden control character")
            elif command[0] == "{python}":
                if len(command) < 2 or not (
                    (command[1].startswith("scripts/") and ".." not in Path(command[1]).parts)
                    or (command[1] == "-m" and len(command) >= 3 and command[2].startswith("textjepa"))
                ):
                    errors.append(f"{prefix}.command Python entry point must be scripts/... or -m textjepa...")
            elif command[0] in {"bash", "/usr/bin/bash"}:
                if len(command) < 2 or not command[1].startswith("scripts/") or ".." in Path(command[1]).parts:
                    errors.append(f"{prefix}.command Bash entry point must be a script under scripts/")
            else:
                errors.append(f"{prefix}.command launcher must be {{python}} or bash")
            gpus = job.get("gpus")
            if not isinstance(gpus, int) or gpus < 1 or gpus > int(limits["max_gpus_per_job"]):
                errors.append(f"{prefix}.gpus is outside policy")
                gpus = 0
            try:
                minutes = parse_walltime_minutes(job.get("walltime_minutes"))
                if minutes < 1 or minutes > int(cluster["max_walltime_minutes"]):
                    errors.append(f"{prefix}.walltime exceeds cluster policy")
            except (ValueError, ResearchCtlError) as exc:
                errors.append(f"{prefix}.walltime: {exc}")
                minutes = 0
            gpu_hours += gpus * minutes / 60
            for key in ("partition", "qos", "account", "gpu_type"):
                if key in job and job[key] and not SAFE_SLURM.fullmatch(str(job[key])):
                    errors.append(f"{prefix}.{key} contains unsafe characters")
            env = job.get("env", {})
            if not isinstance(env, dict) or any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(k)) for k in env):
                errors.append(f"{prefix}.env has invalid variable names")
            elif any("\x00" in str(v) or "\n" in str(v) for v in env.values()):
                errors.append(f"{prefix}.env contains forbidden control characters")
            artifacts = job.get("expected_artifacts", [])
            if not isinstance(artifacts, list) or not all(isinstance(x, str) and x for x in artifacts):
                errors.append(f"{prefix}.expected_artifacts must be a string list")
        project_round_limit = float(self.projects.get(project, {}).get("budget", {}).get(
            "maximum_gpu_hours_per_round", limits["max_gpu_hours_per_round"]
        ))
        round_limit = min(float(limits["max_gpu_hours_per_round"]), project_round_limit)
        if gpu_hours > round_limit:
            errors.append(
                f"projected {gpu_hours:.2f} GPU-hours exceeds round limit {round_limit}"
            )
        if errors:
            raise ResearchCtlError("invalid plan:\n- " + "\n- ".join(errors))
        plan["projected_gpu_hours"] = round(gpu_hours, 3)
        return plan

    def finalize_plan(self, project: str | None = None) -> Path:
        plan = self.validate_plan(self.load_plan(project=project), resolve_auto=True)
        out = self.state_dir / "plans" / f"{plan['round_id']}.resolved.json"
        if out.exists() and json.loads(out.read_text()) != plan:
            raise ResearchCtlError(f"resolved plan already exists with different content: {out}")
        json_dump(out, plan)
        print(out)
        return out

    def _snapshot_path(self, cluster: dict[str, Any], commit: str) -> str:
        return f"{cluster['project_root'].rstrip('/')}/_snapshots/{commit}"

    def _ensure_local_snapshot(self, commit: str) -> Path:
        destination = self.run_root / "_code" / commit
        marker = destination / ".textjepa-commit"
        if marker.exists() and marker.read_text().strip() == commit:
            return destination
        if destination.exists():
            raise ResearchCtlError(f"incomplete snapshot exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f"{commit[:12]}-", dir=destination.parent))
        try:
            p1 = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=self.root, stdout=subprocess.PIPE)
            p2 = subprocess.run(["tar", "-xf", "-", "-C", str(temp)], stdin=p1.stdout, capture_output=True)
            if p1.stdout:
                p1.stdout.close()
            rc = p1.wait()
            if rc or p2.returncode:
                raise ResearchCtlError(p2.stderr.decode(errors="replace") or "git archive failed")
            marker_tmp = temp / ".textjepa-commit"
            marker_tmp.write_text(commit + "\n")
            temp.replace(destination)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return destination

    def _ensure_remote_snapshot(self, name: str, cluster: dict[str, Any], commit: str) -> str:
        destination = self._snapshot_path(cluster, commit)
        check = self._ssh(cluster, f"test -f {shlex.quote(destination + '/.textjepa-commit')}", check=False)
        if check.returncode == 0:
            return destination
        parent = str(Path(destination).parent)
        command = (
            f"set -e; mkdir -p {shlex.quote(parent)}; tmp={shlex.quote(destination + '.tmp.$$')}; "
            "rm -rf \"$tmp\"; mkdir -p \"$tmp\"; tar -xf - -C \"$tmp\"; "
            f"printf '%s\\n' {shlex.quote(commit)} > \"$tmp/.textjepa-commit\"; "
            f"test ! -e {shlex.quote(destination)}; mv \"$tmp\" {shlex.quote(destination)}"
        )
        p1 = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=self.root, stdout=subprocess.PIPE)
        p2 = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", cluster["ssh"], command],
            stdin=p1.stdout, capture_output=True, timeout=300,
        )
        if p1.stdout:
            p1.stdout.close()
        rc = p1.wait()
        if rc or p2.returncode:
            raise ResearchCtlError(
                f"snapshot sync to {name} failed: {p2.stderr.decode(errors='replace').strip()}"
            )
        return destination

    def _expand_command(self, job: dict[str, Any], cluster: dict[str, Any], snapshot: str) -> list[str]:
        replacements = {"{root}": snapshot, "{python}": cluster["python"]}
        return [replacements.get(arg, arg.replace("{root}", snapshot)) for arg in job["command"]]

    def _runner_script(self, job: dict[str, Any], cluster: dict[str, Any], snapshot: str,
                       run_dir: str) -> str:
        command = self._expand_command(job, cluster, snapshot)
        env = {
            "TEXTJEPA_ROOT": snapshot,
            "PYTHONPATH": f"{snapshot}/src",
            "XDG_CACHE_HOME": cluster["cache_root"],
            **{str(k): str(v) for k, v in job.get("env", {}).items()},
        }
        setup = cluster.get("setup", [])
        execution_record = base64.b64encode(json.dumps({
            "job": job, "expanded_command": command, "snapshot": snapshot,
        }, sort_keys=True).encode()).decode()
        init_py = (
            "import base64,json,os,pathlib,platform,socket,sys; p=pathlib.Path(os.environ['RUN_DIR']); "
            "d=json.loads(base64.b64decode(os.environ['EXECUTION_RECORD'])); "
            "(p/'resolved_config.json').write_text(json.dumps(d,indent=2)+'\\n'); "
            "e={'hostname':socket.gethostname(),'platform':platform.platform(),'python':sys.version,"
            "'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}; "
            "(p/'environment.json').write_text(json.dumps(e,indent=2)+'\\n')"
        )
        summary_py = (
            "import json,os,pathlib,time; p=pathlib.Path(os.environ['RUN_DIR']); "
            "d={'schema_version':1,'run_id':os.environ['RUN_ID'],'process_status':os.environ['PROCESS_STATUS'],"
            "'scientific_validity':'not_assessed','exclusion_reason':None,'exit_code':int(os.environ['EXIT_CODE']),"
            "'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'metrics':{},'artifacts':[]}; "
            "(p/'run_summary.json').write_text(json.dumps(d,indent=2)+'\\n')"
        )
        lines = [
            "#!/usr/bin/env bash", "set -uo pipefail", f"RUN_DIR={shlex.quote(run_dir)}",
            f"RUN_ID={shlex.quote(job['id'])}", "mkdir -p \"$RUN_DIR\"", "cd " + shlex.quote(snapshot),
            "printf '%s\\n' RUNNING > \"$RUN_DIR/state\"",
            "printf '%s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \"$RUN_DIR/started_at\"",
        ]
        lines.extend(setup)
        lines.extend(f"export {key}={shlex.quote(value)}" for key, value in env.items())
        lines.append(f"export EXECUTION_RECORD={shlex.quote(execution_record)}")
        lines.append("export RUN_DIR RUN_ID")
        lines.append("export TMPDIR=\"${SLURM_TMPDIR:-$RUN_DIR/tmp-${SLURM_JOB_ID:-direct}}\"")
        lines.append("mkdir -p \"$TMPDIR\"")
        lines.append(f"{shlex.quote(cluster['python'])} -c {shlex.quote(init_py)}")
        lines.append("set +e")
        lines.append(f"timeout --signal=TERM --kill-after=120 {int(parse_walltime_minutes(job['walltime_minutes']) * 60)} " + shlex.join(command) + " >\"$RUN_DIR/stdout.log\" 2>\"$RUN_DIR/stderr.log\"")
        lines.extend([
            "rc=$?", "printf '%s\\n' \"$rc\" > \"$RUN_DIR/exit_code\"",
            "if [ \"$rc\" -eq 0 ]; then status=COMPLETED; elif [ \"$rc\" -eq 124 ]; then status=TIMEOUT; else status=FAILED; fi",
            "printf '%s\\n' \"$status\" > \"$RUN_DIR/state\"",
            "if [ ! -s \"$RUN_DIR/run_summary.json\" ]; then",
            "  export RUN_DIR RUN_ID EXIT_CODE=$rc PROCESS_STATUS=$status",
            f"  {shlex.quote(cluster['python'])} -c {shlex.quote(summary_py)} || true",
            "fi", "exit \"$rc\"",
        ])
        return "\n".join(lines) + "\n"

    def _choose_gruenau(self, job: dict[str, Any], state: dict[str, Any]) -> tuple[int, list[int]]:
        latest = self.state_dir / "inventory/latest.json"
        inventory = json.loads(latest.read_text()) if latest.exists() else self.inventory()
        gpus = inventory["clusters"][job["cluster"]].get("gpus", [])
        reserved = {
            (j.get("node"), gpu)
            for rnd in state.get("rounds", {}).values()
            for j in rnd.get("jobs", {}).values()
            if j.get("state") in ACTIVE_STATES
            for gpu in j.get("gpu_indices", [])
        }
        minimum = int(job.get("min_gpu_memory_mb", 0))
        preferred = job.get("node")
        candidates: dict[int, list[dict[str, Any]]] = {}
        for gpu in gpus:
            key = (gpu["node"], gpu["index"])
            if gpu["free"] and key not in reserved and gpu["memory_total_mb"] >= minimum:
                if preferred is None or gpu["node"] == preferred:
                    candidates.setdefault(gpu["node"], []).append(gpu)
        choices = []
        for node, available in candidates.items():
            if len(available) >= job["gpus"]:
                selected = sorted(available, key=lambda x: (-x["memory_total_mb"], x["index"]))[:job["gpus"]]
                choices.append((sum(x["memory_total_mb"] for x in selected), -node, node, selected))
        if not choices:
            raise ResearchCtlError(f"no currently free Grünau placement for {job['id']}; refresh inventory")
        _, _, node, selected = max(choices)
        return node, sorted(x["index"] for x in selected)

    def _submit_gruenau(self, plan: dict[str, Any], job: dict[str, Any], cluster: dict[str, Any],
                         state: dict[str, Any]) -> dict[str, Any]:
        snapshot = str(self._ensure_local_snapshot(plan["git_commit"]))
        local_dir = self.run_root / plan["project"] / plan["round_id"] / job["id"]
        local_dir.mkdir(parents=True, exist_ok=False)
        node, indices = self._choose_gruenau(job, state)
        script = self._runner_script(job, cluster, snapshot, str(local_dir))
        script_path = local_dir / "job.sh"
        script_path.write_text(script)
        script_path.chmod(0o700)
        json_dump(local_dir / "manifest.json", {"plan": plan, "job": job, "node": node, "gpu_indices": indices})
        ssh_host = cluster["ssh_template"].format(node=node)
        remote = (
            f"CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str, indices)))} "
            f"setsid nohup bash {shlex.quote(str(script_path))} >/dev/null 2>&1 </dev/null & echo $!"
        )
        result = run(["ssh", "-o", "BatchMode=yes", ssh_host, remote], timeout=30)
        return {
            "state": "SUBMITTED", "submitted_at": now(), "backend": "gruenau",
            "gpus": job["gpus"],
            "node": node, "gpu_indices": indices, "backend_id": result.stdout.strip(),
            "local_dir": str(local_dir), "remote_dir": str(local_dir),
        }

    def _sbatch_script(self, plan: dict[str, Any], job: dict[str, Any], cluster: dict[str, Any],
                       snapshot: str, remote_dir: str) -> str:
        opts = [
            "#!/usr/bin/env bash", f"#SBATCH --job-name=tj-{job['id'][:50]}",
            f"#SBATCH --time={slurm_time(parse_walltime_minutes(job['walltime_minutes']))}",
            f"#SBATCH --cpus-per-task={int(job.get('cpus', cluster.get('cpus_per_gpu', 4) * job['gpus']))}",
            f"#SBATCH --output={remote_dir}/slurm-%j.out",
            f"#SBATCH --error={remote_dir}/slurm-%j.err",
        ]
        partition = job.get("partition") or cluster.get("partition")
        if partition:
            opts.append(f"#SBATCH --partition={partition}")
        account = job.get("account") or cluster.get("account")
        if account:
            opts.append(f"#SBATCH --account={account}")
        qos = job.get("qos") or cluster.get("qos")
        if qos:
            opts.append(f"#SBATCH --qos={qos}")
        gpu_type = job.get("gpu_type") or cluster.get("gpu_type")
        gres = f"gpu:{gpu_type}:{job['gpus']}" if gpu_type else f"gpu:{job['gpus']}"
        opts.append(f"#SBATCH --gres={gres}")
        if cluster.get("export_none", False):
            opts.extend(["#SBATCH --export=NONE", "unset SLURM_EXPORT_ENV"])
        opts.append(self._runner_script(job, cluster, snapshot, remote_dir))
        return "\n".join(opts) + "\n"

    def _submit_slurm(self, plan: dict[str, Any], job: dict[str, Any], name: str,
                      cluster: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._ensure_remote_snapshot(name, cluster, plan["git_commit"])
        remote_dir = f"{cluster['project_root'].rstrip('/')}/runs/autonomy/{plan['project']}/{plan['round_id']}/{job['id']}"
        self._ssh(cluster, f"test ! -e {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}")
        script = self._sbatch_script(plan, job, cluster, snapshot, remote_dir)
        submit = self._ssh(cluster, "sbatch --parsable", input_text=script, timeout=60)
        backend_id = submit.stdout.strip().split(";")[0]
        if not backend_id.isdigit():
            raise ResearchCtlError(f"could not parse Slurm job id: {submit.stdout}")
        local_dir = self.run_root / plan["project"] / plan["round_id"] / job["id"]
        local_dir.mkdir(parents=True, exist_ok=False)
        (local_dir / "job.sbatch").write_text(script)
        json_dump(local_dir / "manifest.json", {"plan": plan, "job": job, "backend_id": backend_id})
        return {
            "state": "SUBMITTED", "submitted_at": now(), "backend": "slurm",
            "gpus": job["gpus"],
            "cluster": name, "backend_id": backend_id, "local_dir": str(local_dir),
            "remote_dir": remote_dir,
        }

    def _round_project(self, round_id: str, rnd: dict[str, Any]) -> str:
        return str(rnd.get("project") or infer_legacy_project(round_id, rnd.get("jobs", {}).keys()))

    def _usage(self, state: dict[str, Any]) -> dict[str, dict[str, float]]:
        usage = {slug: {"active_gpus": 0, "pending_jobs": 0, "gpu_hours_7d": 0.0} for slug in PROJECTS}
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
        for rid, rnd in state.get("rounds", {}).items():
            project = self._round_project(rid, rnd)
            if project not in usage:
                continue
            created = rnd.get("created_at")
            with contextlib.suppress(ValueError, TypeError):
                if created and dt.datetime.fromisoformat(created) >= cutoff:
                    usage[project]["gpu_hours_7d"] += float(rnd.get("projected_gpu_hours", 0))
            for job in rnd.get("jobs", {}).values():
                if job.get("state") in ACTIVE_STATES:
                    gpus = len(job.get("gpu_indices", [])) or int(job.get("gpus", 1))
                    usage[project]["active_gpus"] += gpus
                if job.get("state") in {"SUBMITTED", "PENDING"}:
                    usage[project]["pending_jobs"] += 1
        return usage

    def _waiting_projects(self, state: dict[str, Any], exclude_round: str = "") -> set[str]:
        waiting: set[str] = set()
        for path in (self.state_dir / "plans").glob("*.resolved.json"):
            with contextlib.suppress(OSError, json.JSONDecodeError):
                plan = json.loads(path.read_text())
                if plan.get("round_id") not in state.get("rounds", {}) and plan.get("round_id") != exclude_round:
                    if plan.get("project") in self.projects:
                        waiting.add(plan["project"])
        return waiting

    def _fair_admission_guard(self, plan: dict[str, Any], state: dict[str, Any]) -> None:
        project = plan["project"]
        manifest = self.project_manifest(project)
        budget = manifest["budget"]
        usage = self._usage(state)
        requested_gpus = sum(int(job["gpus"]) for job in plan["jobs"])
        after = usage[project]["active_gpus"] + requested_gpus
        if after > int(budget["maximum_active_gpus"]):
            raise ResearchCtlError(
                f"project GPU cap: {project} would use {after} active GPUs; "
                f"maximum is {budget['maximum_active_gpus']}"
            )
        pending_after = usage[project]["pending_jobs"] + sum(
            job["cluster"] != "gruenau" for job in plan["jobs"]
        )
        if pending_after > int(budget["maximum_pending_jobs"]):
            raise ResearchCtlError(
                f"project pending-job cap: {project} would have {pending_after}; "
                f"maximum is {budget['maximum_pending_jobs']}"
            )
        project_week = usage[project]["gpu_hours_7d"] + float(plan["projected_gpu_hours"])
        if project_week > float(budget["maximum_gpu_hours_7d"]):
            raise ResearchCtlError(
                f"project 7-day GPU-hour limit: {project_week:.2f} > "
                f"{float(budget['maximum_gpu_hours_7d']):.2f} for {project}"
            )
        waiting = self._waiting_projects(state, plan["round_id"]) - {project}
        latest = self.state_dir / "inventory/latest.json"
        visible = 0
        if latest.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                inv = json.loads(latest.read_text())
                visible = sum(len(c.get("gpus", [])) for c in inv.get("clusters", {}).values())
        total_slots = max(visible, int(self.cfg["limits"]["max_active_jobs"]), 1)
        fair_cap = max(int(budget["guaranteed_active_gpus"]), int(total_slots * 0.40 + 0.999))
        if waiting and after > fair_cap:
            raise ResearchCtlError(
                f"fair-share deferral: {project} would use {after}/{total_slots} active GPU slots "
                f"while {', '.join(sorted(waiting))} has a runnable plan waiting; cap is {fair_cap}"
            )

    def _global_gpu_hour_guard(self, plan: dict[str, Any], state: dict[str, Any]) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
        recent = 0.0
        for rnd in state.get("rounds", {}).values():
            with contextlib.suppress(ValueError, TypeError):
                if rnd.get("created_at") and dt.datetime.fromisoformat(rnd["created_at"]) >= cutoff:
                    recent += float(rnd.get("projected_gpu_hours", 0))
        limit = float(self.cfg["limits"].get("max_gpu_hours_7d", 1e30))
        if recent + float(plan["projected_gpu_hours"]) > limit:
            raise ResearchCtlError(
                f"7-day GPU-hour limit would be exceeded: {recent:.2f} reserved + "
                f"{plan['projected_gpu_hours']:.2f} planned > {limit:.2f}"
            )

    def projects_status(self) -> None:
        for slug, manifest in self.projects.items():
            p = manifest["project"]
            b = manifest["budget"]
            print(
                f"{slug}: active={p['scientifically_active']} auto_submit={p['autonomous_submission']} "
                f"guarantee={b['guaranteed_active_gpus']} max_active={b['maximum_active_gpus']} "
                f"plan={p['plan_path']} worktree={p['worktree']}"
            )

    def allocation_text(self) -> str:
        state = self.load_state()
        usage = self._usage(state)
        waiting = self._waiting_projects(state)
        global_recent = sum(x["gpu_hours_7d"] for x in usage.values())
        lines = ["Future admissions are fair-share controlled; running jobs are never preempted."]
        for slug in PROJECTS:
            b = self.project_manifest(slug)["budget"]
            u = usage[slug]
            borrowed = max(0, int(u["active_gpus"]) - int(b["guaranteed_active_gpus"]))
            reason = "runnable plan waiting" if slug in waiting else "no unadmitted resolved plan"
            lines.append(
                f"{slug}: active_gpus={int(u['active_gpus'])}, pending_jobs={int(u['pending_jobs'])}, "
                f"guaranteed={b['guaranteed_active_gpus']}, borrowed={borrowed}, "
                f"round_remaining={float(b['maximum_gpu_hours_per_round']):.1f} GPU-h, "
                f"weekly_remaining={max(0.0, float(b['maximum_gpu_hours_7d'])-u['gpu_hours_7d']):.1f} GPU-h; {reason}"
            )
        lines.append(
            f"global weekly remaining={max(0.0, float(self.cfg['limits']['max_gpu_hours_7d'])-global_recent):.1f} GPU-h"
        )
        return "\n".join(lines)

    def allocation(self) -> None:
        print(self.allocation_text())

    def migrate_state(self, *, execute: bool, rollback: Path | None = None) -> None:
        """Upgrade legacy state metadata without altering any job identity or path."""
        with self.lock():
            current_bytes = self.state_path.read_bytes()
            current = json.loads(current_bytes)
            if rollback is not None:
                source = rollback.resolve()
                allowed = (self.state_dir / "migrations").resolve()
                if allowed not in source.parents or not source.is_file():
                    raise ResearchCtlError("rollback source must be a migration state backup")
                restored = json.loads(source.read_text())
                before_ids = {
                    (rid, jid, j.get("backend_id"), j.get("local_dir"), j.get("remote_dir"))
                    for rid, r in current.get("rounds", {}).items() for jid, j in r.get("jobs", {}).items()
                }
                restored_ids = {
                    (rid, jid, j.get("backend_id"), j.get("local_dir"), j.get("remote_dir"))
                    for rid, r in restored.get("rounds", {}).items() for jid, j in r.get("jobs", {}).items()
                }
                if before_ids != restored_ids:
                    raise ResearchCtlError("rollback would change registered job identities or paths")
                json_dump(self.state_path, restored)
                print(f"rolled back controller metadata from {source}; accepted jobs unchanged")
                return
            migrated = json.loads(current_bytes)
            changed = migrated.get("schema_version", 1) < 2
            annotations: dict[str, str] = {}
            for rid, rnd in migrated.get("rounds", {}).items():
                project = self._round_project(rid, rnd)
                annotations[rid] = project
                if "project" not in rnd:
                    rnd["project"] = project
                    rnd["legacy"] = True
                    changed = True
            already_migrated = (
                current.get("schema_version") == 2
                and current.get("migration", {}).get("three_project", {}).get("version") == 1
            )
            if already_migrated:
                print(json.dumps({
                    "would_change": False,
                    "rounds": len(current.get("rounds", {})),
                    "jobs": sum(len(r.get("jobs", {})) for r in current.get("rounds", {}).values()),
                    "classification": annotations,
                }, indent=2, sort_keys=True))
                print("state is already migrated; no write performed")
                return
            migrated["schema_version"] = 2
            migrated.setdefault("migration", {})["three_project"] = {
                "version": 1,
                "source_sha256": hashlib.sha256(current_bytes).hexdigest(),
                "round_projects": annotations,
            }
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = self.state_dir / "migrations" / stamp
            print(json.dumps({
                "would_change": changed or current != migrated,
                "rounds": len(migrated.get("rounds", {})),
                "jobs": sum(len(r.get("jobs", {})) for r in migrated.get("rounds", {}).values()),
                "classification": annotations,
            }, indent=2, sort_keys=True))
            if not execute:
                print("dry run; state was not modified")
                return
            backup_dir.mkdir(parents=True, exist_ok=False)
            backup = backup_dir / "state.before-v2.json"
            backup.write_bytes(current_bytes)
            backup.chmod(0o400)
            (backup_dir / "state.before-v2.sha256").write_text(
                hashlib.sha256(current_bytes).hexdigest() + "  state.before-v2.json\n"
            )
            json_dump(self.state_path, migrated)
            print(f"migrated state atomically; backup={backup}")

    def submit(self, plan_path: Path | None, *, execute: bool) -> None:
        plan = self.validate_plan(self.load_plan(plan_path), resolve_auto=True)
        print(json.dumps({"round_id": plan["round_id"], "jobs": len(plan["jobs"]), "projected_gpu_hours": plan["projected_gpu_hours"]}, indent=2))
        if not execute:
            print("dry run only; add --execute to submit")
            return
        with self.lock():
            if self.stop_path.exists():
                raise ResearchCtlError(f"STOP file present: {self.stop_path}")
            self._local_storage_guard()
            self._human_review_guard()
            state = self.load_state()
            if state.get("paused"):
                raise ResearchCtlError(f"controller is paused: {state.get('pause_reason', '')}")
            active = sum(
                job.get("state") in ACTIVE_STATES
                for rnd in state.get("rounds", {}).values()
                for job in rnd.get("jobs", {}).values()
            )
            if active + len(plan["jobs"]) > int(self.cfg["limits"]["max_active_jobs"]):
                raise ResearchCtlError(
                    f"active-job limit would be exceeded: {active} active + {len(plan['jobs'])} planned"
                )
            self._fair_admission_guard(plan, state)
            self._global_gpu_hour_guard(plan, state)
            if plan["round_id"] in state["rounds"]:
                raise ResearchCtlError(f"round already registered: {plan['round_id']}")
            round_state = {
                "project": plan["project"], "legacy": False,
                "decision": plan["decision"], "git_commit": plan["git_commit"],
                "projected_gpu_hours": plan["projected_gpu_hours"], "created_at": now(),
                "jobs": {}, "oversight_woken": False,
            }
            state["rounds"][plan["round_id"]] = round_state
            self.save_state(state)  # reserve the id before any external mutation
            try:
                for job in plan["jobs"]:
                    cluster = self.cfg["clusters"][job["cluster"]]
                    self._remote_storage_guard(job["cluster"], cluster)
                    if cluster["kind"] == "gruenau":
                        job_state = self._submit_gruenau(plan, job, cluster, state)
                    else:
                        job_state = self._submit_slurm(plan, job, job["cluster"], cluster)
                    round_state["jobs"][job["id"]] = job_state
                    self.save_state(state)
                    print(f"submitted {job['id']} -> {job['cluster']}:{job_state['backend_id']}")
            except Exception as exc:
                round_state["submission_error"] = str(exc)
                round_state["state"] = "PARTIAL_SUBMISSION"
                self.save_state(state)
                raise
            round_state["state"] = "ACTIVE"
            self.save_state(state)

    def _retrieve(self, cluster: dict[str, Any], job_state: dict[str, Any]) -> None:
        local = Path(job_state["local_dir"])
        local.mkdir(parents=True, exist_ok=True)
        max_mb = int(self.cfg["storage"].get("max_retrieved_file_mb", 256))
        includes = self.cfg["storage"].get("retrieve_patterns", [])
        argv = ["rsync", "-a", "--prune-empty-dirs", f"--max-size={max_mb}m"]
        for pattern in includes:
            argv.extend(["--include", pattern])
        argv.extend(["--exclude", "*", f"{cluster['ssh']}:{job_state['remote_dir'].rstrip('/')}/", str(local) + "/"])
        result = run(argv, timeout=180, check=False)
        if result.returncode:
            job_state["retrieval_error"] = (result.stderr or result.stdout).strip()
            job_state["retrieval_attempts"] = int(job_state.get("retrieval_attempts", 0)) + 1
        else:
            job_state["retrieved_at"] = now()
            job_state.pop("retrieval_error", None)

    def refresh(self) -> None:
        with self.lock():
            state = self.load_state()
            for round_id, round_state in state.get("rounds", {}).items():
                for job_id, job_state in round_state.get("jobs", {}).items():
                    if job_state.get("state") not in ACTIVE_STATES:
                        continue
                    if job_state["backend"] == "gruenau":
                        state_file = Path(job_state["local_dir"]) / "state"
                        if state_file.exists():
                            observed = state_file.read_text().strip()
                            if observed:
                                job_state["state"] = observed
                    else:
                        cluster = self.cfg["clusters"][job_state["cluster"]]
                        jid = job_state["backend_id"]
                        query = self._ssh(
                            cluster,
                            f"sacct -n -P -X -j {jid} -o State | head -1; squeue -h -j {jid} -o '%T' | head -1",
                            check=False,
                        )
                        values = [x.strip().split("+")[0] for x in query.stdout.splitlines() if x.strip()]
                        observed = values[0] if values else "UNKNOWN"
                        if observed in TERMINAL_SLURM:
                            job_state["state"] = "COMPLETED" if observed in SUCCESS_SLURM else "FAILED"
                            job_state["scheduler_state"] = observed
                        elif observed:
                            job_state["state"] = observed
                        if job_state["state"] not in ACTIVE_STATES:
                            self._retrieve(cluster, job_state)
                    if job_state.get("state") not in ACTIVE_STATES:
                        job_state["finished_at"] = job_state.get("finished_at", now())
                        print(f"{round_id}/{job_id}: {job_state['state']}")
                jobs = round_state.get("jobs", {}).values()
                if jobs and all(job.get("state") not in ACTIVE_STATES for job in jobs):
                    round_state["state"] = "TERMINAL"
                    round_state["finished_at"] = round_state.get("finished_at", now())
            self.save_state(state)

    def status(self, project: str | None = None) -> None:
        if project:
            self.project_manifest(project)
        state = self.load_state()
        unreviewed = self._unreviewed_reports()
        print(
            f"paused={state.get('paused', False)} stop_file={self.stop_path.exists()} "
            f"review_required_unread={len(unreviewed)}"
        )
        if not state.get("rounds"):
            print("no rounds registered")
            return
        for round_id, rnd in state["rounds"].items():
            round_project = self._round_project(round_id, rnd)
            if project and round_project != project:
                continue
            print(f"{round_id}: {rnd.get('state', 'UNKNOWN')} project={round_project} commit={rnd.get('git_commit', '')[:12]} gpu_h={rnd.get('projected_gpu_hours')}")
            for job_id, job in rnd.get("jobs", {}).items():
                location = f"gruenau{job.get('node')}" if job.get("backend") == "gruenau" else job.get("cluster")
                print(f"  {job_id}: {job.get('state')} {location}:{job.get('backend_id')}")

    def pause(self, reason: str) -> None:
        with self.lock():
            state = self.load_state()
            state.update({"paused": True, "pause_reason": reason, "paused_at": now()})
            self.save_state(state)
        print(f"paused: {reason}")

    def resume(self) -> None:
        with self.lock():
            state = self.load_state()
            state.update({"paused": False, "pause_reason": None, "resumed_at": now()})
            self.save_state(state)
        print("resumed")

    def _assert_clean(self, root: Path | None = None) -> None:
        root = root or self.root
        dirty = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
        if dirty:
            raise ResearchCtlError(
                "autonomous Codex wake requires a clean dedicated checkout; current worktree is dirty"
            )

    def _finalize_oversight_changes(self, root: Path | None = None, project: str | None = None) -> str:
        root = root or self.root
        codex_cfg = self.cfg["codex"]
        tracked = run(
            ["git", "diff", "--name-only", "HEAD"], cwd=root
        ).stdout.splitlines()
        untracked = run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=root
        ).stdout.splitlines()
        paths = sorted(set(tracked + untracked))
        if not paths:
            return run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
        if codex_cfg.get("require_explanatory_report", False) and not any(
            path.startswith("research/reports/") and path.endswith("/REPORT.md") for path in paths
        ):
            raise ResearchCtlError(
                "Codex oversight did not create or update a validated explanatory REPORT.md"
            )
        violations = protected_path_violations(paths, codex_cfg.get("protected_paths", []))
        if project:
            violations.extend(sibling_memory_violations(paths, project))
            violations = sorted(set(violations))
        if violations:
            raise ResearchCtlError(
                "Codex changed protected paths; changes were left for human inspection: "
                + ", ".join(violations)
            )
        max_files = int(codex_cfg.get("max_changed_files", 100))
        if len(paths) > max_files:
            raise ResearchCtlError(f"Codex changed {len(paths)} files; limit is {max_files}")
        total_bytes = sum(
            (root / path).stat().st_size
            for path in paths
            if (root / path).is_file()
        )
        max_bytes = int(codex_cfg.get("max_changed_bytes", 5_000_000))
        if total_bytes > max_bytes:
            raise ResearchCtlError(
                f"Codex changed {total_bytes} bytes; autonomous limit is {max_bytes}"
            )
        verify = codex_cfg.get("verification_command", [])
        if verify:
            verify_argv = [str(x) for x in verify]
            if not Path(verify_argv[0]).is_absolute() and not (root / verify_argv[0]).exists() and (self.root / verify_argv[0]).exists():
                verify_argv[0] = str(self.root / verify_argv[0])
            result = run(verify_argv, cwd=root, timeout=int(codex_cfg.get("verification_timeout_seconds", 600)), check=False)
            verification_log = self.state_dir / "oversight" / "latest-verification.log"
            verification_log.write_text(result.stdout + result.stderr)
            if result.returncode:
                raise ResearchCtlError(
                    f"oversight verification failed ({result.returncode}); see {verification_log}"
                )
        run(["git", "add", "-A", "--", *paths], cwd=root)
        staged = run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False)
        if staged.returncode == 0:
            return run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
        stamp = dt.datetime.now().strftime("%Y-%m-%d")
        run(
            ["git", "commit", "-m", f"research({project or 'shared'}): autonomous decision cycle {stamp}"],
            cwd=root, timeout=120,
        )
        return run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()

    def _wake(self, project: str) -> None:
        manifest = self.project_manifest(project)
        codex_cfg = self.cfg["codex"]
        if not codex_cfg.get("enabled", False):
            raise ResearchCtlError("Codex wake is disabled in configuration")
        if self.stop_path.exists():
            raise ResearchCtlError(f"STOP file present: {self.stop_path}")
        worktree = Path(manifest["project"]["worktree"])
        if not worktree.is_dir():
            raise ResearchCtlError(f"project worktree does not exist: {worktree}")
        self._assert_clean(worktree)
        prompt_path = worktree / manifest["project"]["prompt"]
        state = self.load_state()
        context = [
            f"Integration repository: {self.root}",
            f"Project steering inbox: {self.state_dir / 'steering/inbox' / project}",
            "Newly terminal compact summaries for this project:",
        ]
        for rid, rnd in state.get("rounds", {}).items():
            if self._round_project(rid, rnd) != project or rnd.get("state") != "TERMINAL" or rnd.get("oversight_woken"):
                continue
            for job in rnd.get("jobs", {}).values():
                context.append(str(Path(job["local_dir"]) / "run_summary.json"))
        prompt = prompt_path.read_text() + "\n\n" + "\n".join(context) + "\n\n" + self.allocation_text()
        output_dir = self.state_dir / "oversight" / project
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        last = output_dir / f"{stamp}.md"
        events = output_dir / f"{stamp}.jsonl"
        executable = codex_cfg.get("executable", "codex")
        argv = [
            executable, "--search", "exec", "--model", codex_cfg["model"],
            "--config", f"model_reasoning_effort={json.dumps(codex_cfg['reasoning_effort'])}",
            "--config", 'approval_policy="never"', "--sandbox", codex_cfg.get("sandbox", "workspace-write"),
            "--cd", str(worktree), "--ephemeral", "--json", "--output-last-message", str(last), "-",
        ]
        with events.open("w") as log:
            process = subprocess.run(argv, input=prompt, text=True, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise ResearchCtlError(f"Codex oversight failed ({process.returncode}); see {events}")
        commit = self._finalize_oversight_changes(worktree, project)
        with self.lock():
            state = self.load_state()
            state.setdefault("integration_queue", []).append({
                "project": project, "branch": manifest["project"]["branch"],
                "commit": commit, "created_at": now(), "status": "awaiting_integration",
            })
            self.save_state(state)
        print(f"oversight response: {last}")
        print(f"oversight commit: {commit}")
        plan_path = worktree / manifest["project"]["plan_path"]
        if plan_path.exists():
            plan = self.validate_plan(json.loads(plan_path.read_text()), resolve_auto=False)
            if plan["git_commit"] == "AUTO":
                plan["git_commit"] = commit
            resolved = self.state_dir / "plans" / f"{plan['round_id']}.resolved.json"
            json_dump(resolved, self.validate_plan(plan))
            print(f"resolved plan: {resolved}")
            if codex_cfg.get("auto_submit_after_wake", False) and manifest["project"].get("autonomous_submission", False):
                self.submit(resolved, execute=True)
            else:
                print("plan awaits human review; auto_submit_after_wake=false")
        else:
            print("oversight produced no NEXT_PLAN.json")

    def wake(self, project: str) -> None:
        with self.oversight_lock():
            self._wake(project)

    def tick(self) -> None:
        self._local_storage_guard()
        self.refresh()
        state = self.load_state()
        if self.stop_path.exists() or state.get("paused"):
            print(f"observation only; paused={state.get('paused', False)} STOP={self.stop_path.exists()}")
            return
        terminal = [
            (rid, rnd) for rid, rnd in state.get("rounds", {}).items()
            if rnd.get("state") == "TERMINAL" and not rnd.get("oversight_woken")
        ]
        if not terminal:
            print("no completed round awaiting oversight")
            return
        if not self.cfg["codex"].get("enabled", False):
            print("completed round awaits oversight; Codex wake is disabled")
            return
        if len(self._unreviewed_reports()) >= int(self.cfg["limits"].get("max_unreviewed_reports", 1)):
            print("completed rounds await oversight; unread-report limit reached")
            return
        by_project: dict[str, list[str]] = {}
        for rid, rnd in terminal:
            project = self._round_project(rid, rnd)
            if project in self.projects:
                by_project.setdefault(project, []).append(rid)
        if by_project:
            project = sorted(by_project)[0]
            self.wake(project)
            with self.lock():
                state = self.load_state()
                for rid in by_project[project]:
                    state["rounds"][rid]["oversight_woken"] = True
                    state["rounds"][rid]["oversight_woken_at"] = now()
                self.save_state(state)

    def watch(self, interval: int) -> None:
        print(f"watching every {interval}s; Ctrl-C to stop")
        while True:
            try:
                self.tick()
            except ResearchCtlError as exc:
                print(f"tick error: {exc}", file=sys.stderr)
            time.sleep(interval)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="controller TOML (or set RESEARCH_CONFIG)")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("init", "doctor", "inventory", "storage", "refresh", "resume", "tick", "projects", "allocation"):
        sub.add_parser(name)
    status = sub.add_parser("status")
    status.add_argument("--project", choices=PROJECTS)
    status.add_argument("--all", action="store_true")
    wake = sub.add_parser("wake")
    wake.add_argument("--project", required=True, choices=PROJECTS)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("plan", nargs="?", type=Path)
    validate.add_argument("--project", choices=PROJECTS)
    finalize = sub.add_parser("finalize-plan")
    finalize.add_argument("--project", choices=PROJECTS)
    submit = sub.add_parser("submit-plan")
    submit.add_argument("plan", nargs="?", type=Path)
    submit.add_argument("--execute", action="store_true")
    pause = sub.add_parser("pause")
    pause.add_argument("reason")
    watch = sub.add_parser("watch")
    watch.add_argument("--interval", type=int, default=60)
    migrate = sub.add_parser("migrate-state")
    mode = migrate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--rollback", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        ctl = Controller(args.config)
        if args.command == "validate-plan":
            plan = ctl.validate_plan(ctl.load_plan(args.plan, args.project), resolve_auto=False)
            print(json.dumps(plan, indent=2))
        elif args.command == "submit-plan":
            ctl.submit(args.plan, execute=args.execute)
        elif args.command == "pause":
            ctl.pause(args.reason)
        elif args.command == "watch":
            ctl.watch(max(10, args.interval))
        elif args.command == "finalize-plan":
            ctl.finalize_plan(args.project)
        elif args.command == "status":
            ctl.status(args.project)
        elif args.command == "wake":
            ctl.wake(args.project)
        elif args.command == "projects":
            ctl.projects_status()
        elif args.command == "migrate-state":
            ctl.migrate_state(execute=args.execute, rollback=args.rollback)
        else:
            getattr(ctl, args.command.replace("-", "_"))()
        return 0
    except (ResearchCtlError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"researchctl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
