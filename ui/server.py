#!/usr/bin/env python3
"""Local TextJEPA report dashboard with optional SSH/rsync synchronization."""

from __future__ import annotations

import argparse
import datetime as dt
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import mimetypes
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser


SAFE_SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
STATIC = Path(__file__).resolve().parent / "static"


def inline(text: str, bundle_rel: str) -> str:
    tokens: list[str] = []

    def hold(value: str) -> str:
        tokens.append(value)
        return f"\x00{len(tokens) - 1}\x00"

    def image(match: re.Match[str]) -> str:
        alt, target = match.group(1), match.group(2)
        if "://" in target:
            src = target
        else:
            src = "/files/" + quote(str(Path(bundle_rel) / target))
        return hold(f'<figure><img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}"><figcaption>{html.escape(alt)}</figcaption></figure>')

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        href = target if "://" in target else "/files/" + quote(str(Path(bundle_rel) / target))
        return hold(f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noopener">{html.escape(label)}</a>')

    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)", image, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*", r"<em>\1</em>", text)
    for i, token in enumerate(tokens):
        text = text.replace(f"\x00{i}\x00", token)
    return text


def markdown(text: str, bundle_rel: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + inline(" ".join(x.strip() for x in paragraph), bundle_rel) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_paragraph(); close_list()
            if code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines.clear()
            code = not code
            i += 1
            continue
        if code:
            code_lines.append(line); i += 1; continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-", lines[i + 1]):
            flush_paragraph(); close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            headers, data = rows[0], rows[2:]
            out.append("<div class=table-wrap><table><thead><tr>" + "".join(f"<th>{inline(x, bundle_rel)}</th>" for x in headers) + "</tr></thead><tbody>")
            for row in data:
                out.append("<tr>" + "".join(f"<td>{inline(x, bundle_rel)}</td>" for x in row) + "</tr>")
            out.append("</tbody></table></div>")
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph(); close_list()
            level = len(heading.group(1))
            label = heading.group(2)
            anchor = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
            out.append(f'<h{level} id="{anchor}">{inline(label, bundle_rel)}</h{level}>')
        elif re.match(r"^[-*]\s+", line):
            flush_paragraph()
            if list_kind != "ul":
                close_list(); out.append("<ul>"); list_kind = "ul"
            out.append("<li>" + inline(re.sub(r"^[-*]\s+", "", line), bundle_rel) + "</li>")
        elif re.match(r"^\d+[.)]\s+", line):
            flush_paragraph()
            if list_kind != "ol":
                close_list(); out.append("<ol>"); list_kind = "ol"
            out.append("<li>" + inline(re.sub(r"^\d+[.)]\s+", "", line), bundle_rel) + "</li>")
        elif line.startswith("> "):
            flush_paragraph(); close_list()
            out.append("<blockquote>" + inline(line[2:], bundle_rel) + "</blockquote>")
        elif not line.strip():
            flush_paragraph(); close_list()
        else:
            if list_kind:
                close_list()
            paragraph.append(line)
        i += 1
    flush_paragraph(); close_list()
    return "\n".join(out)


class Dashboard:
    def __init__(self, source: Path, remote: str | None, remote_root: str, sync_seconds: int) -> None:
        self.source = source.resolve()
        self.remote = remote
        self.remote_root = remote_root.rstrip("/")
        self.sync_seconds = sync_seconds
        self.lock = threading.Lock()
        self.last_sync: dict[str, object] = {"at": None, "ok": not remote, "message": "local mode"}
        self.stop = threading.Event()

    def sync(self) -> None:
        if not self.remote:
            self.last_sync = {"at": dt.datetime.now().isoformat(timespec="seconds"), "ok": True, "message": "local mode"}
            return
        self.source.mkdir(parents=True, exist_ok=True)
        command = [
            "rsync", "-az", "--delete", "--prune-empty-dirs",
            "--include=research/", "--include=research/reports/***",
            "--include=.researchctl/", "--include=.researchctl/state.json",
            "--exclude=*", f"{self.remote}:{self.remote_root}/", str(self.source) + "/",
        ]
        with self.lock:
            result = subprocess.run(command, text=True, capture_output=True)
            self.last_sync = {
                "at": dt.datetime.now().isoformat(timespec="seconds"),
                "ok": result.returncode == 0,
                "message": (result.stderr or result.stdout or "synchronized").strip(),
            }

    def sync_loop(self) -> None:
        while not self.stop.wait(self.sync_seconds):
            self.sync()

    def reports(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        report_root = self.source / "research/reports"
        if not report_root.exists():
            return records
        for meta_path in report_root.rglob("report.json"):
            try:
                meta = json.loads(meta_path.read_text())
                report_path = meta_path.parent / meta["report"]
                body = report_path.read_bytes()
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            relative_bundle = str(meta_path.parent.relative_to(self.source))
            record = dict(meta)
            record.update({
                "hash": sha256(body).hexdigest()[:16],
                "bundle": relative_bundle,
                "markdown_path": str(report_path.relative_to(self.source)),
            })
            records.append(record)
        return sorted(records, key=lambda x: (str(x.get("created_at", "")), str(x.get("id", ""))), reverse=True)

    def status(self) -> dict[str, object]:
        path = self.source / ".researchctl/state.json"
        state: dict[str, object] = {}
        if path.exists():
            try:
                state = json.loads(path.read_text())
            except json.JSONDecodeError:
                state = {"error": "controller state could not be parsed"}
        return {"sync": self.last_sync, "controller": state}

    def send_steering(self, project: str, report_id: str, message: str) -> Path:
        if not SAFE_SLUG.fullmatch(project) or not SAFE_SLUG.fullmatch(report_id):
            raise ValueError("invalid project or report id")
        message = message.strip()
        if len(message) < 10 or len(message) > 20_000:
            raise ValueError("steering note must contain 10–20,000 characters")
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        relative = Path(".researchctl/steering/inbox") / project / f"{stamp}-{report_id}.md"
        content = (
            f"# Human steering for {project}\n\n"
            f"Report: {report_id}\n\nReceived: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
            f"## Direction\n\n{message}\n"
        )
        local = self.source / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content)
        if self.remote:
            remote_path = f"{self.remote_root}/{relative}"
            remote_command = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && cat > {shlex.quote(remote_path)}"
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", self.remote, remote_command], input=content, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout).strip())
        return local

    def acknowledge(self, report_id: str, report_hash: str, read: bool) -> Path:
        if not SAFE_SLUG.fullmatch(report_id) or not re.fullmatch(r"[0-9a-f]{16}", report_hash):
            raise ValueError("invalid report receipt")
        relative = Path(".researchctl/read_receipts") / f"{report_id}.json"
        local = self.source / relative
        content = json.dumps({
            "schema_version": 1, "report_id": report_id, "hash": report_hash,
            "read_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2) + "\n"
        if read:
            local.parent.mkdir(parents=True, exist_ok=True); local.write_text(content)
        elif local.exists():
            local.unlink()
        if self.remote:
            remote_path = f"{self.remote_root}/{relative}"
            if read:
                remote_command = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && cat > {shlex.quote(remote_path)}"
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", self.remote, remote_command], input=content, text=True, capture_output=True)
            else:
                remote_command = f"rm -f {shlex.quote(remote_path)}"
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", self.remote, remote_command], text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout).strip())
        return local


def handler_factory(dashboard: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value: object, status: int = 200) -> None:
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data)

        def send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404); return
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers(); self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/reports":
                self.send_json({"reports": dashboard.reports(), "status": dashboard.status()}); return
            if parsed.path == "/api/report":
                wanted = parse_qs(parsed.query).get("id", [""])[0]
                report = next((x for x in dashboard.reports() if x.get("id") == wanted), None)
                if not report:
                    self.send_json({"error": "report not found"}, 404); return
                source = dashboard.source / str(report["markdown_path"])
                self.send_json({"report": report, "html": markdown(source.read_text(), str(report["bundle"]))}); return
            if parsed.path.startswith("/files/"):
                relative = Path(unquote(parsed.path[len("/files/"):]))
                target = (dashboard.source / relative).resolve()
                if dashboard.source not in target.parents:
                    self.send_error(403); return
                self.send_file(target); return
            name = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (STATIC / name).resolve()
            if STATIC not in target.parents and target != STATIC:
                self.send_error(403); return
            self.send_file(target)

        def do_POST(self) -> None:
            if self.path not in {"/api/steer", "/api/ack"} or self.headers.get("X-TextJEPA-UI") != "1":
                self.send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN); return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 25_000)
                value = json.loads(self.rfile.read(length))
                if self.path == "/api/steer":
                    path = dashboard.send_steering(value["project"], value["report_id"], value["message"])
                else:
                    path = dashboard.acknowledge(value["report_id"], value["hash"], bool(value["read"]))
                self.send_json({"ok": True, "saved": str(path)})
            except (ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local repo/cache root")
    parser.add_argument("--remote", help="SSH host or alias used by rsync")
    parser.add_argument("--remote-root", default="/vol/home-vol2/ml/laitenbf/TextJEPA")
    parser.add_argument("--sync-seconds", type=int, default=60)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--once", action="store_true", help="sync and print the report index, then exit")
    args = parser.parse_args()
    if args.source:
        source = args.source
    elif args.remote:
        source = Path.home() / ".cache/textjepa-research-ui"
    else:
        source = Path(__file__).resolve().parents[1]
    dashboard = Dashboard(source, args.remote, args.remote_root, max(15, args.sync_seconds))
    dashboard.sync()
    if args.once:
        print(json.dumps({"reports": dashboard.reports(), "status": dashboard.status()}, indent=2))
        return 0
    if args.remote:
        threading.Thread(target=dashboard.sync_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(dashboard))
    url = f"http://127.0.0.1:{args.port}"
    print(f"TextJEPA Research UI: {url}")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop.set(); server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
