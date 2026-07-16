#!/usr/bin/env python3
"""Validate that autonomous research reports are genuinely explanatory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REQUIRED_SECTIONS = [
    "The one-sentence answer",
    "First, the idea in everyday language",
    "Why this question matters",
    "What we tested",
    "What a fair comparison means here",
    "What happened",
    "The intuitive picture",
    "The technical details",
    "What we can conclude",
    "What we cannot conclude",
    "What happens next",
    "Words used in this report",
    "Questions for you",
]
REQUIRED_METADATA = {
    "schema_version", "id", "project", "title", "created_at", "status",
    "decision", "plain_summary", "report", "review_required",
}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", re.sub(r"`[^`]*`", "", text)))


def sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    return {
        match.group(1).strip(): text[match.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
        for i, match in enumerate(matches)
    }


def validate_bundle(bundle: Path, minimum_words: int) -> list[str]:
    errors: list[str] = []
    report = bundle / "REPORT.md"
    metadata_path = bundle / "report.json"
    if not report.exists():
        return [f"{bundle}: missing REPORT.md"]
    if not metadata_path.exists():
        return [f"{bundle}: missing report.json"]
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{metadata_path}: {exc}"]
    missing = REQUIRED_METADATA - metadata.keys()
    errors.extend(f"{metadata_path}: missing metadata field {field}" for field in sorted(missing))
    if metadata.get("schema_version") != 1:
        errors.append(f"{metadata_path}: schema_version must be 1")
    for field in ("id", "project"):
        if not SAFE_ID.fullmatch(str(metadata.get(field, ""))):
            errors.append(f"{metadata_path}: unsafe {field}")
    if metadata.get("report") != "REPORT.md":
        errors.append(f"{metadata_path}: report must be REPORT.md")
    if not isinstance(metadata.get("review_required"), bool):
        errors.append(f"{metadata_path}: review_required must be true or false")
    text = report.read_text()
    found = sections(text)
    for heading in REQUIRED_SECTIONS:
        if heading not in found:
            errors.append(f"{report}: missing section '## {heading}'")
        elif words(found[heading]) < 20:
            errors.append(f"{report}: section '{heading}' is too short to be explanatory")
    if words(text) < minimum_words:
        errors.append(f"{report}: {words(text)} words; require at least {minimum_words}")
    if words(found.get("First, the idea in everyday language", "")) < 220:
        errors.append(f"{report}: everyday-language explanation must contain at least 220 words")
    if words(found.get("The technical details", "")) < 250:
        errors.append(f"{report}: technical details must contain at least 250 words")
    if not re.search(r"^\|.+\|\s*$\n^\|\s*:?-", text, re.MULTILINE):
        errors.append(f"{report}: include at least one Markdown comparison table")
    images = re.findall(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)", text)
    if not images:
        errors.append(f"{report}: include at least one intuitive figure")
    for alt, target in images:
        if words(alt) < 5:
            errors.append(f"{report}: figure alt text must explain what the reader should see")
        if "://" not in target and not (bundle / target).resolve().is_file():
            errors.append(f"{report}: missing figure {target}")
    glossary = found.get("Words used in this report", "")
    if len(re.findall(r"^[-*]\s+\*?\*?[^\n:]+[:—-]", glossary, re.MULTILINE)) < 5:
        errors.append(f"{report}: glossary must define at least five terms")
    questions = found.get("Questions for you", "")
    if len(re.findall(r"^[-*]\s+", questions, re.MULTILINE)) < 2:
        errors.append(f"{report}: include at least two concrete steering questions")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("research/reports"))
    parser.add_argument("--minimum-words", type=int, default=1200)
    args = parser.parse_args()
    if not args.root.exists():
        print(f"No report root yet: {args.root}")
        return 0
    bundles = sorted({path.parent for path in args.root.rglob("REPORT.md")})
    errors = [error for bundle in bundles for error in validate_bundle(bundle, args.minimum_words)]
    if errors:
        print("Report validation failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(bundles)} explanatory report bundle(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
