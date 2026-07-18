---
name: explain-research
description: Create a self-contained, beginner-first TextJEPA research report when the human explicitly requests one or when a report is materially useful; reports are never prerequisites for experiments or plans.
---

# Explain a research cycle from first principles

Read `research/templates/REPORT.md` and use it without renaming or omitting
sections. Write one bundle under
`research/reports/<project>/<date>-<intuitive-name>/` containing `REPORT.md`,
`report.json`, and local figures.

Assume the reader barely knows what a neural network or language model is.
Begin with a concrete everyday analogy. Explain the problem, inputs, desired
behavior, and meaning of success before using project vocabulary. Introduce one
new concept at a time. Define every abbreviation on first use and in the
glossary. Prefer a longer clear explanation over compressed specialist prose.

Use an ICLR-paper standard for evidence but a beginner-first teaching order:
plain claim, motivation, concrete design, fairness, results, intuition,
technical protocol, supported conclusion, unsupported conclusion, and next
decision. State what every model can see. Explain every metric in words before
showing its value. Keep failed and invalid runs visible.

Create at least one figure that teaches a mechanism or comparison rather than
decorating the report. Give it descriptive alt text and a caption telling the
reader what to notice and why. Create a readable comparison table and interpret
each important row in prose. Use intuitive full names rather than internal run
abbreviations.

End with concrete questions through which the human can change priorities,
risk tolerance, compute allocation, or the next scientific decision. Mark
`review_required: true` in `report.json` for scientific conclusions or
direction-changing proposals.

Run:

```bash
.venv2/bin/python automation/validate_reports.py research/reports
```

Validation applies only to a report that is intentionally produced. A missing
or incomplete report must never block a cycle, experiment plan, or submission.
Use `$beamer-synthesis` afterward when a slide companion is useful.
