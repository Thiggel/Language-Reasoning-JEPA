# Reading and steering autonomous TextJEPA research

## What “autonomous” means

The system can independently inspect completed results, update scientific
memory, choose one bounded next question, implement it, validate its report and
code, and—after you enable the final gate—submit the next small experiment
round. It can poll jobs and continue after terminals disconnect.

Autonomy does not mean that Codex owns the research goal. You still decide what
is scientifically important, how much evidence is enough, which risks are
acceptable, how GPUs are divided among projects, and when work may scale toward
a paper. The system is designed to continue for a small number of unread
cycles, then stop admitting new work until you catch up.

## Your normal check-in

Two or three times per week—or whenever the dashboard shows a decision-ready
report—use this routine:

1. Press **Read next unread report**. Reports open oldest-first so the argument
   develops in order.
2. Read the one-sentence answer and everyday explanation. If these are not
   genuinely understandable, send that as steering before judging the metric.
3. Inspect **What a fair comparison means here**. Check whether both systems
   received the same information, tuning care, and evaluation opportunities.
4. Inspect the figure and table. Their captions must say what to notice and why
   it matters.
5. Read **What we cannot conclude**. This is often more important than the
   headline result.
6. Answer one or more steering questions in the form at the end. Be explicit
   about priorities, for example: “Prefer a strong falsification test over a
   score improvement,” or “Give hard text twice the baseline GPU budget.”
7. Mark the report as read. The dashboard sends a read receipt to the server,
   releasing the human-review guard when enough reports have been reviewed.

Steering affects the next not-yet-admitted decision. It does not rewrite a
completed observation and does not automatically cancel a running job.

## Recommended autonomy levels

| Level | What Codex does | What you do | Use when |
|---|---|---|---|
| 0 — Observe | Inventory, report, and draft plans | Submit every job manually | Initial backend bring-up |
| 1 — Assist | Analyze, implement, test, and commit | Review and submit each plan | First scientific cycles |
| 2 — Bounded autonomy | Automatically submit small rounds within budgets | Read reports and steer regularly | After failure and retrieval drills pass |
| 3 — Scale proposal | Prepare a paper-scale design but do not submit | Approve protocol, budget, and claims | Only after the mechanism is trustworthy |

Keep paper-scale training, multi-node work, new datasets, major budget changes,
publication, and destructive cleanup human-approved at every level.

## Installing the reader on your computer

```bash
mkdir -p ~/TextJEPA-Research-UI
rsync -a \
  laitenbf@gruenau1.informatik.hu-berlin.de:/vol/home-vol2/ml/laitenbf/TextJEPA/ui/ \
  ~/TextJEPA-Research-UI/
cd ~/TextJEPA-Research-UI
python3 server.py \
  --remote laitenbf@gruenau1.informatik.hu-berlin.de
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. The
reader synchronizes compact reports and controller status every minute. It
does not download datasets, checkpoints, complete source snapshots, or raw run
directories. It listens only on your computer's localhost interface.

The left sidebar filters `intent_phrase`, `token_igsm`, `sequence_edit`, and
shared infrastructure reports. Read oldest unread first. Sending steering from
a report automatically targets that report's project. Marking it read creates
a content-hash receipt; if the report changes, it becomes unread again.

## Approving work and checking allocation

```bash
automation/bin/researchctl allocation
automation/bin/researchctl status --project intent_phrase
automation/bin/researchctl validate-plan --project intent_phrase
automation/bin/researchctl finalize-plan --project intent_phrase
automation/bin/researchctl submit-plan .researchctl/plans/<round>.resolved.json
# Add --execute only after reading the plan and deciding to admit it.
```

Automatic analysis and plan drafting are enabled, but plan submission remains
manual. A later change to `auto_submit_after_wake=true` is a human policy
decision and should follow several successful observation/retrieval/report
cycles. Per-project manifests must also authorize autonomous submission.

## If you want the system to stop immediately

```bash
touch research/STOP
automation/bin/researchctl pause "human steering checkpoint"
```

This prevents new controller work. Existing accepted jobs continue until you
explicitly identify and cancel them.
