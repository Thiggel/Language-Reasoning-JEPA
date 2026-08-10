# Experiment index

## Planned run families

- `2026-08-10-qwen-frozen-transition-diagnostic-v1`: frozen Qwen full and
  no-action cells; excluded parser failures before model execution.
- `2026-08-10-qwen-frozen-transition-diagnostic-v2`: excluded FP16 optimizer
  failures at the first update.
- `2026-08-10-qwen-frozen-transition-diagnostic-v3`: completed full cell;
  smaller no-action cell retained only as an operational precursor.
- `2026-08-10-qwen-frozen-transition-diagnostic-v4`: completed
  capacity-matched no-action control; this is the admitted comparison cell.
- `2026-08-10-qwen-frozen-transition-diagnostic-v5`: corrected independent
  scale weighting, packed document attention, full versus equal-capacity
  action-only; both completed, positive direction evidence and unresolved
  full-state scale mismatch.
- `qwen-stage1-screen-*`: 20M-token upper-half rank-16 LoRA controls.
- `olmo1b-stage1-main-*`: 100M-token three-seed main comparison.
- `olmo1b-stage2-h{1,4,8,16,32}-*`: recurrent rollout curriculum.
- `olmo1b-stage3-oracle-*`: explicitly oracle terminal-state probes.
- `olmo1b-stage3-goal-*`: prompt-conditioned metric and direct-value models.
- `olmo1b-stage3-exact-beam-*`: exact-state planning gate.
- `olmo1b-stage3-jump-beam-*`: predicted-state planning after the exact gate.

Every cell uses `runs/autonomy/predictive_state/<round-id>/<job-id>/` with
`job.sh`, `state`, `stdout.log`, `stderr.log`, `exit_code`, a resolved config,
and result JSONs. Run names are seed-qualified and immutable.
