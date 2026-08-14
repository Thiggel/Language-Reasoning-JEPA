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
- `2026-08-12-qwen-stage1-lora-screen-v1`: first adapted-backbone round.
  20M tokens at context 1024 on WikiText-103 articles, BF16, rank-16 LoRA in
  layers 13–24 with layers 1–12 and the target frozen; five token-matched cells
  (full, ntp_only, no_action, action_only, full at `λ_scale = 0.1`) sharing one
  token-block file so the NTP-only arm isolates the auxiliary objective.
- `2026-08-12-qwen-stage1-pressure-v1`: prediction-weight ladder (0.3,
  1.0, 3.0) and predictor-projection bottleneck ladder (32, 8) at 20M
  tokens each, anchored on the screen's `full-scale0.1` corner; tests
  whether any auxiliary pressure reaches the backbone.
- `2026-08-13-qwen-stage1-objective-balance-v1`: six cells. Explicit
  `ntp_weight` at 0.01 and 0, direct full finetuning of layers 13-24, a
  higher backbone learning rate, and one 80M-token cell. Established that
  the earlier rounds never left next-token dominance, and that longer
  training improves the transition and then plateaus.
- `2026-08-13-qwen-stage1-fineweb-scale-v1`: 300M tokens at global batch
  65,536 on FineWeb-Edu, with the NTP-only control re-measured on the new
  corpus. Includes the fresh frozen-checkpoint sufficiency probes and the
  untrained open-loop rollout evaluation.
- `2026-08-13-qwen-stage2-curriculum-v1`: horizon 1/4/8/16/32 chained
  curriculum from both FineWeb checkpoints, under-budgeted at roughly 7M
  tokens against the protocol's 60M.
- `2026-08-14-qwen-stage3a-oracle-geometry-v1`: 4,800 verified GSM8K
  trajectories from stock Qwen2.5-0.5B-Instruct, plus boundary-state
  collection and the oracle terminal-geometry probe.
- `qwen-stage1-screen-*`: further 20M-token upper-half rank-16 LoRA controls
  (same_layer, nitp, frozen-LM-plus-predictor, equal-FLOP NTP).
- `olmo1b-stage1-main-*`: 100M-token three-seed main comparison.
- `olmo1b-stage2-h{1,4,8,16,32}-*`: recurrent rollout curriculum.
- `olmo1b-stage3-oracle-*`: explicitly oracle terminal-state probes.
- `olmo1b-stage3-goal-*`: prompt-conditioned metric and direct-value models.
- `olmo1b-stage3-exact-beam-*`: exact-state planning gate.
- `olmo1b-stage3-jump-beam-*`: predicted-state planning after the exact gate.

Every cell uses `runs/autonomy/predictive_state/<round-id>/<job-id>/` with
`job.sh`, `state`, `stdout.log`, `stderr.log`, `exit_code`, a resolved config,
and result JSONs. Run names are seed-qualified and immutable.
