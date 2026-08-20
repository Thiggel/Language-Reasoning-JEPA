# Round notes

## 2026-08-19 orphaned-watcher incident (epoch0 discarded)

The superseded 20:38 launch (snapshot 6c6a39a, appended-junk prefix variant)
was stopped by killing its `timeout` wrapper and python trainer, but NOT its
watcher subshell. Its `train_finished` sentinel moved with the run directory
when the cell was renamed to `*-superseded-lengthcue`, so the orphan watcher's
loop condition never became true. It kept polling the ORIGINAL path string --
which had since been recreated as the live cell -- and wrote
`watch/epoch0/plan_*.json` there using the OLD code. The live watcher's
`[ -f ... ] && continue` guard then skipped those files.

Consequence: the epoch0 `plan_full_catalogue_d1.json` reported at 19:01 UTC
(.770 / .780) is NOT trustworthy -- it may have been produced by the
superseded objective. Orphans killed at 19:50 UTC (pids 242171 242174 242225
242228, evals 17634 17639); epoch0 watch dirs, curve.jsonl and eval logs were
deleted on both live cells. Training was never affected (verified running from
snapshot c8115c9 throughout, GPUs at 100%).

epoch0 is therefore PARTIAL and should be ignored. Every watcher pass from
epoch 2 onward is clean and complete.

LESSON for future relaunches into a reused round: kill the watcher subshell
explicitly (or `touch train_finished` in the path the orphan still holds),
never only the trainer.

## epoch0 backfill (second-order effect of the same incident)

`rm -rf watch/epoch0` at 19:50 UTC ran while the live watcher's epoch-0 pass
was mid-flight. `plan_flat.py` had already opened its `--out` file, so the
depth-4 result it finished writing at ~20:30 landed in an unlinked inode and
vanished, along with the deleted `probe_auc.json` and the depth-1 file. The
pass itself was never restarted -- it simply continued to depth 8, which
writes normally because that process opened its output after the delete.

Backfilled after the pass completed, from `.epoch0_ck.pt` (a byte-copy of the
epoch-0 checkpoint, preserved before the next poll could overwrite
`.probe_ck.pt`) and from snapshot c8115c9: full_catalogue depth 1 and depth 4.
epoch0 is a step-500 checkpoint and is a sanity row only; epoch 2 onward is
unaffected and complete.

## depth 8 removed from the watcher (cost, not policy)

Under the NEW eval defaults (energy-guided beam expansion + `--aggregate
mean_prefix`) a depth-8 run costs ~18 h per 100 episodes: measured 10/100
episodes in 1 h 48 m on an L40. That would have blocked every subsequent
watcher pass, so epoch-2 numbers would never have arrived. Depth 8 was killed
and sentinel JSONs were placed at
`watch/epoch{0,2,4,6,8,10}/plan_full_catalogue_d8.json` so the watcher's
`[ -f ] && continue` guard skips it; the curve summariser tolerates the empty
`latent_planner` block. The same was done for depth 8 and 16 in `final_id`
and depth 16 in `final_ood` (300 episodes there -- even more expensive).

The watcher therefore reports full_catalogue depth 1/4 and prior_propose
depth 1/4. That still answers the question the round was launched for ("does
depth help?"), which needs depth 1 next to depth 4.

NOTE this is itself a finding: the eval-time fixes from the 08-20 diagnosis
made deep search far more expensive than the old random-tail expansion. If a
depth-8 or depth-16 number is wanted for the paper it needs its own dedicated
long-running cell, not a training-time watcher.
