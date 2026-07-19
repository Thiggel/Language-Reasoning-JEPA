# Evidence

- The non-symbolic faithful token-edit data and model path passes local end-to-end smoke tests.
- The first full-scale round is invalid because it failed before training.
- Replay checkpoints from seeds 0 and 1 produce positive, target-free current-buffer
  GAR normalized edit-distance improvement at horizon 2 on 128 test episodes with
  256 candidates: `0.05048` and `0.04876`, respectively. The matched random policy
  is strongly negative.
- At the identical protocol, horizon 3 falls to `0.03118` and `0.03208`. The third
  selected edit has negative mean true advantage for both checkpoints (`-0.12500`
  and `-0.07813`). This satisfies the predeclared rule for adopting horizon 2 rather
  than horizon 3.
- These are deployment-feasible candidate and scoring results: targets are used for
  evaluation only. The separately labelled oracle-injected condition is
  candidate-privileged and is not evidence for deployable performance.
- Exact recovery remains zero at these short horizons; the supported claim is modest
  edit-distance improvement, not solved trajectories.
