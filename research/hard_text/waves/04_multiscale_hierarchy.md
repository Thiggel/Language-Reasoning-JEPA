# Wave 04 - multiscale token-to-span hierarchy

Status: active design and implementation stage beginning 2026-07-14.

## Initial experiment matrix

| Axis | Cells |
|---|---|
| Level-1 fixed span | 4, 6, 8, 10 tokens |
| Level-1 macro dim | 8, 16, 32, 64 |
| Level-2 fixed span | 20, 24, 30 tokens |
| Level-3 fixed span | 64, 96, 128 tokens |
| Macro distribution | deterministic+density; Gaussian q/p |
| Planner | discrete text-span; prior shooting; prior-CEM |
| High horizon | 1, 2, 4 |
| Energy | oracle terminal latent; query-conditioned learned value |

Only Level-1 span/dimension screening is launched first. Later rows are gated
on healthy direct long-horizon prediction, macro support, and subgoal
reachability.

No result from this wave is complete yet.

## Active launch - 2026-07-14 09:47 CEST

These are Level-1 screens only. All use token transitions at Level 0,
ordered projected concatenation for each Level-1 action span, a shared latent
state space, high-level value distillation, and fixed EMA/VICReg base
training. The only probabilistic ingredient in `var` cells is the macro-action
encoder/prior. Each run chains direct-high versus recursive-low prediction,
prior support/shooting, and discrete observed-span retrieval audits.

| Runs | Span | Macro dim | Macro model | GPUs |
|---|---:|---:|---|---|
| `text_hier_span4_d32_{det,var}` | 4 | 32 | det / Gaussian | gruenau11:2,3 |
| `text_hier_span6_d32_{det,var}` | 6 | 32 | det / Gaussian | gruenau10:1,2 |
| `text_hier_span8_d32_{det,var}` | 8 | 32 | det / Gaussian | gruenau9:0, gruenau7:1 |
| `text_hier_span10_d32_{det,var}` | 10 | 32 | det / Gaussian | gruenau7:2,3 |
| `text_hier_span8_d8_{det,var}` | 8 | 8 | det / Gaussian | gruenau1:0,1 |
| `text_hier_span8_d16_{det,var}` | 8 | 16 | det / Gaussian | gruenau1:2, gruenau2:0 |
| `text_hier_span8_d64_{det,var}` | 8 | 64 | det / Gaussian | gruenau2:1,2 |

The discrete text-span and top-down free-macro interfaces are represented in
the audit, but full planner comparisons are gated on a Level-1 model passing
the prediction/support/reachability checks. Level 2/3 remain staged, not
silently omitted.
