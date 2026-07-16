# Intent-phrase log index

The exact console logs are the `*.log` files in this directory. Naming maps
to waves as follows:

| Pattern | Wave |
|---|---|
| `disc_base*`, `disc_combo*`, `disc_chunkpred*`, `lm_*`, `sentlm_*` | 00-01 |
| `disc_straight*`, `disc_mono*`, `disc_rank*`, `disc_mdr*` | 02 |
| `disc_georank*`, `disc_cf*`, `disc_gar*`, `disc_latent_goal*` | 03-04 |
| `real_*`, `lm_intent_faithful*`, `sentlm_*faithful*` | 05 |
| `dldad_*`, `dvldad_*`, `dvjepa_*` | 06 |
| `audit*`, `probe*`, `emergence*` | 07 |
| `hier*`, `intent_hier_*`, `intent_hi*`, `intent_lo*`, `intent_controller*` | 04 and 08-11 |

Interrupted logs remain intentionally present. Absence of `DONE` in the
corresponding `runs/<name>/` directory means a chain was stopped before its
full training/evaluation bundle completed.
