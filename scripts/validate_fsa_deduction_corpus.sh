#!/usr/bin/env bash
# Data-side admission checks for the FSA logical-deduction corpus.
#
# Reuses scripts/validate_intent_reasoning_data.py (the same schema, catalogue,
# replay, goal and disjointness gate that the compiled-domain admission gate
# runs as its step 1) once for the in-distribution triple and once per OOD
# band, then adds a deterministic regeneration check: the corpus is rebuilt
# from its recorded seeds and compared byte-for-byte.
#
# Usage: scripts/validate_fsa_deduction_corpus.sh <python> <data_root> <out_dir>
set -euo pipefail

py=${1:?python executable}
data_root=${2:?directory holding the compiled bands}
out_dir=${3:?report directory}
root=${TEXTJEPA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
replay_limit=${REPLAY_LIMIT:-0}
determinism_episodes=${DETERMINISM_EPISODES:-25}
mkdir -p "$out_dir"

"$py" "$root/scripts/validate_intent_reasoning_data.py" \
  --domain fsa-deduction \
  --dataset "train=$data_root/train.jsonl" \
  --dataset "val=$data_root/val.jsonl" \
  --dataset "test=$data_root/test_id.jsonl" \
  --replay-limit "$replay_limit" \
  --out "$out_dir/validation_id.json"

for band in test_ood_27_32 test_ood_33_40 test_ood_41_50; do
  "$py" "$root/scripts/validate_intent_reasoning_data.py" \
    --domain fsa-deduction \
    --dataset "train=$data_root/train.jsonl" \
    --dataset "val=$data_root/val.jsonl" \
    --dataset "test=$data_root/$band.jsonl" \
    --replay-limit "$replay_limit" \
    --out "$out_dir/validation_$band.json"
done

PYTHONPATH="$root/src:${PYTHONPATH:-}" "$py" \
  "$root/scripts/check_fsa_deduction_determinism.py" \
  --root "$data_root" --episodes "$determinism_episodes" \
  --out "$out_dir/determinism.json"

"$py" - "$out_dir" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
summary = {"bands": {}}
for path in sorted(out.glob("validation_*.json")):
    payload = json.loads(path.read_text())
    summary["bands"][path.stem.replace("validation_", "")] = payload["splits"]
summary["determinism"] = json.loads((out / "determinism.json").read_text())
(out / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
