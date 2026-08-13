"""Re-derive compiled FSA episodes from their manifest seeds and compare.

The generator is a pure function of ``(seed, index, depth, knobs)``, so a
compiled corpus must be reproducible exactly.  This walks each band, re-runs
the sampler for the recorded episode ids, recompiles them with the manifest's
knobs, and checks the serialised episodes are identical.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

from textjepa.data.fsa_deduction import compile_fsa_episode, sample_fsa_problem

_ID = re.compile(r"^fsa-deduction-s(\d+)-d(\d+)-k(\d+)-i(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.root / "MANIFEST.json").read_text())
    design = manifest["design"]
    report = {"episodes_per_band": args.episodes, "bands": {}}
    for band in sorted(manifest["splits"]):
        path = args.root / f"{band}.jsonl"
        checked = 0
        with path.open() as handle:
            for line in handle:
                if checked >= args.episodes:
                    break
                record = json.loads(line)
                match = _ID.match(record["episode_id"])
                if match is None:
                    raise RuntimeError(f"unparsable id {record['episode_id']}")
                seed, depth, branching, index = (int(v) for v in match.groups())
                problem = sample_fsa_problem(
                    seed=seed, index=index, depth=depth,
                    branching_factor=branching,
                    side_facts_per_step=int(design["side_facts_per_step"]),
                )
                rebuilt = asdict(compile_fsa_episode(
                    problem, record["split"],
                    teacher_horizon=int(design["teacher_horizon"]),
                    counterfactual_k=int(design["counterfactual_k"]),
                    catalogue_cap=int(design["catalogue_cap"]),
                ))
                if json.dumps(rebuilt, sort_keys=True) != json.dumps(
                    record, sort_keys=True
                ):
                    raise RuntimeError(
                        f"non-deterministic replay for {record['episode_id']}"
                    )
                checked += 1
        report["bands"][band] = {"checked": checked, "identical": checked}
    report["deterministic"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
