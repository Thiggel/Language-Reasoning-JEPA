"""Fidelity gate for the faithful-iGSM adapter.

For N problems: run an oracle episode through OUR FaithfulEnv (their
dependency graph, our step rendering) and feed the resulting full
solution string to the OFFICIAL checker ``true_correct``. Also verify
our per-parameter values against their answer. 100% pass required.

    .venv/bin/python scripts/validate_faithful.py [N]
"""

from __future__ import annotations

import sys

from textjepa.data.faithful import FaithfulEnv, FaithfulProblem, gen_problem


def main(n: int = 50) -> None:
    from tools.tools_test import true_correct

    ok = ans_ok = 0
    for i in range(n):
        gen = gen_problem(f"val:{i}", 15, 20, (2, 15))
        fp = FaithfulProblem(gen)
        env = FaithfulEnv(fp)
        sents = []
        while not env.solved:
            nec = [q for q in env.feasible_actions() if q in fp.necessary]
            sents.append(env.step(sorted(nec)[0]))
        sol = " " + " ".join(sents)
        correct, my_print, _ = true_correct(sol, gen.problem)
        ok += int(bool(correct))
        ans_ok += int(fp.values[fp.query] == fp.answer)
        if not correct and ok + 5 > i:  # print first few failures
            print(f"--- FAIL {i}:\n{sol}\n")
            my_print.display()
    print(f"official-checker pass: {ok}/{n}; answer consistency: {ans_ok}/{n}")
    if ok < n or ans_ok < n:
        sys.exit(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
