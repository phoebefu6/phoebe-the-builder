"""Generate demo.ipynb.

Plumbing copied from `golden-set-builder` (Day 186), content written fresh. The engine is EXTRACTED from
gate.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The gate numbers are exact (per-case binomials, Poisson-binomial counts), so the notebook reproduces them to the digit.
Only the Monte Carlo parts run at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/prompt-regression-ci"
NB_REPS = 4000


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER.

    Tuple targets are walked, because an extractor reading only `targets[0].id` drops them
    silently - the Day 178 defect.
    """
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("gate.py")

EXPORTS = [
    "ALPHA_GATE", "N_STABLE_PASS", "N_STABLE_FAIL", "N_FLAKY", "N", "STABLE_P", "Q_RUNS", "N_BREAK", "PRS_PER_MONTH",
    "SEED", "RULES", "SCENARIOS", "MC_REPS", "held_out_set", "scenario", "reject_table", "flag_prob", "not_quarantined",
    "count_pmf", "red_prob", "threshold", "evaluate", "simulate", "wilson", "calibrate", "audit",
]

HEADER = """from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
"""

cells: List[Dict[str, Any]] = []
_N = 0


def _lines(src: str) -> List[str]:
    """nbformat wants each source line to KEEP its trailing newline."""
    return src.strip("\n").splitlines(keepends=True)


def _nid() -> str:
    global _N
    _N += 1
    return f"c{_N:02d}"


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _lines(src)})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "id": _nid(), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _lines(src)})


# ---------------------------------------------------------------------------

md(f"""
# Did the prompt change break something, or is that the noise?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A pull request edits a prompt. CI runs a 300-case held-out set under the old and new prompt and posts
a behavioural diff: the cases that passed before and fail now. An LLM does not give the same answer
twice, so that list is never empty - even when nothing changed. This notebook computes, exactly, how
often each gating rule goes red for nothing, what threshold makes it honest, and which real breaks
each rule can and cannot see.

1. Calibration - exact probabilities vs raw simulated runs
2. A no-op PR under six rules
3. Three kinds of real break
4. Negative result - quarantine goes blind where the set is flaky
5. Try your own
""")

md("""
## The engine

Extracted from `gate.py` at build time, character for character. Each case has a per-run pass
probability under each prompt and runs are independent, so the chance that a rule lists a case as broken
is a sum of binomial pmf products, and the number of listed cases is a Poisson-binomial computed by exact
convolution. Nothing below is sampled except the calibration check.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every gate number here is exact and matches `evidence.txt` to the digit.
Only the calibration's raw simulation runs at {NB_REPS:,} replicates instead of the study's 20,000.
""")

md("## 1. Calibration")

code(f"""
cal = calibrate({NB_REPS})
print(f"Poisson-binomial vs scipy binom: max gap {{cal['poisson_binomial_vs_binom_gap']:.1e}}")
for r in cal["red_rates"]:
    print(f"{{r['rule']:<27}} {{r['scenario']:<20}} T={{r['T']:>2}} exact {{r['exact']:.4f}} MC {{r['mc']:.4f}} "
          f"{{'inside' if r['inside_99'] else 'OUTSIDE'}}")
""")

md("## 2. A no-op PR - the prompt did not change")

code("""
pi_a = held_out_set()
ev = evaluate(pi_a)
for r in ev:
    print(f"{r['rule']:<27} phantom breaks {r['noop_expected_flags']:5.2f}   red at T=1 {r['noop_red_at_T1']:.3f}   "
          f"calibrated T {r['T']:>2}   red at T {r['noop_red_at_T']:.3f}   months with a false red {r['month_any_false_red_T']:.2f}")
""")

md("""
The one-run diff lists 7.8 "broken" cases on a PR that changed nothing, so a zero-tolerance gate is red on
essentially every PR. Most of that comes from the 30 flaky cases (a pass probability near 0.5 flips
half the time); the 246 stable cases still add about 1.2 between them. Holding the false-red rate under
5% needs a threshold of 13 cases for the naive diff - and even then, at 40 PRs a month, some PR goes red
for nothing in 78% of months.
""")

md("## 3. Three kinds of real break, at each rule's calibrated threshold")

code("""
for sc in SCENARIOS[1:]:
    print(sc)
    for r in ev:
        x = r[sc]
        print(f"   {r['rule']:<27} P(red) {x['red_at_T']:.3f}   recall {x['recall']:.3f}   precision {x['precision']:.3f}"
              f"   calls {r['calls_per_pr']:,}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for name, col in (("naive k=1", "#c8562b"), ("quarantine + majority k=3", "#2e8b57"), ("fisher k=5", "#2f6fdb")):
    lab, kind, k, quar = next(x for x in RULES if x[0] == name)
    axes[0].plot(count_pmf(flag_prob(pi_a, pi_a, kind, k, quar))[:21], "o-", color=col, ms=3, label=name)
axes[0].set_xlabel("cases listed as broken on a no-op PR"); axes[0].set_ylabel("probability"); axes[0].legend(frameon=False)
axes[0].set_title("Phantom breaks", loc="left")
xs = np.arange(3)
for j, r in enumerate(ev):
    axes[1].bar(xs + (j - 2.5) * 0.13, [r[sc]["red_at_T"] for sc in SCENARIOS[1:]], 0.12, label=r["rule"])
axes[1].set_xticks(xs); axes[1].set_xticklabels(SCENARIOS[1:]); axes[1].set_ylabel("P(red) at calibrated T")
axes[1].legend(frameon=False, fontsize=7); axes[1].set_title("Real breaks", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
Fisher's exact test on 5 runs each catches a hard break every time and 97% of the cases it lists are real,
at five times the model calls of one run. But it is the weakest rule on an *intermittent* break (a case
that now passes half the time): with 5 runs, 5-of-5 against 2-of-5 is not significant, so recall drops to
0.18. Majority-of-k has the opposite profile. No rule wins on all three kinds of break.

A side result pinned by a test: one-sided Fisher at k = 3 can never fire, because 3/3 vs 0/3 gives
p = 0.05 exactly. Choose k >= 4 before choosing Fisher.
""")

md("## 4. Negative result - quarantine")

code("""
q, m = next(r for r in ev if r["rule"] == "quarantine + majority k=3"), next(r for r in ev if r["rule"] == "majority k=3")
print(f"cases gated after quarantine: {q['gated_cases']:.1f} of {N}")
for sc in SCENARIOS[1:]:
    print(f"{sc:<22} majority k=3 {m[sc]['red_at_T']:.3f}  ->  with quarantine {q[sc]['red_at_T']:.3f}")
""")

md("""
Quarantining cases whose baseline runs disagree is the standard fix for flaky tests, and here it does
what it promises: phantom breaks fall from 5.6 to 1.1, the threshold from 10 to 4, and power on stable
breaks rises to 1.00 and 0.63. The price is that a break on a flaky case is caught 7% of the time. The
gate is quiet because it has stopped looking at 28 of the 300 cases.

## Summary

| claim | what the numbers say |
|---|---|
| "zero regressions allowed" | a no-op PR is red 100% of the time on one run, 19% even with Fisher on 5 |
| "the diff report lists what broke" | on a real 6-case break, 43% of the naive list is real |
| "just re-run it more" | k = 5 majority: 55% precision at 5x the calls; Fisher 97%, but 0.18 recall on intermittent breaks |
| "quarantine the flaky cases" | quiet and powerful on stable cases, 0.07 on a break among the flaky ones |
""")

md("## 5. Try your own")

code("""
# baseline  = [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]]   # cases x runs, 0/1
# candidate = [[0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]]
# res = audit(baseline, candidate)
# print("naive:", res["naive"], " majority:", res["majority"], " flaky:", res["flaky"],
#       f" no-op floor {res['noop_floor_naive']:.2f}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own baseline and candidate runs:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`golden-set-builder`](../golden-set-builder) asks whether the held-out set still
represents traffic; [`prompt-registry`](../prompt-registry) versions the prompt text;
[`prompt-linter`](../prompt-linter) reviews it statically.
""")


def write() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open("demo.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote demo.ipynb with {len(cells)} cells "
          f"({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
